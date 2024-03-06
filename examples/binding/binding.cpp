#include "binding.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

struct model_ctx {
    llama_model_params model_params;
    llama_context_params ctx_params;
    llama_model *model;

    ~model_ctx() {
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
    }
};

std::mutex locker;
std::map<std::string, model_ctx *> models;

int load_model(const char *model_id, const char *model_path,
               const int32_t gpu_layers, const int split_mode,
               const int32_t main_gpu, const bool vocab_only,
               const bool use_mmap, const bool use_mlock) {
    if (model_path == NULL) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(locker);
    if (models.find(model_id) != models.end()) {
        return 1;
    }

    model_ctx *ctx = new model_ctx();
    ctx->model_params = llama_model_default_params();
    ctx->model_params.n_gpu_layers = gpu_layers;
    ctx->model_params.split_mode = (llama_split_mode)split_mode;
    ctx->model_params.main_gpu = main_gpu;
    ctx->model_params.vocab_only = vocab_only;
    ctx->model_params.use_mmap = use_mmap;
    ctx->model_params.use_mlock = use_mlock;

    ctx->model = llama_load_model_from_file(model_path, ctx->model_params);
    if (ctx->model == NULL) {
        delete ctx;
        return -1;
    }

    ctx->ctx_params = llama_context_default_params();
    models[model_id] = ctx;
    return 0;
}

int unload_model(const char *model_id) {
    std::lock_guard<std::mutex> lock(locker);
    auto it = models.find(model_id);
    if (it == models.end()) {
        return -1;
    }

    llama_free_model(it->second->model);
    delete it->second;
    models.erase(it);
    return 0;
}

model_ctx *get_model(const char *model_id);
model_ctx *get_model(const char *model_id) {
    std::lock_guard<std::mutex> lock(locker);
    auto it = models.find(model_id);
    if (it == models.end()) {
        return NULL;
    }

    return it->second;
}

int set_seed(const char *model_id, int seed) {
    model_ctx *ctx = get_model(model_id);
    if (ctx == NULL) {
        return -1;
    }

    ctx->ctx_params.seed = seed;
    return 0;
}

int set_ctx_size(const char *model_id, int ctx_size) {
    model_ctx *ctx = get_model(model_id);
    if (ctx == NULL) {
        return -1;
    }

    if (ctx_size < 2048) {
        ctx_size = 2048;
    }

    ctx->ctx_params.n_ctx = ctx_size;
    return 0;
}

int set_batch_size(const char *model_id, int batch_size) {
    model_ctx *ctx = get_model(model_id);
    if (ctx == NULL) {
        return -1;
    }

    ctx->ctx_params.n_batch = batch_size;
    return 0;
}

int set_threads(const char *model_id, int threads) {
    model_ctx *ctx = get_model(model_id);
    if (ctx == NULL) {
        return -1;
    }

    ctx->ctx_params.n_threads = threads;
    return 0;
}

int set_threads_batch(const char *model_id, int threads) {
    model_ctx *ctx = get_model(model_id);
    if (ctx == NULL) {
        return -1;
    }

    ctx->ctx_params.n_threads_batch = threads;
    return 0;
}

int llama_test(const char *model_id, const char *prompt) {
    // total length of the sequence including the prompt
    const int n_len = 32;

    model_ctx *ctx = get_model(model_id);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return -1;
    }

    llama_context *llama_ctx =
        llama_new_context_with_model(ctx->model, ctx->ctx_params);

    if (llama_ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n",
                __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(llama_ctx, prompt, true);

    const int n_ctx = llama_n_ctx(llama_ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len,
            n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated
    // tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is "
                "not big enough\n",
                __func__);
        LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(llama_ctx, id).c_str());
    }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, {0}, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(llama_ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_len) {
        // sample the next token
        {
            auto n_vocab = llama_n_vocab(ctx->model);
            auto *logits = llama_get_logits_ith(llama_ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(
                    llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(),
                                                   candidates.size(), false};

            // sample the most likely token
            const llama_token new_token_id =
                llama_sample_token_greedy(llama_ctx, &candidates_p);

            // is it an end of stream?
            if (new_token_id == llama_token_eos(ctx->model) || n_cur == n_len) {
                LOG_TEE("\n");

                break;
            }

            LOG_TEE("%s",
                    llama_token_to_piece(llama_ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, {0}, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(llama_ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__,
                    1);
            return 1;
        }
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__,
            n_decode, (t_main_end - t_main_start) / 1000000.0f,
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(llama_ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(llama_ctx);
    // llama_free_model(ctx->model);

    // llama_backend_free();

    return 0;
}
