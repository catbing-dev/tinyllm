#include "llama.h"

extern "C" {
int set_seed(const char *model_id, int seed);
int set_ctx_size(const char *model_id, int ctx_size);
int set_batch_size(const char *model_id, int batch_size);
int set_threads(const char *model_id, int threads);
int set_threads_batch(const char *model_id, int threads);

int load_model(const char *model_id, const char *model_path,
               const int32_t gpu_layers, const int split_mode,
               const int32_t main_gpu, const bool vocab_only,
               const bool use_mmap, const bool use_mlock);
int unload_model(const char *model_id);

int llama_test(const char *model_id, const char *prompt);
}
