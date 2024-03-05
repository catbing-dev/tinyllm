# tinyllm (Forked by [llama.cpp](https://github.com/ggerganov/llama.cpp) )

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Thank you, [ggerganov](https://github.com/ggerganov)! You is simply our deity!**

### How to use

Just run `cmake ..` and build. like this:

```bash
cmake -B build -DBUILD_SHARED_LIBS=1 -DLLAMA_CUBLAS=1 -DLLAMA_CUDA_F16=1 && cmake --build build -j --config Release
```
