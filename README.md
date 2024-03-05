# tinyllm (fork by llama.cpp)

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Thank you, ggerganov! You is simply our deity!**

### How to use

Just run `cmake ..` and build. like this:

```bash
cmake -B build -DBUILD_SHARED_LIBS=1 -DLLAMA_CUBLAS=1 -DLLAMA_CUDA_F16=1 && cmake --build build -j --config Release
```
