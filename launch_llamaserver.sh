# !/bin/bash

# llamaserver launch script for ubuntu x64

# Prequisites:
# 1. Download llamacpp binary for your OS from https://github.com/ggml-org/llama.cpp/releases?q=b7601&expanded=true
#   - Ubuntu x64: https://github.com/ggml-org/llama.cpp/releases/download/b7601/llama-b7601-bin-ubuntu-x64.tar.gz
#   - macOS Apple Silicon (arm64): https://github.com/ggml-org/llama.cpp/releases/download/b7601/llama-b7601-bin-macos-arm64.tar.gz
#   - macOS Intel (x64): https://github.com/ggml-org/llama.cpp/releases/download/b7601/llama-b7601-bin-macos-x64.tar.gz
#   Extract it to ./llama-b7601-bin-ubuntu-x64, there should be no intermediate folders after extraction.

# 2. Download Vocaela GGUF model files from: 
#   - For Vocaela-2-500M-1024R2-GGUF: https://huggingface.co/vocaela/Vocaela-2-500M-1024R2-GGUF
#   - For Vocaela-500M-GGUF: https://huggingface.co/vocaela/Vocaela-500M-GGUF
#   - For Vocaela-2-256M-512R2-GGUF: TBD, LlamaCpp converted model cannot exactly match HF model, in investigation.

llamaserver_bin_dir="./llama-b7601-bin-ubuntu-x64/"

# # for Vocaela-2-500M-1024R2 model
# gguf_model_path="./Vocaela-2-500M-1024R2-GGUF/Vocaela-2-500M-1024R2-Q8_0.gguf"
# gguf_mmproj_path="./Vocaela-2-500M-1024R2-GGUF/mmproj-Vocaela-2-500M-1024R2-Q8_0.gguf"

# # for Vocaela-500M model
# gguf_model_path="./Vocaela-500M-GGUF/Vocaela-500M-Q8_0.gguf"
# gguf_mmproj_path="./Vocaela-500M-GGUF/mmproj-Vocaela-500M-Q8_0.gguf"

gguf_model_path=$(realpath $gguf_model_path)
gguf_mmproj_path=$(realpath $gguf_mmproj_path)

cd "$llamaserver_bin_dir"
# parallelism set as -t 4, adjust according to your CPU cores
./llama-server -m "$gguf_model_path" --mmproj "$gguf_mmproj_path" --port 8081 -c 4096 -t 4