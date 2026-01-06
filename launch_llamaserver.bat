@REM llamaserver launch script for Windows

@REM # Prequisites:
@REM # 1. Download llamacpp binary for your OS (win cpu x64 in this case) from: https://github.com/ggml-org/llama.cpp/releases/download/b7601/llama-b7601-bin-win-cpu-x64.zip and unzip it to ./llama-b7601-bin-win-cpu-x64
@REM # 2. Download Vocaela GGUF model files from: 
@REM #   - For Vocaela-2-500M-1024R2-GGUF: https://huggingface.co/vocaela/Vocaela-2-500M-1024R2-GGUF
@REM #   - For Vocaela-500M-GGUF: https://huggingface.co/vocaela/Vocaela-500M-GGUF
@REM #   - For Vocaela-2-256M-512R2-GGUF: TBD, LlamaCpp converted model cannot exactly match HF model, in investigation.

set llamaserver_exe_dir=.\\llama-b7601-bin-win-cpu-x64

@REM For Vocaela-2-500M-1024R2-GGUF
set gguf_model_path=.\\Vocaela-2-500M-1024R2-GGUF\\Vocaela-2-500M-1024R2-Q8_0.gguf
set gguf_mmproj_path=.\\Vocaela-2-500M-1024R2-GGUF\\mmproj-Vocaela-2-500M-1024R2-Q8_0.gguf

@REM @REM For Vocaela-500M-GGUF
@REM set gguf_model_path=.\\Vocaela-500M-GGUF\\Vocaela-500M-Q8_0.gguf
@REM set gguf_mmproj_path=.\\Vocaela-500M-GGUF\\mmproj-Vocaela-500M-Q8_0.gguf

@REM Convert to absolute paths
for %%I in ("%gguf_model_path%") do set "gguf_model_path=%%~fI"
for %%I in ("%gguf_mmproj_path%") do set "gguf_mmproj_path=%%~fI"

cd /d %llamaserver_exe_dir%
@REM parallelism set as -t 4, adjust according to your CPU cores
llama-server.exe -m "%gguf_model_path%" --mmproj "%gguf_mmproj_path%" --port 8081 -c 4096 -t 4