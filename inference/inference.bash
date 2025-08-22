#!/bin/bash

# CLI Inference Command
# Replace /workspace/Tacotron-TTS/ft_out/tacotron-ft-v4e-5-250e-August-21-2025_05+18PM-379b5b8/best_model_540.pth
# and /workspace/Tacotron-TTS/ft_out/tacotron-ft-v4e-5-250e-August-21-2025_05+18PM-379b5b8/config.json
# with the actual paths to your fine-tuned model and its configuration file.
tts --model_path /workspace/Tacotron-TTS/ft_out/tacotron-ft-v4e-5-250e-August-21-2025_05+18PM-379b5b8/best_model_540.pth --config_path /workspace/Tacotron-TTS/ft_out/tacotron-ft-v4e-5-250e-August-21-2025_05+18PM-379b5b8/config.json --text "hello world"  --use_cuda
