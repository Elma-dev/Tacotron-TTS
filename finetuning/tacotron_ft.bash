#!/bin/bash

# Load the model and test it
tts --model_name tts_models/en/mai/tacotron2-DDC --text "HELLO"

# Adjust configuration file
vim /root/.local/share/tts/tts_models--en--ljspeech--tacotron2-DDC/config.json

# Clear GPU cache
python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache()"

# Replace paths with actual paths to your config.json and model.pth
CUDA_VISIBLE_DEVICES="0" python train_tts.py \
--config_path /root/.local/share/tts/tts_models--en--ljspeech--tacotron2-DDC/config.json \
--restore_path /root/.local/share/tts/tts_models--en--ljspeech--tacotron2-DDC/model.pth
