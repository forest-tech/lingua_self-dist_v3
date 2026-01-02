#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

module load cuda


MODEL_DIR="./model/gpt2"
MODEL_BIN_DIR="./model/gpt2_bin"

uv run lm_eval --model hf \
    --model_args pretrained=${MODEL_DIR},trust_remote_code=True \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output_eval \
    --tasks lambada,wikitext 

uv run lm_eval --model hf \
    --model_args pretrained=${MODEL_BIN_DIR},trust_remote_code=True \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output_eval \
    --tasks lambada,wikitext