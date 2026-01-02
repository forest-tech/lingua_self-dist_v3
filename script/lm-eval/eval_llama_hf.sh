#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

module load cuda

uv run lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output_eval \
    --tasks lambada,wikitext 
