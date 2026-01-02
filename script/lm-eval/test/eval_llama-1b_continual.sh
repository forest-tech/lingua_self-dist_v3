#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

TASKS="lambada,wikitext"

module load cuda
uv run -m lm_eval \
    --model hf \
    --model_args pretrained='output/llama_3.2_1b_continual/checkpoints/bin' \
    --device cuda:0 \
    --batch_size 8 \
    --output_path 'output_eval/llama_3.2_1b_continual/' \
    --tasks "$TASKS"