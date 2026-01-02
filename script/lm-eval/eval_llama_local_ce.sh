#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

module load cuda

MODEL_DIR="output/SD_llama-1b_ce_20251205_062619/checkpoints/0000000500_hf"
# MODEL_DIR="/home/pj24001974/ku50001532/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

TASKS="lambada hellaswag winogrande arc_easy arc_challenge mmlu"

uv run lm_eval --model hf \
  --model_args pretrained=${MODEL_DIR},trust_remote_code=True \
  --device cuda:0 \
  --batch_size 8 \
  --output_path output_eval \
  --tasks lambada,wikitext
