#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

set -eu

RUN_NAME="SD_llama-1b_ce_20251210_142358"
CHECKPOINT_STEP="3"
CHECKPOINT_STEP_PADDED=$(printf "%010d" "$CHECKPOINT_STEP")
INPUT_DIR="output/${RUN_NAME}/checkpoints/${CHECKPOINT_STEP_PADDED}/consolidated"
OUTPUT_DIR="output/${RUN_NAME}/checkpoints/bin"
EVAL_OUTPUT_DIR="output_eval/${RUN_NAME}_${CHECKPOINT_STEP_PADDED}"

module load cuda

# Convert the specified Lingua checkpoint to Hugging Face format.
uv run converter/ckpt_lingua_to_hf.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

mkdir -p "$EVAL_OUTPUT_DIR"

# Run LM Eval on the converted checkpoint.
uv run -m lm_eval \
    --model hf \
    --model_args pretrained="./$OUTPUT_DIR" \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "$EVAL_OUTPUT_DIR" \
    --tasks lambada,wikitext