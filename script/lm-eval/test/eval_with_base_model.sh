#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=1:29:00
#PJM -j

set -eu

RUN_NAME="SD_llama-1b_ce_20251210_142358"
CHECKPOINT_STEP="300"
CHECKPOINT_STEP_PADDED=$(printf "%010d" "$CHECKPOINT_STEP")
INPUT_DIR="output/${RUN_NAME}/checkpoints/${CHECKPOINT_STEP_PADDED}/consolidated"
OUTPUT_DIR="output/${RUN_NAME}/checkpoints/bin"
EVAL_OUTPUT_DIR="output_eval/${RUN_NAME}_${CHECKPOINT_STEP_PADDED}"
HF_OUTPUT_DIR="$OUTPUT_DIR"
LM_EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR"
BASE_MODEL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}_base_model"

TASKS="lambada,wikitext,hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,mmlu"

module load cuda

# Convert the specified Lingua checkpoint to Hugging Face format.
uv run converter/ckpt_lingua_to_hf.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

mkdir -p "$EVAL_OUTPUT_DIR"

# Evaluate the converted checkpoint with lm-eval-harness.
uv run -m lm_eval \
    --model hf \
    --model_args pretrained="./$HF_OUTPUT_DIR" \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "$LM_EVAL_OUTPUT_DIR" \
    --tasks "$TASKS"

# ベースモデルに対しても同様に評価を実行
mkdir -p "$BASE_MODEL_OUTPUT_DIR"
uv run -m lm_eval \
    --model hf \
    --model_args pretrained="./model/llama-3.2-1b" \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "$BASE_MODEL_OUTPUT_DIR" \
    --tasks "$TASKS"