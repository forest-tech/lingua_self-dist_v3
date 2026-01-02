#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

set -eu

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
NAME_PREFIX=${NAME_PREFIX:-SD_llama-1b_ce}
NAME="${NAME_PREFIX}_${TIME_STAMP}"
DUMP_DIR="output/${NAME}"

STEPS=100
CHECKPOINT_EVERY=100 # デフォで100
CHECKPOINT_STEP=${CHECKPOINT_STEP:-$STEPS}
CHECKPOINT_STEP_PADDED=$(printf "%010d" "$CHECKPOINT_STEP")
CKPT_DIR="${DUMP_DIR}/checkpoints/${CHECKPOINT_STEP_PADDED}"
EVAL_DUMP_DIR="${DUMP_DIR}/evals"
HF_OUTPUT_DIR="${DUMP_DIR}/checkpoints/bin"
LM_EVAL_OUTPUT_DIR="output_eval/${NAME}_${CHECKPOINT_STEP_PADDED}"
BASE_MODEL_OUTPUT_DIR="output_eval/llama-1b_base_${TIME_STAMP}"

TASKS="lambada,wikitext,hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,mmlu"

module load cuda

# Train the self_distillation run.
uv run -m apps.self_dist.train \
    config=apps/self_dist/configs/llama_continual_test.yaml \
    dump_dir="$DUMP_DIR" \
    name="$NAME" \
    steps="$STEPS" \
    checkpoint.dump.every="$CHECKPOINT_EVERY" \
    data.sources.fineweb_edu_10bt_shuffled=0.1 \
    # logging.freq=1 \

mkdir -p "$EVAL_DUMP_DIR"

# Run the built-in evaluation on the Lingua checkpoint.
# uv run -m apps.self_distillation.eval \
#     config=apps/self_distillation/configs/eval.yaml \
#     ckpt_dir="$CKPT_DIR" \
#     dump_dir="$EVAL_DUMP_DIR"

# Convert the Lingua checkpoint to Hugging Face format for LM Eval.
uv run converter/ckpt_lingua_to_hf.py \
    --input_dir "${CKPT_DIR}/consolidated" \
    --output_dir "$HF_OUTPUT_DIR"

mkdir -p "$LM_EVAL_OUTPUT_DIR"

# Evaluate the converted checkpoint with lm-eval-harness.
uv run -m lm_eval \
    --model hf \
    --model_args pretrained="./$HF_OUTPUT_DIR" \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "$LM_EVAL_OUTPUT_DIR" \
    --tasks "$TASKS"

# ベースモデルに対しても同様に評価を実行
# mkdir -p "$BASE_MODEL_OUTPUT_DIR"
# uv run -m lm_eval \
#     --model hf \
#     --model_args pretrained="./model/llama-3.2-1b" \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path "$BASE_MODEL_OUTPUT_DIR" \
#     --tasks "$TASKS"