#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

set -eu

log() {
    printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"
}

log "Flow script started"

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
NAME_PREFIX=${NAME_PREFIX:-SD_smallLM_ce}
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
BASE_MODEL_OUTPUT_DIR="output_eval/smallLM-360M_base_${TIME_STAMP}"

TASKS="lambada,wikitext,hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,mmlu"

module load cuda

log "Loaded CUDA module"

# 最初にディレクトリを出力
log "Dump directory: $DUMP_DIR"
log "Checkpoint directory: $CKPT_DIR"
log "Evaluation dump directory: $EVAL_DUMP_DIR"
log "Hugging Face output directory: $HF_OUTPUT_DIR"
log "LM Eval output directory: $LM_EVAL_OUTPUT_DIR"
log "Base model output directory: $BASE_MODEL_OUTPUT_DIR"


# Train the self_distillation run.
log "Starting self_distillation training"
uv run -m apps.self_dist.train \
    config=apps/self_dist/configs/smallLM_continual_debug.yaml \
    dump_dir=$DUMP_DIR \
    name=$NAME \
    steps=$STEPS \
    checkpoint.dump.every=$CHECKPOINT_EVERY \
    data.sources.fineweb_edu_10bt_shuffled=1.0
log "Self-distillation training finished"

mkdir -p "$EVAL_DUMP_DIR"
log "Ensured eval dump directory exists"

# Run the built-in evaluation on the Lingua checkpoint.
# uv run -m apps.self_distillation.eval \
#     config=apps/self_distillation/configs/eval.yaml \
#     ckpt_dir="$CKPT_DIR" \
#     dump_dir="$EVAL_DUMP_DIR"

# Convert the Lingua checkpoint to Hugging Face format for LM Eval.
log "Converting Lingua checkpoint to Hugging Face format"
uv run converter/ckpt_lingua_to_hf.py \
    --input_dir "${CKPT_DIR}/consolidated" \
    --output_dir "$HF_OUTPUT_DIR"
log "Conversion to Hugging Face format finished"

mkdir -p "$LM_EVAL_OUTPUT_DIR"
log "Ensured LM Eval output directory exists"

# Evaluate the converted checkpoint with lm-eval-harness.
log "Starting LM Eval on distilled checkpoint"
uv run -m lm_eval \
    --model hf \
    --model_args pretrained="$HF_OUTPUT_DIR" \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "$LM_EVAL_OUTPUT_DIR" \
    --tasks "$TASKS"
log "LM Eval on distilled checkpoint finished"

# ベースモデルに対しても同様に評価を実行
mkdir -p "$BASE_MODEL_OUTPUT_DIR"
log "Starting LM Eval on base model"
uv run -m lm_eval \
    --model hf \
    --model_args pretrained="./model/smallLM_360M" \
    --device cuda:0 \
    --batch_size 8 \
    --output_path "$BASE_MODEL_OUTPUT_DIR" \
    --tasks "$TASKS"
log "LM Eval on base model finished"

log "Flow script finished"