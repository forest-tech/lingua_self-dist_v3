#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

set -eu

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
NAME_PREFIX=${NAME_PREFIX:-SD_llama-1b_ce}
NAME="${NAME_PREFIX}_${TIME_STAMP}"
DUMP_DIR="outputs/${NAME}"

STEPS=100
CHECKPOINT_EVERY=100 # デフォで100
CHECKPOINT_STEP=${CHECKPOINT_STEP:-$STEPS}
CHECKPOINT_STEP_PADDED=$(printf "%010d" "$CHECKPOINT_STEP")
CKPT_DIR="${DUMP_DIR}/checkpoints/${CHECKPOINT_STEP_PADDED}"
EVAL_DUMP_DIR="${DUMP_DIR}/evals"
HF_OUTPUT_DIR="${DUMP_DIR}/checkpoints/bin"
LM_EVAL_OUTPUT_DIR="outputs_eval/${NAME}_${CHECKPOINT_STEP_PADDED}"
BASE_MODEL_OUTPUT_DIR="outputs_eval/llama-1b_base_${TIME_STAMP}"

TASKS="lambada,wikitext,hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,mmlu"

module load cuda

# Train the self_distillation run.
uv run  -m torch.distributed.run --nproc-per-node 4 -m apps.self_distillation.train_ce \
    config=apps/self_distillation/configs/llama-1b_convert.yaml \
    dump_dir="$DUMP_DIR" \
    name="$NAME" \
    steps="$STEPS" \
    checkpoint.dump.every="$CHECKPOINT_EVERY" \
    data.sources.fineweb_edu_10bt_shuffled=0.01 \
    # logging.freq=1 \