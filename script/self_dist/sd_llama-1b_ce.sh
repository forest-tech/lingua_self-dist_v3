#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=10:29:00
#PJM -j

# llamaの教師モデルを用いた自己蒸留学習の実行スクリプト
# loss はクロスエントロピー損失のみ

# 日付付きの出力ディレクトリを作成
DUMP_DIR="output/SD_CE_debug_llama_teacher_$(date +%Y%m%d_%H%M%S)"
NAME="SD_CE_debug_llama_teacher_$(date +%Y%m%d_%H%M%S)"
module load cuda
uv run -m apps.self_dist.train_ce config=apps/self_dist/configs/llama_teacher_model.yaml \
    dump_dir=$DUMP_DIR \
    name=$NAME