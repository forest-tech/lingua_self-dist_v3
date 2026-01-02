#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=5:29:00
#PJM -j

# 日付付きの出力ディレクトリを作成
DUMP_DIR="output/SD_lossSearch_llama_$(date +%Y%m%d_%H%M%S)"
NAME="SD_lossSearch_llama_$(date +%Y%m%d_%H%M%S)"

module load cuda
uv run -m apps.self_distillation.train_lossSearch config=apps/self_distillation/configs/llama_lossSearch.yaml dump_dir=$DUMP_DIR name=$NAME
