#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

# 日付付きの出力ディレクトリを作成
# DUMP_DIR="output/SD_llama-1b_$(date +%Y%m%d_%H%M%S)"
# NAME="SD_llama-1b_$(date +%Y%m%d_%H%M%S)"

module load cuda
uv run -m apps.self_dist.train config=apps/self_dist/configs/llama_continual_test.yaml \
    steps=100

