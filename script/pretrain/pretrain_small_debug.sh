#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

# 日付付きの出力ディレクトリを作成
DUMP_DIR="output/pretrained_small_model_debug"
NAME="pretrained_small_model_debug"

module load cuda
uv run -m apps.main.train config=apps/main/configs/small.yaml dump_dir=$DUMP_DIR name=$NAME \
	steps=10 checkpoint.dump.every=10
