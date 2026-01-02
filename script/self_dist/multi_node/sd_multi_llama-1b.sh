#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

# 日付付きの出力ディレクトリを作成
DUMP_DIR="output/SD_llama-1b_$(date +%Y%m%d_%H%M%S)"
NAME="SD_llama-1b_$(date +%Y%m%d_%H%M%S)"

module load cuda

# if you want to launch locally you can use torchrun
uv run -m torch.distributed.run --nproc-per-node 4 -m apps.self_dist.train config=apps/self_dist/configs/llama-1b.yaml \
    dump_dir=$DUMP_DIR \
    name=$NAME \
    steps=10 \
    logging.freq=1 \
    probe_freq=1
