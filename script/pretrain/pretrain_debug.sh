#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -j
#PJM -o debug_std.txt

module load cuda
uv run -m apps.main.train config=apps/main/configs/debug.yaml
