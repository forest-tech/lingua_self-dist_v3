#!/bin/sh
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -j

module load cuda
uv run -m apps.main.train config=apps/main/configs/debug.yaml
