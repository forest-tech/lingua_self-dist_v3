#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

module load cuda
uv run -m apps.self_dist.eval config=apps/self_dist/configs/eval.yaml 
