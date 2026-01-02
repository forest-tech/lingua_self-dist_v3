#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=0:29:00
#PJM -j

uv run converter/ckpt_pth_to_dcp.py \
    --input_dir ./model/llama-3.2-1b_lingua \
    --output_dir ./model/llama-3.2-1b_dcp 
