#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

INPUT_DIR='output/llama_3.2_1b_continual/checkpoints/0000000100/consolidated'
OUTPUT_DIR='output/llama_3.2_1b_continual/checkpoints/bin'

uv run converter/ckpt_lingua_to_hf.py \
	--input_dir $INPUT_DIR \
	--output_dir $OUTPUT_DIR
