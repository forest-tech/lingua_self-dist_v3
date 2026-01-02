#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

INPUT_DIR='model/qwen3-0.6b'
OUTPUT_DIR='model/qwen3-0.6b_lingua'
OUTPUT_2_DIR='model/qwen3-0.6b_dcp'

uv run converter/ckpt_hf_to_lingua_autoconfig.py \
	--input_dir $INPUT_DIR \
	--output_dir $OUTPUT_DIR

uv run converter/ckpt_pth_to_dcp.py \
	--input_dir $OUTPUT_DIR \
	--output_dir $OUTPUT_2_DIR
