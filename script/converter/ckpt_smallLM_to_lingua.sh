#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

INPUT_DIR='model/smallLM_360M'
OUTPUT_DIR='model/smallLM_360M_lingua'
OUTPUT_2_DIR='model/smallLM_360M_dcp'

uv run converter/ckpt_hf_to_lingua.py \
	--input_dir $INPUT_DIR \
	--output_dir $OUTPUT_DIR

uv run converter/ckpt_pth_to_dcp.py \
	--input_dir $OUTPUT_DIR \
	--output_dir $OUTPUT_2_DIR
