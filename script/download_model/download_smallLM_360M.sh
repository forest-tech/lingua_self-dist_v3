#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:29:00
#PJM -j

uvx hf download HuggingFaceTB/SmolLM2-360M --local-dir ./model/smallLM_360M