#!/usr/bin/env bash
# PCBA 测试集推理，8 卡 DDP 并行
set -euo pipefail
cd "$(dirname "$0")"

NPROC_PER_NODE=8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=600000 \
MIN_PIXELS=3136 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
python3 -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" PCBA/test_pcba.py
