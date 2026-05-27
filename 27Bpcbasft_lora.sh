#!/usr/bin/env bash
# Qwen3.5-27B PCBA 电路板图片问答 LoRA SFT（训练后自动 merge）
# 跳过数据组装: BUILD_DATASET=0 bash 27Bpcbasft_lora.sh
# 不合并、只保留 LoRA checkpoint: MERGE_LORA_AFTER=0 bash 27Bpcbasft_lora.sh
# eval 为生成式答题，按 seq_acc 选 best checkpoint（见 output/.../predict.jsonl）

set -euo pipefail
cd "$(dirname "$0")"

PCBA_ROOT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
DATASET="PCBA/pcba_sft_train.jsonl"
VAL_DATASET="PCBA/pcba_sft_val.jsonl"
OUTPUT_DIR="output/Qwen3.5-27B-pcba-lora"

if [[ "${BUILD_DATASET:-1}" == "1" ]]; then
  (cd PCBA && PCBA_ROOT="${PCBA_ROOT}" \
    OUT_JSONL="pcba_sft_train.jsonl" \
    OUT_VAL_JSONL="pcba_sft_val.jsonl" \
    VAL_RATIO="0.05" \
    python3 build_pcba_sft_dataset.py)
fi
[[ -f "${DATASET}" ]] || { echo "[error] 缺少 ${DATASET}"; exit 1; }
[[ -f "${VAL_DATASET}" ]] || { echo "[error] 缺少 ${VAL_DATASET}"; exit 1; }

# Standard 最多 11 张图，RealWorld 单图；单图像素与 PDF 任务保持一致
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=600000 \
MIN_PIXELS=3136 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
  --model /inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-27B \
  --tuner_type lora \
  --lora_rank 512 \
  --lora_alpha 512 \
  --target_modules all-linear \
  --freeze_vit false \
  --freeze_aligner false \
  --dataset "${DATASET}" \
  --val_dataset "${VAL_DATASET}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --enable_thinking false \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 3 \
  --predict_with_generate true \
  --max_new_tokens 16 \
  --temperature 0 \
  --eval_metric acc \
  --acc_strategy seq \
  --metric_for_best_model seq_acc \
  --logging_steps 5 \
  --max_length 32000 \
  --warmup_ratio 0.03 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --model_author swift \
  --attn_impl sdpa \
  --model_name swift-robot

if [[ "${MERGE_LORA_AFTER:-1}" == "1" ]]; then
  RUN_DIR="$(find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'v*' 2>/dev/null | LC_ALL=C sort | tail -1)"
  LAST_CKPT=""
  if [[ -n "${RUN_DIR}" ]]; then
    LAST_CKPT="$(find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' 2>/dev/null | LC_ALL=C sort -V | tail -1)"
  fi
  if [[ -z "${LAST_CKPT}" ]]; then
    echo "[warn] 未找到 checkpoint-*，跳过合并。" >&2
  else
    OUT_FULL="${LAST_CKPT}.__hf__"
    rm -rf "${OUT_FULL}"
    CUDA_VISIBLE_DEVICES=0 swift export \
      --adapters "${LAST_CKPT}" \
      --merge_lora true \
      --output_dir "${OUT_FULL}" \
      --exist_ok true
    rm -rf "${LAST_CKPT}"
    mv "${OUT_FULL}" "${LAST_CKPT}"
  fi
fi



# 手动merge中间的ckpt:
# CUDA_VISIBLE_DEVICES=0 swift export \
#   --adapters output/Qwen3.5-27B-pcba-lora/v0-20260527-165421/checkpoint-400 \
#   --merge_lora true \
#   --output_dir output/Qwen3.5-27B-pcba-lora/v0-20260527-165421/checkpoint-400-merged \
#   --exist_ok true