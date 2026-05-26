#!/usr/bin/env python3
"""将 PCBA Standard-to-Real Challenge 训练 JSON 转为 ms-swift 多模态 SFT jsonl。

用法:
  python3 build_pcba_sft_dataset.py

环境变量:
  PCBA_ROOT      数据集根目录，默认见 utils.DEFAULT_PCBA_ROOT
  OUT_JSONL      训练集输出，默认 pcba_sft_train.jsonl（相对本目录）
  OUT_VAL_JSONL  验证集输出，默认 pcba_sft_val.jsonl（相对本目录）
  VAL_RATIO      验证集比例，默认 0.01
  VAL_SEED       划分随机种子，默认 42
"""

from __future__ import annotations

import json
import os
import random

from utils import DEFAULT_PCBA_ROOT, iter_train_rows

PCBA_ROOT = os.environ.get("PCBA_ROOT", DEFAULT_PCBA_ROOT)
OUT_JSONL = os.environ.get("OUT_JSONL", "pcba_sft_train.jsonl")
OUT_VAL_JSONL = os.environ.get("OUT_VAL_JSONL", "pcba_sft_val.jsonl")
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.01"))
VAL_SEED = int(os.environ.get("VAL_SEED", "42"))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _write_jsonl(path: str, samples) -> int:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    samples = list(iter_train_rows(PCBA_ROOT))
    rng = random.Random(VAL_SEED)
    rng.shuffle(samples)

    val_size = max(int(len(samples) * VAL_RATIO), 1)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    train_path = _resolve_path(OUT_JSONL)
    val_path = _resolve_path(OUT_VAL_JSONL)
    train_count = _write_jsonl(train_path, train_samples)
    val_count = _write_jsonl(val_path, val_samples)

    print(f"Wrote {train_count} train + {val_count} val samples "
          f"(val_ratio={VAL_RATIO}, seed={VAL_SEED})")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")


if __name__ == "__main__":
    main()
