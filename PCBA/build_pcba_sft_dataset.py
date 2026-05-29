#!/usr/bin/env python3
"""将 PCBA Standard-to-Real Challenge 训练 JSON 转为 ms-swift 多模态 SFT jsonl。

用法:
  python3 build_pcba_sft_dataset.py

环境变量:
  PCBA_ROOT      数据集根目录，默认见 utils.DEFAULT_PCBA_ROOT
  OUT_JSONL       训练集输出，默认 pcba_sft_train.jsonl（相对本目录）
  OUT_VAL_JSONL   验证集输出，默认 pcba_sft_val.jsonl（相对本目录）
  VAL_RATIO       官方数据验证集比例，默认 0.01
  VAL_SEED        划分随机种子，默认 42
  EXTRA_SFT_JSONLS  额外 SFT jsonl，逗号分隔；只加入训练集，不参与验证集划分
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
EXTRA_SFT_JSONLS = os.environ.get("EXTRA_SFT_JSONLS", "")


PCBA_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PCBA_DIR, path)


def _resolve_image_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PCBA_DIR, path))


def _extra_sft_paths() -> list[str]:
    return [p.strip() for p in EXTRA_SFT_JSONLS.split(",") if p.strip()]


def _load_extra_sft_rows() -> list[dict]:
    rows = []
    for rel_path in _extra_sft_paths():
        path = _resolve_path(rel_path)
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"额外 SFT 数据行不是 object: {path}:{line_no}")
                images = row.get("images") or []
                if images:
                    row["images"] = [_resolve_image_path(str(p)) for p in images]
                rows.append(row)
    return rows


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
    official_samples = list(iter_train_rows(PCBA_ROOT))
    extra_samples = _load_extra_sft_rows()

    rng = random.Random(VAL_SEED)
    rng.shuffle(official_samples)

    val_size = max(int(len(official_samples) * VAL_RATIO), 1)
    val_samples = official_samples[:val_size]
    train_samples = official_samples[val_size:] + extra_samples

    train_path = _resolve_path(OUT_JSONL)
    val_path = _resolve_path(OUT_VAL_JSONL)
    train_count = _write_jsonl(train_path, train_samples)
    val_count = _write_jsonl(val_path, val_samples)

    print(f"Wrote {train_count} train + {val_count} val samples "
          f"(official_val_ratio={VAL_RATIO}, seed={VAL_SEED}, extra_train={len(extra_samples)})")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")


if __name__ == "__main__":
    main()
