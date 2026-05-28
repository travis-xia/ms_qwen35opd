#!/usr/bin/env python3
"""将 PCBA/task_type 中带 task 标签的训练 JSON 转为 ms-swift 多模态 SFT jsonl。

用法:
  python3 build_pcba_task_sft_dataset.py
  python3 build_pcba_task_sft_dataset.py --splits standard
  python3 build_pcba_task_sft_dataset.py --splits realworld --task-types defect_type,defect_existence

环境变量:
  PCBA_ROOT      数据集根目录（图片路径），默认见 utils.DEFAULT_PCBA_ROOT
  TASK_TYPE_DIR  带 task 标签的 JSON 目录，默认 task_type（相对本脚本目录）
  TRAIN_SPLITS   逗号分隔: standard,realworld，默认两者都用
  TASK_TYPES     逗号分隔 task 类型，默认空表示全部
  OUT_JSONL      训练集输出，默认 pcba_sft_train.jsonl（相对本目录）
  OUT_VAL_JSONL  验证集输出，默认 pcba_sft_val.jsonl（相对本目录）
  VAL_RATIO      验证集比例，默认 0.02
  VAL_SEED       划分随机种子，默认 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from utils import DEFAULT_PCBA_ROOT, build_sample, load_json_rows

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASK_TYPE_DIR = os.path.join(SCRIPT_DIR, "task_type")

SPLIT_CONFIGS: Dict[str, Dict[str, str]] = {
    "standard": {
        "json": "standard_mm_vqa_train_public_with_task.json",
        "image_root": "Train/Standard",
    },
    "realworld": {
        "json": "realworld_mm_vqa_train_public_with_task.json",
        "image_root": "Train/RealWorld",
    },
}

ALL_SPLITS = tuple(SPLIT_CONFIGS)
ALL_TASK_TYPES = (
    "standard_knowledge",
    "component_type",
    "mount_side",
    "defect_existence",
    "defect_type",
    "count_component",
    "count_pin_lead",
    "attribute_reasoning",
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def _normalize_splits(splits: Optional[List[str]]) -> Tuple[str, ...]:
    selected = splits or list(ALL_SPLITS)
    unknown = [s for s in selected if s not in SPLIT_CONFIGS]
    if unknown:
        raise ValueError(f"未知 split: {unknown}，可选: {', '.join(ALL_SPLITS)}")
    return tuple(selected)


def _normalize_task_types(task_types: Optional[List[str]]) -> Optional[Set[str]]:
    if not task_types:
        return None
    unknown = [t for t in task_types if t not in ALL_TASK_TYPES]
    if unknown:
        raise ValueError(f"未知 task_type: {unknown}，可选: {', '.join(ALL_TASK_TYPES)}")
    return set(task_types)


def _write_jsonl(path: str, samples: Iterable[Dict]) -> int:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_task_rows(
    *,
    pcba_root: str,
    task_type_dir: str,
    splits: Tuple[str, ...],
    task_types: Optional[Set[str]],
) -> Iterable[Dict]:
    for split in splits:
        cfg = SPLIT_CONFIGS[split]
        json_path = os.path.join(task_type_dir, cfg["json"])
        image_root = os.path.join(pcba_root, cfg["image_root"])
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"缺少 JSON: {json_path}")

        for row in load_json_rows(json_path):
            task = row.get("task")
            if task_types is not None and task not in task_types:
                continue
            sample = build_sample(row, image_root)
            sample["id"] = f"{split}-{row.get('qid', 'unknown')}"
            if task:
                sample["task"] = task
            yield sample


def build_samples(
    *,
    pcba_root: str,
    task_type_dir: str,
    splits: Tuple[str, ...],
    task_types: Optional[Set[str]],
) -> List[Dict]:
    return list(
        iter_task_rows(
            pcba_root=pcba_root,
            task_type_dir=task_type_dir,
            splits=splits,
            task_types=task_types,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PCBA task-filtered SFT jsonl.")
    parser.add_argument(
        "--splits",
        default=os.environ.get("TRAIN_SPLITS"),
        help="Comma-separated splits: standard,realworld (default: both)",
    )
    parser.add_argument(
        "--task-types",
        default=os.environ.get("TASK_TYPES"),
        help="Comma-separated task types (default: all)",
    )
    parser.add_argument(
        "--pcba-root",
        default=os.environ.get("PCBA_ROOT", DEFAULT_PCBA_ROOT),
        help="PCBA dataset root for image files",
    )
    parser.add_argument(
        "--task-type-dir",
        default=os.environ.get("TASK_TYPE_DIR", DEFAULT_TASK_TYPE_DIR),
        help="Directory containing *_with_task.json files",
    )
    parser.add_argument(
        "--out-jsonl",
        default=os.environ.get("OUT_JSONL", "pcba_sft_train.jsonl"),
        help="Output train jsonl path",
    )
    parser.add_argument(
        "--out-val-jsonl",
        default=os.environ.get("OUT_VAL_JSONL", "pcba_sft_val.jsonl"),
        help="Output val jsonl path",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(os.environ.get("VAL_RATIO", "0.02")),
        help="Validation split ratio",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=int(os.environ.get("VAL_SEED", "42")),
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = _normalize_splits(_parse_csv(args.splits))
    task_types = _normalize_task_types(_parse_csv(args.task_types))
    task_type_dir = _resolve_path(args.task_type_dir)

    samples = build_samples(
        pcba_root=args.pcba_root,
        task_type_dir=task_type_dir,
        splits=splits,
        task_types=task_types,
    )
    if not samples:
        raise SystemExit("[error] 过滤后没有样本，请检查 splits / task_types 设置")

    rng = random.Random(args.val_seed)
    rng.shuffle(samples)

    val_size = max(int(len(samples) * args.val_ratio), 1)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    train_path = _resolve_path(args.out_jsonl)
    val_path = _resolve_path(args.out_val_jsonl)
    train_count = _write_jsonl(train_path, train_samples)
    val_count = _write_jsonl(val_path, val_samples)

    task_counter = Counter(s.get("task", "unknown") for s in samples)
    split_counter = Counter(s["id"].split("-", 1)[0] for s in samples)

    print(f"Wrote {train_count} train + {val_count} val samples "
          f"(val_ratio={args.val_ratio}, seed={args.val_seed})")
    print(f"  splits:     {', '.join(splits)}")
    print(f"  task_types: {', '.join(sorted(task_types)) if task_types else 'all'}")
    print(f"  by split:   {dict(split_counter)}")
    print(f"  by task:    {dict(task_counter)}")
    print(f"  train:      {train_path}")
    print(f"  val:        {val_path}")


if __name__ == "__main__":
    main()
