#!/usr/bin/env python3
"""Build fixed PCBA train/validation splits for post-competition ablations.

The script reuses the task-labeled PCBA SFT conversion code and writes six
SWIFT-compatible jsonl files:

  - train_all.jsonl / val_all.jsonl
  - train_standard.jsonl / val_standard.jsonl
  - train_realworld.jsonl / val_realworld.jsonl

Each sample receives a ``domain`` field so later teacher routing can choose the
matching domain adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from build_pcba_task_sft_dataset import (
    ALL_SPLITS,
    ALL_TASK_TYPES,
    build_samples,
)
from utils import DEFAULT_PCBA_ROOT


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LOCAL_PCBA_ROOT = os.path.join(REPO_ROOT, "PCBA_Standard-to-Real_Challenge")
DEFAULT_TASK_TYPE_DIR = os.path.join(SCRIPT_DIR, "task_type")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "ablation")


def _default_pcba_root() -> str:
    if os.environ.get("PCBA_ROOT"):
        return os.environ["PCBA_ROOT"]
    if os.path.isdir(LOCAL_PCBA_ROOT):
        return LOCAL_PCBA_ROOT
    return DEFAULT_PCBA_ROOT


def _resolve_path(path: str, *, base_dir: str = SCRIPT_DIR) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def _normalize_task_types(task_types: Optional[List[str]]) -> Optional[set[str]]:
    if not task_types:
        return None
    unknown = [task_type for task_type in task_types if task_type not in ALL_TASK_TYPES]
    if unknown:
        choices = ", ".join(ALL_TASK_TYPES)
        raise ValueError(f"Unknown task types: {unknown}. Choices: {choices}")
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


def _write_json(path: str, payload: Dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.write("\n")


def _load_domain_samples(
    *,
    pcba_root: str,
    task_type_dir: str,
    domain: str,
    task_types: Optional[set[str]],
) -> List[Dict]:
    samples = build_samples(
        pcba_root=pcba_root,
        task_type_dir=task_type_dir,
        splits=(domain,),
        task_types=task_types,
    )
    for sample in samples:
        sample["domain"] = domain
    return sorted(samples, key=lambda sample: str(sample["id"]))


def _split_samples(samples: Sequence[Dict], *, val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not samples:
        return [], []
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    val_size = min(val_size, len(shuffled) - 1) if len(shuffled) > 1 else 1
    return shuffled[val_size:], shuffled[:val_size]


def _stable_shuffle(samples: Sequence[Dict], *, seed: int) -> List[Dict]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _count(samples: Sequence[Dict]) -> Dict:
    return {
        "total": len(samples),
        "by_domain": dict(Counter(sample.get("domain", "unknown") for sample in samples)),
        "by_task": dict(Counter(sample.get("task", "unknown") for sample in samples)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed PCBA ablation splits.")
    parser.add_argument(
        "--pcba-root",
        default=_default_pcba_root(),
        help="PCBA dataset root for image files.",
    )
    parser.add_argument(
        "--task-type-dir",
        default=os.environ.get("TASK_TYPE_DIR", DEFAULT_TASK_TYPE_DIR),
        help="Directory containing task-labeled PCBA train JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("OUT_DIR", DEFAULT_OUT_DIR),
        help="Output directory for split jsonl files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(os.environ.get("VAL_RATIO", "0.10")),
        help="Validation ratio applied independently to each domain.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=int(os.environ.get("VAL_SEED", "42")),
        help="Random seed for fixed train/validation splits.",
    )
    parser.add_argument(
        "--task-types",
        default=os.environ.get("TASK_TYPES"),
        help="Optional comma-separated task types. Defaults to all task types.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("[error] --val-ratio must be between 0 and 1.")

    task_type_dir = _resolve_path(args.task_type_dir)
    out_dir = _resolve_path(args.out_dir)
    task_types = _normalize_task_types(_parse_csv(args.task_types))

    domain_train: Dict[str, List[Dict]] = {}
    domain_val: Dict[str, List[Dict]] = {}
    for offset, domain in enumerate(ALL_SPLITS):
        samples = _load_domain_samples(
            pcba_root=args.pcba_root,
            task_type_dir=task_type_dir,
            domain=domain,
            task_types=task_types,
        )
        if not samples:
            raise SystemExit(f"[error] No samples found for domain: {domain}")
        train_samples, val_samples = _split_samples(
            samples,
            val_ratio=args.val_ratio,
            seed=args.val_seed + offset,
        )
        domain_train[domain] = train_samples
        domain_val[domain] = val_samples

    train_all = _stable_shuffle(
        [sample for domain in ALL_SPLITS for sample in domain_train[domain]],
        seed=args.val_seed + 1000,
    )
    val_all = _stable_shuffle(
        [sample for domain in ALL_SPLITS for sample in domain_val[domain]],
        seed=args.val_seed + 2000,
    )

    outputs = {
        "train_all": train_all,
        "val_all": val_all,
        "train_standard": domain_train["standard"],
        "val_standard": domain_val["standard"],
        "train_realworld": domain_train["realworld"],
        "val_realworld": domain_val["realworld"],
    }

    counts = {}
    files = {}
    for name, samples in outputs.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        _write_jsonl(path, samples)
        files[name] = path
        counts[name] = _count(samples)

    manifest_path = os.path.join(out_dir, "manifest.json")
    _write_json(
        manifest_path,
        {
            "pcba_root": args.pcba_root,
            "task_type_dir": task_type_dir,
            "val_ratio": args.val_ratio,
            "val_seed": args.val_seed,
            "task_types": sorted(task_types) if task_types else "all",
            "files": files,
            "counts": counts,
        },
    )

    print(f"Wrote fixed PCBA ablation splits to {out_dir}")
    print(f"  val_ratio: {args.val_ratio}")
    print(f"  val_seed:  {args.val_seed}")
    for name in outputs:
        print(f"  {name}: {counts[name]['total']}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
