#!/usr/bin/env python3
"""Score SWIFT PCBA validation inference results.

The input is the JSONL produced by ``swift infer`` on PCBA/ablation/val_all.jsonl.
It prints the aggregate score for val_all and the two domain subsets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OPTION_LINE_RE = re.compile(r"^\s*([A-Z])\.\s+", re.MULTILINE)


def _strip_thinking(text: str) -> str:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    elif text.startswith("<think>"):
        return ""
    return text.strip()


def _as_decimal(text: str) -> Optional[Decimal]:
    try:
        return Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        return None


def _is_numeric_gold(gold: str) -> bool:
    return _as_decimal(gold) is not None


def _message_text(messages: Iterable[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunks)


def _valid_choices(row: Dict[str, Any], gold: str) -> List[str]:
    text = _message_text(row.get("messages") or [])
    choices = sorted(set(OPTION_LINE_RE.findall(text)))
    if choices:
        return choices
    if re.fullmatch(r"[A-Za-z]", gold):
        return ["A", "B", "C", "D"]
    return []


def _normalize_choice(response: str, valid_choices: List[str]) -> str:
    valid = set(valid_choices)
    text = _strip_thinking(response).strip()
    upper = text.upper()
    if upper in valid:
        return upper

    patterns = [
        r"(?:ANSWER|OPTION|CHOICE|FINAL)\s*(?:IS|:)?\s*([A-Z])\b",
        r"^\s*[\(\[]?([A-Z])[\)\].:\-\s]",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, upper):
            candidate = match.group(1)
            if candidate in valid:
                return candidate

    for char in upper:
        if char in valid:
            return char
    return ""


def _normalize_numeric(response: str) -> str:
    text = _strip_thinking(response).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else text.strip()


def normalize_prediction(row: Dict[str, Any]) -> Tuple[str, str]:
    gold = str(row.get("labels", "")).strip()
    response = str(row.get("response", "")).strip()
    if _is_numeric_gold(gold):
        return _normalize_numeric(response), gold

    gold = gold.upper()
    choices = _valid_choices(row, gold)
    if choices:
        return _normalize_choice(response, choices), gold
    return _strip_thinking(response).strip(), gold


def is_correct(prediction: str, gold: str) -> bool:
    pred_num = _as_decimal(prediction)
    gold_num = _as_decimal(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    return prediction.strip().upper() == gold.strip().upper()


def _empty_stats() -> Dict[str, int]:
    return {"correct": 0, "total": 0, "invalid": 0}


def _update(stats: Dict[str, int], *, correct: bool, invalid: bool) -> None:
    stats["total"] += 1
    stats["correct"] += int(correct)
    stats["invalid"] += int(invalid)


def _format_stats(name: str, stats: Dict[str, int]) -> str:
    total = stats["total"]
    correct = stats["correct"]
    acc = correct / total if total else 0.0
    invalid = stats["invalid"]
    return f"{name}: {correct}/{total} = {acc:.4f} (invalid={invalid})"


def score_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = _empty_stats()
    by_domain = defaultdict(_empty_stats)
    by_task = defaultdict(_empty_stats)

    examples = []
    for row in rows:
        prediction, gold = normalize_prediction(row)
        correct = is_correct(prediction, gold)
        invalid = prediction == ""
        domain = str(row.get("domain") or "unknown")
        task = str(row.get("task") or "unknown")
        _update(overall, correct=correct, invalid=invalid)
        _update(by_domain[domain], correct=correct, invalid=invalid)
        _update(by_task[task], correct=correct, invalid=invalid)
        if len(examples) < 20 and not correct:
            examples.append({
                "id": row.get("id"),
                "domain": domain,
                "task": task,
                "gold": gold,
                "prediction": prediction,
                "response": row.get("response", ""),
            })

    return {
        "val_all": dict(overall),
        "by_domain": {key: dict(value) for key, value in sorted(by_domain.items())},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "wrong_examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score PCBA swift infer result JSONL.")
    parser.add_argument("result_jsonl", help="Path to swift infer result JSONL.")
    parser.add_argument("--summary-json", help="Optional path to write score summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = Path(args.result_jsonl)
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"[error] empty result file: {result_path}")

    summary = score_rows(rows)
    summary["result_jsonl"] = str(result_path)

    print(_format_stats("val_all", summary["val_all"]))
    for domain in ("standard", "realworld"):
        stats = summary["by_domain"].get(domain, _empty_stats())
        print(_format_stats(f"val_{domain}", stats))

    print("by_task:")
    for task, stats in summary["by_task"].items():
        print("  " + _format_stats(task, stats))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"summary_json: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
