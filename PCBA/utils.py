"""PCBA Standard-to-Real Challenge 数据集与 prompt 工具。"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_PCBA_ROOT = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/hf_download/PCBA_Standard-to-Real_Challenge"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert in PCBA visual inspection and manufacturing standards. "
    "Answer the multiple-choice question with the option letter only."
)

SYSTEM_PROMPT_QUANTITATIVE = (
    "You are an expert in PCBA visual inspection and manufacturing standards. "
    "Answer the quantitative question with a number only."
)

TRAIN_JSONS = (
    ("Train/Standard/standard_mm_vqa_train_public.json", "Train/Standard"),
    ("Train/RealWorld/realworld_mm_vqa_train_public.json", "Train/RealWorld"),
)

TEST_JSON = ("Test/vqa_test_public.json", "Test")


def is_quantitative(row: Dict[str, Any]) -> bool:
    return not (row.get("options") or {})


def format_mcq_prompt(question: str, options: Dict[str, str]) -> str:
    lines = [question, "", "Options:"]
    for key in sorted(options):
        lines.append(f"{key}. {options[key]}")
    lines.append("")
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def format_quantitative_prompt(question: str) -> str:
    return f"{question}\n\nAnswer with the number only."


def build_sample(row: Dict[str, Any], image_root: str, *, with_answer: bool = True) -> Dict[str, Any]:
    image_paths = row.get("image_paths") or []
    images: List[str] = []
    for rel in image_paths:
        abs_path = os.path.join(image_root, rel)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"图片不存在: {abs_path}")
        images.append(os.path.abspath(abs_path))

    if is_quantitative(row):
        system_prompt = SYSTEM_PROMPT_QUANTITATIVE
        prompt_text = format_quantitative_prompt(row["question"])
    else:
        system_prompt = SYSTEM_PROMPT_MCQ
        prompt_text = format_mcq_prompt(row["question"], row["options"])

    user_content = "".join(["<image>"] * len(images)) + prompt_text
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": row["answer"]})

    sample: Dict[str, Any] = {
        "id": f"{row.get('qid', 'unknown')}",
        "messages": messages,
    }
    if images:
        sample["images"] = images
    return sample


def load_json_rows(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_train_rows(pcba_root: str) -> Iterable[Dict[str, Any]]:
    for rel_json, rel_image_root in TRAIN_JSONS:
        json_path = os.path.join(pcba_root, rel_json)
        image_root = os.path.join(pcba_root, rel_image_root)
        for row in load_json_rows(json_path):
            yield build_sample(row, image_root)





def iter_test_rows(pcba_root: str) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield (raw_test_row, infer_sample) pairs from the public test split."""
    rel_json, rel_image_root = TEST_JSON
    json_path = os.path.join(pcba_root, rel_json)
    image_root = os.path.join(pcba_root, rel_image_root)
    for row in load_json_rows(json_path):
        yield row, build_sample(row, image_root, with_answer=False)


def _strip_thinking(text: str) -> str:
    text = (text or '').strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    elif text.startswith('<think>'):
        return ''
    return text.strip()


def normalize_answer(raw: str, row: Dict[str, Any]) -> str:
    """Post-process model output into a submission-friendly answer."""
    text = _strip_thinking(raw)
    if not text:
        return text
    if is_quantitative(row):
        match = re.search(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
        return match.group(0) if match else text
    options = row.get('options') or {}
    valid = {str(k).upper() for k in options}
    upper = text.strip().upper()
    if upper in valid:
        return upper
    for line in reversed(text.splitlines()):
        candidate = line.strip().upper()
        if candidate in valid:
            return candidate
    match = re.match(r'^([A-Za-z])\b', text.strip())
    if match and match.group(1).upper() in valid:
        return match.group(1).upper()
    for ch in upper:
        if ch in valid:
            return ch
    return text.strip()
