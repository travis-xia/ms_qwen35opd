#!/usr/bin/env python3
"""抽样评测 GPT-5.5 在 standard 训练集 MCQ 上的表现。"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASET_PATH = SCRIPT_DIR / "task_type" / "standard_mm_vqa_train_public_with_task.json"
IMAGE_ROOT = REPO_ROOT / "PCBA_Standard-to-Real_Challenge" / "Train" / "Standard"
OUTPUT_PATH = SCRIPT_DIR / "gpt55_standard_train_sample32_eval.json"

BASE = os.environ.get("OPENAI_BASE_URL", "https://ai.deeptoken.site/v1")
MODEL = os.environ.get("GPT_STANDARD_EVAL_MODEL", "gpt-5.5")
REASONING_EFFORT = os.environ.get("GPT_STANDARD_EVAL_REASONING_EFFORT", "medium")
API_KEY = os.environ.get(
    "API_KEY",
    "sk-0ee0017ec97ddea77286375770b4bd5bbd378c7bd8ade9badf02d72c17e634b0",
)
SAMPLE_SIZE = int(os.environ.get("GPT_STANDARD_EVAL_SAMPLE_SIZE", "128"))
SEED = int(os.environ.get("GPT_STANDARD_EVAL_SEED", "20260530"))
WORKERS = int(os.environ.get("GPT_STANDARD_EVAL_WORKERS", "32"))
MAX_COMPLETION_TOKENS = int(os.environ.get("GPT_STANDARD_EVAL_MAX_TOKENS", "1024"))
REQUEST_TIMEOUT = int(os.environ.get("GPT_STANDARD_EVAL_TIMEOUT", "180"))
MAX_RETRIES = int(os.environ.get("GPT_STANDARD_EVAL_MAX_RETRIES", "3"))

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def image_to_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def format_question(row: dict[str, Any]) -> str:
    options = row["options"]
    return (
        f"Question: {row['question']}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n\n"
        "Answer with only one letter: A, B, C, or D."
    )


def build_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": format_question(row)}]
    for rel in row.get("image_paths") or []:
        path = IMAGE_ROOT / rel
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return [{"role": "user", "content": content}]


def call_gpt(messages: list[dict[str, Any]]) -> str:
    if not API_KEY:
        raise RuntimeError("未设置 API_KEY")
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in PCBA visual inspection and IPC-A-610 acceptability standards. "
                    "Solve the multiple-choice question using the provided image(s). Return only the option letter."
                ),
            },
            *messages,
        ],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }
    req = urllib.request.Request(
        f"{BASE.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "curl/8.7.1",
            "Accept": "application/json",
        },
    )
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return str(msg.get("content") or "").strip()


def call_with_retry(messages: list[dict[str, Any]]) -> str:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_gpt(messages)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GPT 请求失败（重试 {MAX_RETRIES} 次）: {last_err}") from last_err


def extract_choice(text: str) -> str:
    text = (text or "").strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return text[:1] if text[:1] in {"A", "B", "C", "D"} else ""


def eval_one(idx: int, total: int, row: dict[str, Any]) -> dict[str, Any]:
    qid = row.get("qid")
    log(f"[{idx}/{total}] qid={qid} images={len(row.get('image_paths') or [])} ...")
    started = time.perf_counter()
    try:
        raw = call_with_retry(build_messages(row))
        pred = extract_choice(raw)
        ok = pred == str(row.get("answer", "")).strip().upper()
        error = None
    except Exception as e:
        raw = ""
        pred = ""
        ok = False
        error = str(e)
    elapsed = round(time.perf_counter() - started, 3)
    log(f"    qid={qid} pred={pred or '-'} gold={row.get('answer')} correct={ok} elapsed={elapsed}s")
    return {
        "qid": qid,
        "question": row.get("question"),
        "options": row.get("options"),
        "answer": row.get("answer"),
        "prediction": pred,
        "raw_prediction": raw,
        "correct": ok,
        "image_paths": row.get("image_paths") or [],
        "error": error,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    standard_rows = [r for r in rows if r.get("task") == "standard_knowledge"]
    rng = random.Random(SEED)
    sample = rng.sample(standard_rows, min(SAMPLE_SIZE, len(standard_rows)))

    results: list[dict[str, Any] | None] = [None] * len(sample)
    workers = max(1, WORKERS)
    if workers == 1:
        for idx, row in enumerate(sample, start=1):
            results[idx - 1] = eval_one(idx, len(sample), row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(eval_one, idx, len(sample), row): idx - 1 for idx, row in enumerate(sample, start=1)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

    final_results = [r for r in results if r is not None]
    correct = sum(1 for r in final_results if r["correct"])
    summary = {
        "dataset_path": str(DATASET_PATH.relative_to(SCRIPT_DIR)),
        "image_root": str(IMAGE_ROOT.relative_to(REPO_ROOT)),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "sample_size": len(final_results),
        "seed": SEED,
        "workers": workers,
        "correct": correct,
        "accuracy": correct / len(final_results) if final_results else 0,
        "results": final_results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\naccuracy={correct}/{len(final_results)} = {summary['accuracy']:.4f}")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
