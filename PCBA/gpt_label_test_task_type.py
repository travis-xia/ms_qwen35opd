#!/usr/bin/env python3
"""
为 PCBA VQA 测试集判断是否属于 Standard / 标准知识题。

直接改下方「运行配置」，然后执行：
  python3 gpt_label_test_task_type.py

注意：测试集没有 Standard / RealWorld 直接来源字段，所以这里只做一件事：
判断每条测试样本是否是 standard 题。不要在这里细分 component_type、defect_type 等任务，
避免规则抢占导致 standard 题被误分到其他类。

策略：全部交给模型二分类 + jsonl checkpoint 断点续跑。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from gpt_label_task_type import (
    LIMIT,
    NO_RESUME,
    OUTPUT_DIR,
    PCBA_CHALLENGE_ROOT,
    QID,
    QUIET,
    append_jsonl,
    load_checkpoint,
    load_json_array,
    log,
    qid_key,
)

# =============================================================================
# 运行配置（只改这里）
# =============================================================================

TEST_JSON_REL = "Test/vqa_test_public.json"

# 输出文件名。默认写到 PCBA/task_type/vqa_test_public_with_standard.json
OUTPUT_JSON_NAME = "vqa_test_public_with_standard.json"
CHECKPOINT_NAME = "test_standard_checkpoint.jsonl"
FAILURE_NAME = "test_standard_failure.jsonl"

MODEL = os.environ.get("GPT_TASK_LABEL_MODEL", "gpt-5.4")
REASONING_EFFORT = os.environ.get("GPT_TASK_LABEL_REASONING_EFFORT", "low")
API_KEY = os.environ.get(
    "API_KEY",
    "sk-0ee0017ec97ddea77286375770b4bd5bbd378c7bd8ade9badf02d72c17e634b0",
)
BASE = os.environ.get("GPT_TASK_LABEL_BASE", "https://ai.deeptoken.site/v1")
MAX_COMPLETION_TOKENS = int(os.environ.get("GPT_TASK_LABEL_MAX_TOKENS", "512"))
MAX_RETRIES = int(os.environ.get("GPT_TASK_LABEL_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("GPT_TASK_LABEL_TIMEOUT", "120"))
WORKERS = int(os.environ.get("GPT_TEST_STANDARD_WORKERS", "32"))

# =============================================================================

_print_lock = threading.Lock()


def safe_log(msg: str) -> None:
    with _print_lock:
        log(msg)


def resolve_test_paths() -> tuple[Path, Path, Path, Path]:
    root = PCBA_CHALLENGE_ROOT.resolve()
    out_dir = OUTPUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = root / TEST_JSON_REL
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到测试集: {input_path}")

    checkpoint_path = out_dir / CHECKPOINT_NAME
    failure_path = out_dir / FAILURE_NAME
    output_path = out_dir / OUTPUT_JSON_NAME
    return input_path, checkpoint_path, failure_path, output_path


def call_gpt(system: str, user: str) -> str:
    url = f"{BASE.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }
    req = urllib.request.Request(
        url,
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
    return (msg.get("content") or "").strip()


def call_gpt_with_retry(system: str, user: str) -> str:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_gpt(system, user)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GPT 请求失败（重试 {MAX_RETRIES} 次）: {last_err}") from last_err


def build_standard_messages(row: dict[str, Any]) -> tuple[str, str]:
    options = row.get("options") or {}
    opt_lines = []
    for key in sorted(options, key=lambda x: str(x)):
        opt_lines.append(f"{key}. {options[key]}")
    opt_block = "\n".join(opt_lines) if opt_lines else "(none — quantitative / open numeric answer)"

    image_paths = row.get("image_paths") or []
    image_block = "\n".join(str(p) for p in image_paths) if image_paths else "(none)"

    system = (
        "You classify PCBA VQA test questions into exactly one binary source type: "
        "standard or non_standard. Output ONLY valid JSON on one line, no markdown:\n"
        '{"is_standard":true|false,"confidence":"high|medium|low","reason":"<short>"}\n\n'
        "Definitions:\n"
        "- standard: the question is about IPC/PCBA standard document knowledge, standard figures, "
        "standard diagrams, figure numbers, classes, acceptability/nonconformance criteria from a standard, "
        "or judging based on a shown standard/reference illustration rather than a real inspection photo.\n"
        "- non_standard: the question is about real PCB/PCBA photos, component identification, mount side, "
        "defect existence/type in real images, component counts, pin/lead counts, or geometric attributes in real photos.\n\n"
        "Important disambiguation:\n"
        "- Do NOT classify as non_standard merely because the question asks about defects, shapes, components, "
        "or yes/no. If the evidence is a standard diagram/standard figure/reference from IPC, it is standard.\n"
        "- If the question mentions Figure, Class 1/2/3, Acceptable, Nonconforming, target condition, "
        "process indicator, IPC, standard, or standard document, strongly consider standard.\n"
        "- If uncertain, use the image path, wording, options, and whether it resembles a standard-document QA item.\n"
        "Do NOT answer the VQA question itself; only classify whether it is standard."
    )

    user = (
        f"qid: {row.get('qid')}\n"
        f"question: {row.get('question')}\n"
        f"image_paths:\n{image_block}\n"
        f"options:\n{opt_block}\n"
    )
    return system, user


def parse_standard_json(text: str) -> tuple[bool | None, str, str]:
    text = (text or "").strip()
    if not text:
        return None, "low", "empty_response"

    def parse_obj(obj: Any) -> tuple[bool | None, str, str]:
        if not isinstance(obj, dict):
            return None, "low", "not_object"
        if "is_standard" in obj:
            value = obj.get("is_standard")
            if isinstance(value, bool):
                conf = str(obj.get("confidence") or "medium").strip().lower()
                reason = str(obj.get("reason") or "api_json")
                return value, conf if conf in {"high", "medium", "low"} else "medium", reason
            if isinstance(value, str):
                vl = value.strip().lower()
                if vl in {"true", "yes", "standard"}:
                    conf = str(obj.get("confidence") or "medium").strip().lower()
                    reason = str(obj.get("reason") or "api_json_string_bool")
                    return True, conf if conf in {"high", "medium", "low"} else "medium", reason
                if vl in {"false", "no", "non_standard", "non-standard"}:
                    conf = str(obj.get("confidence") or "medium").strip().lower()
                    reason = str(obj.get("reason") or "api_json_string_bool")
                    return False, conf if conf in {"high", "medium", "low"} else "medium", reason
        if "label" in obj:
            label = str(obj.get("label") or "").strip().lower()
            if label in {"standard", "standard_knowledge"}:
                return True, "medium", str(obj.get("reason") or "api_json_label")
            if label in {"non_standard", "non-standard", "realworld", "real_world"}:
                return False, "medium", str(obj.get("reason") or "api_json_label")
        return None, "low", "missing_is_standard"

    try:
        return parse_obj(json.loads(text))
    except json.JSONDecodeError:
        pass

    for m in re.finditer(r"\{[^{}]*(?:is_standard|label)[^{}]*\}", text, flags=re.I):
        try:
            parsed = parse_obj(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
        if parsed[0] is not None:
            return parsed

    lower = text.lower()
    if "non_standard" in lower or "non-standard" in lower:
        return False, "low", "api_text_non_standard"
    if "standard" in lower:
        return True, "low", "api_text_standard"
    return None, "low", "parse_failed"


def process_one_standard(row: dict[str, Any]) -> dict[str, Any]:
    qid = qid_key(row)
    base: dict[str, Any] = {
        "qid": qid,
        "source": "test",
        "question": row.get("question"),
        "has_options": bool(row.get("options")),
    }

    t0 = time.perf_counter()
    sys_msg, usr_msg = build_standard_messages(row)
    try:
        raw = call_gpt_with_retry(sys_msg, usr_msg)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            **base,
            "status": "failed",
            "reason": "api_error",
            "detail": str(e),
            "elapsed_sec": round(elapsed, 3),
        }

    is_standard, conf, reason = parse_standard_json(raw)
    elapsed = time.perf_counter() - t0
    if is_standard is None:
        return {
            **base,
            "status": "failed",
            "reason": reason,
            "raw_gpt": raw,
            "elapsed_sec": round(elapsed, 3),
        }

    return {
        **base,
        "status": "ok",
        "is_standard": is_standard,
        "standard_label": "standard" if is_standard else "non_standard",
        "confidence": conf,
        "method": "api_binary_standard",
        "reason": reason,
        "raw_gpt": raw,
        "elapsed_sec": round(elapsed, 3),
    }


def merge_standard_output(
    rows: list[dict[str, Any]],
    labeled: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        qid = qid_key(row)
        rec = labeled.get(qid)
        item = dict(row)
        if rec and rec.get("status") == "ok":
            item["is_standard"] = rec["is_standard"]
            item["standard_label"] = rec["standard_label"]
            item["standard_confidence"] = rec.get("confidence")
            item["standard_method"] = rec.get("method")
            item["standard_reason"] = rec.get("reason")
        out.append(item)
    return out


def summarize_standard(labeled: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {"standard": 0, "non_standard": 0, "other": 0}
    for rec in labeled.values():
        if rec.get("is_standard") is True:
            counts["standard"] += 1
        elif rec.get("is_standard") is False:
            counts["non_standard"] += 1
        else:
            counts["other"] += 1
    return counts


def run_test_dataset(
    *,
    input_path: Path,
    checkpoint_path: Path,
    failure_path: Path,
    output_path: Path,
) -> int:
    rows = load_json_array(input_path)
    if QID:
        rows = [r for r in rows if qid_key(r) == str(QID).strip()]
        if not rows:
            print(f"[test] 未找到 qid={QID!r}", file=sys.stderr)
            return 2
    elif LIMIT is not None:
        rows = rows[:LIMIT]

    done: dict[str, dict[str, Any]] = {}
    if not NO_RESUME:
        done = load_checkpoint(checkpoint_path)

    pending = [r for r in rows if qid_key(r) not in done]
    safe_log(
        f"\n=== test standard binary ===\n"
        f"输入: {input_path}\n"
        f"共 {len(rows)} 条 | 已完成 {len(rows) - len(pending)} | 待处理 {len(pending)} | "
        f"model={MODEL} workers={WORKERS}\n"
        "说明: 只判断 is_standard，不做 8 类 task 细分；所有样本都交给模型二分类。"
    )

    if NO_RESUME and checkpoint_path.is_file() and pending:
        checkpoint_path.unlink()

    run_failures: list[str] = []

    def handle(record: dict[str, Any]) -> None:
        qid = record["qid"]
        if record.get("status") == "ok":
            append_jsonl(checkpoint_path, record)
            done[qid] = record
            if not QUIET:
                safe_log(
                    f"[test:{qid}] OK label={record['standard_label']} "
                    f"conf={record.get('confidence')} ({record.get('elapsed_sec')}s)"
                )
            return

        append_jsonl(failure_path, record)
        run_failures.append(qid)
        if not QUIET:
            safe_log(f"[test:{qid}] FAIL reason={record.get('reason')} ({record.get('elapsed_sec')}s)")

    if pending:
        if WORKERS <= 1:
            for i, row in enumerate(pending, 1):
                if not QUIET:
                    safe_log(f"({i}/{len(pending)}) test qid={qid_key(row)} ...")
                handle(process_one_standard(row))
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futures = {ex.submit(process_one_standard, row): row for row in pending}
                for fut in as_completed(futures):
                    handle(fut.result())

    merged = merge_standard_output(rows, done)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    counts = summarize_standard(done)
    safe_log(f"[test] 已写出: {output_path}")
    safe_log(f"[test] checkpoint: {checkpoint_path} ({len(done)} 条)")
    safe_log(
        "[test] standard 分布: "
        + ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
    )

    missing = [qid_key(r) for r in rows if qid_key(r) not in done]
    if missing:
        safe_log(f"[test] 仍缺 {len(missing)} 条，重新运行可续跑")
        if run_failures:
            safe_log(
                f"[test] 本轮失败: {', '.join(run_failures[:20])}"
                + (" ..." if len(run_failures) > 20 else "")
            )
        return 1
    return 0


def main() -> int:
    if not PCBA_CHALLENGE_ROOT.is_dir():
        print(f"PCBA_CHALLENGE_ROOT 不存在: {PCBA_CHALLENGE_ROOT}", file=sys.stderr)
        return 2

    input_path, checkpoint_path, failure_path, output_path = resolve_test_paths()
    safe_log(f"PCBA_CHALLENGE_ROOT = {PCBA_CHALLENGE_ROOT.resolve()}")
    safe_log(f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    safe_log(f"TEST_JSON_REL = {TEST_JSON_REL}")
    safe_log("task = binary_standard_detection")

    return run_test_dataset(
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        failure_path=failure_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
