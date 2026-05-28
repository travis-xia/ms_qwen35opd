#!/usr/bin/env python3
"""
为 PCBA VQA 训练样本标注评测任务类型（task type）。

直接改下方「运行配置」，然后执行：
  python3 gpt_label_task_type.py

策略：规则优先 + API 兜底 + jsonl checkpoint 断点续跑。
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

# =============================================================================
# 运行配置（只改这里）
# =============================================================================

PCBA_CHALLENGE_ROOT = Path(
    "/Users/xiasheng/Documents/创智学院/challenge/ms_qwen35opd/PCBA_Standard-to-Real_Challenge"
)

STANDARD_JSON_REL = "Train/Standard/standard_mm_vqa_train_public.json"
REALWORLD_JSON_REL = "Train/RealWorld/realworld_mm_vqa_train_public.json"

# 处理哪些集："standard" | "realworld" | "both"
RUN_DATASETS = "both"

# 所有输出（checkpoint、失败记录、带 task 的 JSON）均写在 PCBA/task_type/
OUTPUT_DIR = Path(__file__).resolve().parent / "task_type"

WORKERS = 30
RULES_ONLY = False  # True：不调 API，仅规则
API_ALL = False  # True：全部走 API（调试）
NO_RESUME = False  # True：清空 checkpoint 重跑
QUIET = False

# 调试：None 表示全量
LIMIT: int | None = None
QID: str | None = None  # 例如 "42"，只跑单题

# =============================================================================

BASE = "https://ai.deeptoken.site/v1"
MODEL = os.environ.get("GPT_TASK_LABEL_MODEL", "gpt-5.4")
REASONING_EFFORT = os.environ.get("GPT_TASK_LABEL_REASONING_EFFORT", "low")
API_KEY = os.environ.get(
    "API_KEY",
    "sk-0ee0017ec97ddea77286375770b4bd5bbd378c7bd8ade9badf02d72c17e634b0",
)
MAX_COMPLETION_TOKENS = int(os.environ.get("GPT_TASK_LABEL_MAX_TOKENS", "512"))
MAX_RETRIES = int(os.environ.get("GPT_TASK_LABEL_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("GPT_TASK_LABEL_TIMEOUT", "120"))

TASK_TYPES = (
    "standard_knowledge",
    "component_type",
    "mount_side",
    "defect_existence",
    "defect_type",
    "count_component",
    "count_pin_lead",
    "attribute_reasoning",
)

YES_NO = frozenset({"yes", "no"})

_print_lock = threading.Lock()
_checkpoint_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 规则分类
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _option_texts(options: dict[str, Any]) -> list[str]:
    return [_norm(str(v)) for v in (options or {}).values()]


def _is_yes_no_options(options: dict[str, Any]) -> bool:
    if len(options) != 2:
        return False
    texts = set(_option_texts(options))
    return texts == YES_NO or texts <= YES_NO


def _has_pin_count_question(q: str) -> bool:
    ql = _norm(q)
    if "pin" not in ql and "lead" not in ql:
        return False
    return any(
        k in ql
        for k in (
            "pin count",
            "pins on",
            "number of pins",
            "pin quantity",
            "total pin",
            "count the pins",
            "pins associated",
            "pins does",
            "leads on the primary",
        )
    )


def _has_component_count_question(q: str) -> bool:
    ql = _norm(q)
    if "component" not in ql and "electronic" not in ql:
        return False
    return any(
        k in ql
        for k in (
            "how many electronic",
            "total number of electronic",
            "count every electronic component",
            "count of all electronic",
            "count the electronic components",
            "components on the board",
            "components on the pcb",
            "components visible on the pcb",
            "components are on the pcb",
            "components are visible",
            "identify and count every electronic",
            "perform a count of all electronic",
            "calculate the total number of electronic",
            "please count the electronic components",
            "please provide the exact count of electronic",
            "quantity of electronic components",
            "precise quantity of electronic components",
            "number of electronic components present",
            "electronic components detected on the pcb",
            "electronic components present on the board",
        )
    )


def _is_shape_question(q: str) -> bool:
    ql = _norm(q)
    if "shape" not in ql and "geometric" not in ql and "outline" not in ql:
        return False
    if "standard shape" in ql or "closest standard geometric" in ql:
        return True
    if "shape classification" in ql or "shape does" in ql or "shape best" in ql:
        return True
    if "most closely resemble" in ql and "shape" in ql:
        return True
    if "geometric form" in ql or "geometric shape" in ql:
        return True
    return False


def _is_component_type_question(q: str) -> bool:
    ql = _norm(q)
    if _is_shape_question(q):
        return False
    markers = (
        "red box",
        "red-bordered",
        "highlighted by the red",
        "framed by the red",
        "enclosed within the red",
        "inside the red box",
        "within the red box",
    )
    if not any(m in ql for m in markers):
        return False
    return any(
        k in ql
        for k in (
            "what is the name",
            "what specific component",
            "what type of component",
            "what is the component",
            "name of the main electronic",
            "name of the component",
            "designation of the main part",
            "prominent electronic component",
            "primary component highlighted",
            "primary electronic device",
            "main electronic component",
            "called?",
            "what is it specifically",
        )
    )


def _is_mount_side_question(q: str) -> bool:
    ql = _norm(q)
    return any(
        k in ql
        for k in (
            "component side",
            "primary component side",
            "top-side or bottom-side",
            "top side or bottom",
            "which side is the component side",
            "relative to the board",
            "pcb population",
            "mounted on either side",
        )
    )


def _is_defect_existence_question(q: str, options: dict[str, Any]) -> bool:
    if not _is_yes_no_options(options):
        return False
    ql = _norm(q)
    markers = (
        "any defect",
        "any abnormalities",
        "any unusual",
        "any irregular",
        "any inconsistencies",
        "any discrepancies",
        "any aberration",
        "any deviance",
        "any anomaly",
        "any incongruit",
        "departure from the standard",
        "defects compared to the reference",
        "defects in the test image",
        "inconsistencies between the test",
        "irregular occurrences in the test",
        "unusual elements in the test",
        "aberrations in the test",
        "compared to the reference normal",
        "reference normal image",
    )
    return any(m in ql for m in markers)


def _is_defect_type_question(q: str, options: dict[str, Any]) -> bool:
    if not options:
        return False
    ql = _norm(q)
    if _is_defect_existence_question(q, options):
        return False
    markers = (
        "defect",
        "inspection result",
        "inspection finding",
        "inspection status",
        "visual state",
        "visual inspection",
        "condition of the displayed",
        "outcome of the visual",
        "what does the image reveal",
        "describes the inspection",
        "best reflects the inspection",
        "accurately identifies the condition",
    )
    return any(m in ql for m in markers)


def classify_by_rules(row: dict[str, Any], *, source: str) -> tuple[str | None, str, str]:
    """
    返回 (task 或 None, confidence, reason)。
    confidence: high | medium | low
    """
    if source == "standard":
        return "standard_knowledge", "high", "source=standard"

    q = row.get("question") or ""
    options = row.get("options") or {}
    empty_opts = not options

    if empty_opts and _has_pin_count_question(q):
        return "count_pin_lead", "high", "empty_options+pin_count"

    if empty_opts and _has_component_count_question(q):
        return "count_component", "high", "empty_options+component_count"

    if empty_opts:
        return None, "low", "empty_options_unmatched"

    if _is_mount_side_question(q):
        return "mount_side", "high", "mount_side_keywords"

    if _is_shape_question(q):
        return "attribute_reasoning", "high", "shape_keywords"

    if _is_component_type_question(q):
        return "component_type", "high", "red_box_component_name"

    if _is_defect_existence_question(q, options):
        return "defect_existence", "high", "yes_no_defect_existence"

    if _is_defect_type_question(q, options):
        return "defect_type", "high", "inspection_defect_multiclass"

    # 弱规则：红框元件名但题干略偏
    ql = _norm(q)
    if "red box" in ql and any(k in ql for k in ("component", "device", "part")):
        if not _is_shape_question(q):
            return "component_type", "medium", "red_box_component_weak"

    if len(options) == 2 and _is_yes_no_options(options):
        return "defect_existence", "medium", "yes_no_fallback"

    if options:
        return "defect_type", "medium", "mcq_fallback_defect_type"

    return None, "low", "no_rule_match"


# ---------------------------------------------------------------------------
# GPT API（与 gpt_review_checkpoint.py 相同网关）
# ---------------------------------------------------------------------------


def build_api_messages(row: dict[str, Any], *, source: str) -> tuple[str, str]:
    options = row.get("options") or {}
    opt_lines = []
    for key in sorted(options, key=lambda x: str(x)):
        opt_lines.append(f"{key}. {options[key]}")
    opt_block = "\n".join(opt_lines) if opt_lines else "(none — quantitative / open numeric answer)"

    system = (
        "You classify PCBA VQA questions into exactly one evaluation task type. "
        "Output ONLY valid JSON on one line, no markdown:\n"
        '{"task":"<one_of_allowed>","confidence":"high|medium|low","reason":"<short>"}\n\n'
        "Allowed task values:\n"
        + "\n".join(f"- {t}" for t in TASK_TYPES)
        + "\n\n"
        "Critical disambiguation rules:\n"
        "- standard_knowledge: IPC/standard figure knowledge ([Figure N], Class, Acceptable/Nonconforming, "
        "compliance from standard diagrams). NOT RealWorld inspection photos unless clearly standard-doc QA.\n"
        "- component_type: identify component type/name in red box (QFN, LED, R, C, etc.).\n"
        "- mount_side: Top / Bottom / cannot determine component side.\n"
        "- defect_existence: binary Yes/No whether defect/anomaly/inconsistency exists "
        "(including vs reference normal image).\n"
        "- defect_type: multi-class inspection outcome / specific defect label "
        "(Pass, Missing component, Short Circuit, etc.).\n"
        "- count_component: count total components on board (numeric, usually no options).\n"
        "- count_pin_lead: count pins/leads of red-box primary component (numeric).\n"
        "- attribute_reasoning: geometric shape / outline of component (NOT component type name).\n"
        "- 'departure from the standard' with Yes/No => defect_existence, NOT standard_knowledge.\n"
        "- 'standard shape' / geometric outline => attribute_reasoning, NOT standard_knowledge.\n"
        "Do NOT answer the question itself; only classify."
    )

    user = (
        f"dataset_source: {source}\n"
        f"qid: {row.get('qid')}\n"
        f"question: {row.get('question')}\n"
        f"options:\n{opt_block}\n"
    )
    return system, user


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


def parse_task_json(text: str) -> tuple[str | None, str, str]:
    text = (text or "").strip()
    if not text:
        return None, "low", "empty_response"

    # 直接 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "task" in obj:
            task = _norm(str(obj["task"]))
            conf = _norm(str(obj.get("confidence") or "medium"))
            reason = str(obj.get("reason") or "api_json")
            if task in TASK_TYPES:
                return task, conf if conf in ("high", "medium", "low") else "medium", reason
    except json.JSONDecodeError:
        pass

    # 代码块或行内 JSON
    for m in re.finditer(r"\{[^{}]*\"task\"\s*:\s*\"[^\"]+\"[^{}]*\}", text, flags=re.I):
        try:
            obj = json.loads(m.group(0))
            task = _norm(str(obj.get("task")))
            if task in TASK_TYPES:
                conf = _norm(str(obj.get("confidence") or "medium"))
                return task, conf if conf in ("high", "medium", "low") else "medium", str(
                    obj.get("reason") or "api_json_embedded"
                )
        except json.JSONDecodeError:
            continue

    # 兜底：文本中出现合法 task 名
    lower = text.lower()
    found = [t for t in TASK_TYPES if t in lower]
    if len(found) == 1:
        return found[0], "low", "api_text_single_match"

    return None, "low", "parse_failed"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (row.get("status") or "ok") != "ok":
                continue
            qid = str(row.get("qid", "")).strip()
            if qid:
                done[qid] = row
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with _checkpoint_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def qid_key(row: dict[str, Any]) -> str:
    return str(row.get("qid", "")).strip()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# 单条处理
# ---------------------------------------------------------------------------


def process_one(
    row: dict[str, Any],
    *,
    source: str,
    force_api: bool,
    rules_only: bool,
) -> dict[str, Any]:
    qid = qid_key(row)
    base: dict[str, Any] = {
        "qid": qid,
        "source": source,
        "question": row.get("question"),
        "has_options": bool(row.get("options")),
    }

    t0 = time.perf_counter()

    if not force_api:
        task, conf, reason = classify_by_rules(row, source=source)
        if task and conf == "high" and not rules_only:
            elapsed = time.perf_counter() - t0
            return {
                **base,
                "status": "ok",
                "task": task,
                "confidence": conf,
                "method": "rule",
                "reason": reason,
                "elapsed_sec": round(elapsed, 3),
            }
        if rules_only and task:
            elapsed = time.perf_counter() - t0
            return {
                **base,
                "status": "ok",
                "task": task,
                "confidence": conf,
                "method": "rule",
                "reason": reason,
                "elapsed_sec": round(elapsed, 3),
            }
        rule_guess = task
        rule_conf = conf
        rule_reason = reason
    else:
        rule_guess = None
        rule_conf = "low"
        rule_reason = "force_api"

    if rules_only:
        elapsed = time.perf_counter() - t0
        if rule_guess:
            return {
                **base,
                "status": "ok",
                "task": rule_guess,
                "confidence": rule_conf,
                "method": "rule",
                "reason": rule_reason,
                "elapsed_sec": round(elapsed, 3),
            }
        return {
            **base,
            "status": "failed",
            "reason": "rules_only_no_match",
            "rule_guess": None,
            "elapsed_sec": round(elapsed, 3),
        }

    sys_msg, usr_msg = build_api_messages(row, source=source)
    try:
        raw = call_gpt_with_retry(sys_msg, usr_msg)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            **base,
            "status": "failed",
            "reason": "api_error",
            "detail": str(e),
            "rule_guess": rule_guess,
            "elapsed_sec": round(elapsed, 3),
        }

    task, conf, reason = parse_task_json(raw)
    elapsed = time.perf_counter() - t0
    if not task:
        return {
            **base,
            "status": "failed",
            "reason": reason,
            "raw_gpt": raw,
            "rule_guess": rule_guess,
            "elapsed_sec": round(elapsed, 3),
        }

    return {
        **base,
        "status": "ok",
        "task": task,
        "confidence": conf,
        "method": "api",
        "reason": reason,
        "raw_gpt": raw,
        "rule_guess": rule_guess,
        "elapsed_sec": round(elapsed, 3),
    }


def merge_output(
    rows: list[dict[str, Any]],
    labeled: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        qid = qid_key(row)
        rec = labeled.get(qid)
        item = dict(row)
        if rec and rec.get("status") == "ok":
            item["task"] = rec["task"]
            item["task_confidence"] = rec.get("confidence")
            item["task_method"] = rec.get("method")
        out.append(item)
    return out


def summarize(labeled: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {t: 0 for t in TASK_TYPES}
    other = 0
    for rec in labeled.values():
        task = rec.get("task")
        if task in counts:
            counts[task] += 1
        else:
            other += 1
    counts["_other"] = other
    return counts


# ---------------------------------------------------------------------------
# 数据集路径
# ---------------------------------------------------------------------------


def resolve_dataset_paths() -> list[tuple[str, Path, Path, Path, Path]]:
    """
    返回 [(source, input_json, checkpoint, failure, output_json), ...]
    """
    root = PCBA_CHALLENGE_ROOT.resolve()
    out_dir = OUTPUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str]] = []
    if RUN_DATASETS in ("standard", "both"):
        specs.append(("standard", STANDARD_JSON_REL))
    if RUN_DATASETS in ("realworld", "both"):
        specs.append(("realworld", REALWORLD_JSON_REL))
    if not specs:
        raise ValueError(f"无效的 RUN_DATASETS={RUN_DATASETS!r}")

    result: list[tuple[str, Path, Path, Path, Path]] = []
    for source, rel in specs:
        input_path = root / rel
        if not input_path.is_file():
            raise FileNotFoundError(f"找不到训练集: {input_path}")
        checkpoint = out_dir / f"{source}_task_type_checkpoint.jsonl"
        failure = out_dir / f"{source}_task_type_failure.jsonl"
        output_path = out_dir / f"{input_path.stem}_with_task.json"
        result.append((source, input_path, checkpoint, failure, output_path))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_one_dataset(
    *,
    source: str,
    input_path: Path,
    checkpoint_path: Path,
    failure_path: Path,
    output_path: Path,
) -> int:
    rows = load_json_array(input_path)
    if QID:
        rows = [r for r in rows if qid_key(r) == str(QID).strip()]
        if not rows:
            print(f"[{source}] 未找到 qid={QID!r}", file=sys.stderr)
            return 2
    elif LIMIT is not None:
        rows = rows[:LIMIT]

    done: dict[str, dict[str, Any]] = {}
    if not NO_RESUME:
        done = load_checkpoint(checkpoint_path)

    pending = [r for r in rows if qid_key(r) not in done]
    log(
        f"\n=== {source} ===\n"
        f"输入: {input_path}\n"
        f"共 {len(rows)} 条 | 已完成 {len(rows) - len(pending)} | 待处理 {len(pending)} | "
        f"model={MODEL} workers={WORKERS} | rules_only={RULES_ONLY} api_all={API_ALL}"
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
                log(
                    f"[{source}:{qid}] OK task={record['task']} "
                    f"method={record.get('method')} conf={record.get('confidence')} "
                    f"({record.get('elapsed_sec')}s)"
                )
            return

        append_jsonl(failure_path, record)
        run_failures.append(qid)
        if not QUIET:
            log(f"[{source}:{qid}] FAIL reason={record.get('reason')} ({record.get('elapsed_sec')}s)")

    if pending:
        if WORKERS <= 1:
            for i, row in enumerate(pending, 1):
                if not QUIET:
                    log(f"({i}/{len(pending)}) {source} qid={qid_key(row)} ...")
                handle(
                    process_one(
                        row,
                        source=source,
                        force_api=API_ALL,
                        rules_only=RULES_ONLY,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futures = {
                    ex.submit(
                        process_one,
                        row,
                        source=source,
                        force_api=API_ALL,
                        rules_only=RULES_ONLY,
                    ): row
                    for row in pending
                }
                for fut in as_completed(futures):
                    handle(fut.result())

    merged = merge_output(rows, done)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    counts = summarize(done)
    log(f"[{source}] 已写出: {output_path}")
    log(f"[{source}] checkpoint: {checkpoint_path} ({len(done)} 条)")
    log(
        f"[{source}] 任务分布: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v > 0)
    )

    missing = [qid_key(r) for r in rows if qid_key(r) not in done]
    if missing:
        log(f"[{source}] 仍缺 {len(missing)} 条，重新运行可续跑")
        if run_failures:
            log(
                f"[{source}] 本轮失败: {', '.join(run_failures[:20])}"
                + (" ..." if len(run_failures) > 20 else "")
            )
        return 1
    return 0


def main() -> int:
    if not PCBA_CHALLENGE_ROOT.is_dir():
        print(f"PCBA_CHALLENGE_ROOT 不存在: {PCBA_CHALLENGE_ROOT}", file=sys.stderr)
        return 2

    datasets = resolve_dataset_paths()
    log(f"PCBA_CHALLENGE_ROOT = {PCBA_CHALLENGE_ROOT.resolve()}")
    log(f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    log(f"RUN_DATASETS = {RUN_DATASETS}")

    exit_code = 0
    for source, input_path, ckpt, fail, out in datasets:
        code = run_one_dataset(
            source=source,
            input_path=input_path,
            checkpoint_path=ckpt,
            failure_path=fail,
            output_path=out,
        )
        if code != 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
