#!/usr/bin/env python3
"""
全页穷举：不做 RAG / MinerU，直接对原始 PDF 逐页筛查 + 汇总作答（只求准确率）。

流程（按 file_id 分组，同一 PDF 上逐题处理）：
  1) 将该 PDF 每一页渲染为图，与设问一起送入 VLM，判断是否有相关信息；
     无则该页记为「无」，有则抽取与设问相关的事实文本。
  2) 汇总作答（两阶段，提高计算/比较题准确率）：
     a) 基于各页抽取文本，逐步推理并显式计算（<reasoning>）；
     b) 从推理结论中提取最终 <answer> 与 <evidence>。

依赖：transformers、torch、vllm、PyMuPDF（fitz）、qwen_vl_utils（可选）

示例:
  cd rag
  python3 pdf_qwen_exhaustive_pages.py

环境变量:
  MODEL_PATH          默认 Qwen3.5-27B
  QUESTIONS_CSV       默认 lava/test.csv
  PDF_DIR             默认 lava/test_pdfs/test_pdfs
  PDF_DPI             渲染 DPI，默认 200
  MUPDF_DISPLAY_ERRORS  设为 1 可在 stderr 显示 MuPDF 警告/错误（默认静默）
  PAGE_BATCH_SIZE     每题逐页推理的 vLLM batch 大小，默认 64
  TENSOR_PARALLEL_SIZE  张量并行 GPU 数；0=自动用全部可见卡（默认）
  TWO_STAGE_FINAL=1   汇总先推理再作答（默认开启；0 关闭）
  REASONING_MAX_TOKENS  推理阶段 max_tokens，默认 12000
  SUBMISSION_CSV / OUT_JSON / CHECKPOINT_JSONL / RUN_META_JSON（运行起止与耗时）
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from utils import (
    apply_generation_prompt_with_brief_thinking,
    apply_generation_prompt_without_thinking,
    build_messages,
    dump_list,
    fallback_list,
    normalize_submission_answer,
    parse_answer_tag,
    parse_evidence,
    parse_system_msg,
    parse_user_msg,
    pdf_page_count,
    prepare_mm_data,
    release_torch_memory,
    render_pdf_pages_to_png_paths,
    set_random_seed,
    system_msg_ja,
    system_msg_vi,
    try_parse_list,
)

_DEFAULT_MODEL = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/hf_download/Qwen3.5-27B"
)
_DEFAULT_QUESTIONS_CSV = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/lava/test.csv"
)
_DEFAULT_PDF_DIR = (
    "/inspire/qb-ilm/project/traffic-congestion-management/"
    "xiacheng-240108120111/lava/test_pdfs/test_pdfs"
)

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
QUESTIONS_CSV = os.environ.get(
    "QUESTIONS_CSV",
    os.environ.get("TEST_CSV", _DEFAULT_QUESTIONS_CSV),
)
PDF_DIR = os.environ.get("PDF_DIR", _DEFAULT_PDF_DIR)
SUBMISSION_CSV = os.environ.get("SUBMISSION_CSV", "submission_exhaustive.csv")
OUT_JSON = os.environ.get("OUT_JSON", "pdf_exhaustive_pred.json")
CHECKPOINT_JSONL = os.environ.get("CHECKPOINT_JSONL", "pdf_exhaustive_checkpoint.jsonl")
SEED = int(os.environ.get("SEED", "42"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
PDF_DPI = int(os.environ.get("PDF_DPI", "200"))
PAGE_BATCH_SIZE = int(os.environ.get("PAGE_BATCH_SIZE", "64"))
_TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "0"))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "64000"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
LIMIT_MM_PER_PROMPT = {"image": 4, "video": 0}

SAMPLING_PAGE = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    max_tokens=int(os.environ.get("PAGE_MAX_TOKENS", "2048")),
    seed=SEED,
)
SAMPLING_REASONING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    max_tokens=int(os.environ.get("REASONING_MAX_TOKENS", "12000")),
    seed=SEED,
)
SAMPLING_FINAL = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    max_tokens=int(os.environ.get("FINAL_MAX_TOKENS", "1024")),
    seed=SEED,
)
TWO_STAGE_FINAL = os.environ.get("TWO_STAGE_FINAL", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
SAMPLING_PARSE = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    max_tokens=256,
    seed=SEED,
)
SAMPLING_LIST_FIX = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    max_tokens=int(os.environ.get("LIST_FIX_MAX_TOKENS", "512")),
    seed=SEED,
)

NO_INFO = "无"
_MIN_EXTRACT_CHARS = int(os.environ.get("MIN_EXTRACT_CHARS", "8"))
_EMPTY_EXTRACT_MARKERS = frozenset(
    {
        "无", "無", "なし", "ない", "none", "n/a", "na",
        "該当なし", "関連なし", "関連情報なし", "情報なし",
        "không", "khong", "không có", "khong co",
    }
)
_RELEVANT_YES_EXACT = frozenset(
    {"はい", "yes", "y", "true", "1", "有", "有り", "あり", "是", "对", "對", "có", "co"}
)
_RELEVANT_NO_EXACT = frozenset(
    {"いいえ", "no", "n", "false", "0", "否", "无", "無", "không", "khong"}
)


def _wall_time_str(ts: Optional[float] = None) -> str:
    """本地 wall-clock 时间字符串。"""
    return datetime.fromtimestamp(ts or time.time()).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds: float) -> str:
    """人类可读时长：X小时Y分Z秒 / Y分Z秒 / Z秒。"""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def _print_run_banner(phase: str, wall_start: float, wall_end: Optional[float] = None) -> None:
    """打印脚本开始/结束时间横幅。"""
    if phase == "start":
        print(f"[脚本开始] 本地时间: {_wall_time_str(wall_start)}")
        return
    assert wall_end is not None
    elapsed = wall_end - wall_start
    print(f"[脚本结束] 本地时间: {_wall_time_str(wall_end)}")
    print(
        f"[脚本总耗时] {elapsed:.3f}s（{_format_duration(elapsed)}，"
        f"{elapsed / 60.0:.2f} 分钟）"
    )
    print(f"[起止时间] {_wall_time_str(wall_start)} → {_wall_time_str(wall_end)}")


def format_evidence_column(pages: List[int]) -> str:
    """与 sample_submission.csv 中 evidence_page_number 列风格一致，如 [1]、[1,2]。"""
    if not pages:
        return "[]"
    return "[" + ",".join(str(p) for p in pages) + "]"


def _resolve(path: str, script_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def count_visible_gpus() -> int:
    """可见 GPU 数量（已考虑 CUDA_VISIBLE_DEVICES）。"""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def resolve_tensor_parallel_size() -> int:
    """0 表示自动使用全部可见 GPU；>0 则使用指定张量并行度。"""
    visible = count_visible_gpus()
    if _TENSOR_PARALLEL_SIZE > 0:
        return _TENSOR_PARALLEL_SIZE
    return max(1, visible) if visible > 0 else 1


def resolve_pdf_path(pdf_dir: str, file_id: str) -> str:
    """在 PDF_DIR 下按 file_id 查找原始 PDF。"""
    direct = [
        os.path.join(pdf_dir, f"{file_id}.pdf"),
        os.path.join(pdf_dir, f"{file_id}_origin.pdf"),
        os.path.join(pdf_dir, file_id, f"{file_id}.pdf"),
        os.path.join(pdf_dir, file_id, f"{file_id}_origin.pdf"),
    ]
    for p in direct:
        if os.path.isfile(p):
            return os.path.abspath(p)
    pattern = os.path.join(pdf_dir, "**", f"{file_id}*.pdf")
    hits = sorted(glob.glob(pattern, recursive=True))
    if hits:
        return os.path.abspath(hits[0])
    raise FileNotFoundError(
        f"未找到 PDF: file_id={file_id}，已查 {pdf_dir} 及子目录 {file_id}*.pdf"
    )


def system_msg_for_lang(lang: str) -> str:
    if (lang or "").strip().lower() == "vi":
        return system_msg_vi()
    return system_msg_ja()


def system_msg_page_screen(lang: str) -> str:
    """逐页筛查专用 system：强调高召回，避免漏页。"""
    if (lang or "").strip().lower() == "vi":
        return (
            "Bạn là trợ lý trích xuất thông tin từng trang PDF. "
            "Mục tiêu là **độ nhạy cao (high recall)**: thà nhận nhầm còn hơn bỏ sót trang có manh mối. "
            "Một trang được coi là liên quan nếu có bất kỳ từ khóa, số liệu, bảng, biểu đồ, "
            "chú thích, địa danh, ngày tháng, tên riêng… có thể giúp trả lời hoặc suy luận. "
            "Chỉ đánh dấu không liên quan khi trang hoàn toàn trống/không đọc được/không có gì "
            "giao với câu hỏi. Khi không chắc, hãy đánh dấu liên quan và trích xuất. "
            "Chỉ xuất đúng các thẻ được yêu cầu."
        )
    return (
        "あなたは PDF 各ページから情報を抽出するアシスタントです。"
        "**高い再現率（recall）**を最優先し、関連ページの取りこぼしを避けてください。"
        "設問のキーワード・数値・日付・固有名詞・表・グラフ・地図・脚注など、"
        "回答や推論に**役立ちうる**内容が1つでもあれば関連ありとします。"
        "このページ単独では答えが完結しなくても構いません。間接的な手がかりも含めてください。"
        "完全に無関係（空白・装飾のみ・設問と分野がまったく交わらない）の場合のみ関連なし。"
        "迷ったら関連ありとして可能な限り抽出してください。"
        "指定タグ以外は出力しないでください。"
    )


def answer_format_hint(answer_format: str, lang: str) -> str:
    af = (answer_format or "string").strip()
    if (lang or "").strip().lower() == "vi":
        m = {
            "string": "một câu trả lời ngắn (tên, cụm từ, thời gian…)",
            "number": "một con số kèm đơn vị nếu có",
            "ordered_list": "danh sách có thứ tự (nhiều mục)",
            "unordered_list": "danh sách không cần thứ tự (nhiều mục)",
        }
        return m.get(af, m["string"])
    m = {
        "string": "短い文字列（名称・時刻・固有名詞など）",
        "number": "数値（必要なら単位付き）",
        "ordered_list": "順序のある複数項目",
        "unordered_list": "順不同の複数項目",
    }
    return m.get(af, m["string"])


def prompt_page_screen(
    question: str,
    page_num: int,
    answer_format: str,
    lang: str,
    *,
    lenient: bool = False,
) -> str:
    """逐页：宽松判定 + 尽量完整抽取（高 recall）。"""
    fmt_hint = answer_format_hint(answer_format, lang)
    extra_ja = (
        "【重要】今回は前回すべて「関連なし」だったため、さらに寛容に判定してください。"
        "少しでも設問と接点があれば必ず「はい」にし、見える文字・数字をすべて書き出してください。"
        if lenient
        else ""
    )
    extra_vi = (
        "【Quan trọng】Lần trước mọi trang đều không liên quan; hãy đánh giá rộng hơn nữa. "
        "Chỉ cần có chút liên quan thì đánh dấu có và ghi lại mọi chữ/số nhìn thấy."
        if lenient
        else ""
    )
    if (lang or "").strip().lower() == "vi":
        return (
            f"{extra_vi}\n"
            f"Ảnh dưới đây là trang {page_num} của PDF (đếm từ 1).\n\n"
            f"Câu hỏi:\n{question}\n\n"
            f"Loại câu trả lời mong đợi: {fmt_hint}\n\n"
            "Nhiệm vụ: tìm **mọi** thông tin trên trang có thể liên quan đến câu hỏi "
            "(từ khóa, số, bảng, biểu đồ, chú thích, tên riêng, ngày tháng…). "
            "Không cần trang này trả lời trọn vẹn; manh mối gián tiếp cũng được.\n"
            "Chỉ xuất đúng hai thẻ (không giải thích, không Markdown):\n"
            "Nếu có bất kỳ manh mối nào:\n"
            "  <relevant>có</relevant>\n"
            "  <extract>（liệt kê chi tiết mọi sự kiện/số/chữ nhìn thấy trên trang）</extract>\n"
            "Chỉ khi trang hoàn toàn không liên quan hoặc không đọc được:\n"
            "  <relevant>không</relevant>\n"
            f"  <extract>{NO_INFO}</extract>\n"
            "Khi không chắc, chọn có và trích xuất."
        )
    return (
        f"{extra_ja}\n"
        f"以下の画像は PDF の第 {page_num} ページ（1 始まり）です。\n\n"
        f"設問：\n{question}\n\n"
        f"期待する回答の型：{fmt_hint}\n\n"
        "【タスク】このページ画像から、設問に**役立ちうる**情報をすべて探し出してください。"
        "本文・見出し・表のセル・グラフの軸・凡例・地図上の名称・脚注・日付・数値など、"
        "読み取れるものは可能な限り具体的に書き出します。"
        "この1ページだけでは答えが完結しなくても構いません。手がかりレベルでも関連ありです。\n"
        "出力は次の2タグのみ（説明・Markdown 禁止）：\n"
        "手がかりが1つでもある場合：\n"
        "  <relevant>はい</relevant>\n"
        "  <extract>（ページ上の関連事実を箇条書きまたは短文で、できるだけ詳しく）</extract>\n"
        "ページが完全に無関係、または判読不能な場合のみ：\n"
        "  <relevant>いいえ</relevant>\n"
        f"  <extract>{NO_INFO}</extract>\n"
        "迷った場合は必ず「はい」として抽出してください。"
    )


_CALC_QUESTION_RE = re.compile(
    r"比べ|差|開き|最も|最大|最小|何倍|合計|平均|どちら|より.*?[多少]|"
    r"何[%％]|何ポイント|何か月|何年|何人|何ℓ|何トン|何万|何億|"
    r"割合|比率|構成比|増減|何位|第[一二三四五六七八九十]",
    re.I,
)


def filter_extractions_for_final(
    page_extractions: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """汇总阶段只保留有实质内容的页，减少噪声。"""
    out: List[Tuple[int, str]] = []
    for page_num, text in page_extractions:
        t = (text or "").strip()
        if _is_empty_extract(t):
            continue
        out.append((page_num, t))
    return out


def needs_explicit_calculation(question: str) -> bool:
    return bool(_CALC_QUESTION_RE.search(question or ""))


def format_extraction_blocks(
    page_extractions: List[Tuple[int, str]], lang: str
) -> str:
    if not page_extractions:
        return f"（有用な抽出はありません / すべて「{NO_INFO}」）"
    if (lang or "").strip().lower() == "vi":
        return "\n\n".join(
            f"【Trang {p}】\n{t.strip()}" for p, t in page_extractions
        )
    return "\n\n".join(
        f"【ページ {p}】\n{t.strip()}" for p, t in page_extractions
    )


def system_msg_final_reasoning(lang: str) -> str:
    if (lang or "").strip().lower() == "vi":
        return (
            "Bạn là chuyên gia phân tích tài liệu, làm việc cẩn thận và chính xác. "
            "Phải suy luận từng bước; khi cần so sánh/tính toán, hãy ghi rõ công thức và kết quả. "
            "Dữ liệu có thể nằm ở nhiều trang — cần ghép đúng cùng năm/cùng phân loại. "
            "Không dùng kiến thức bên ngoài. Chỉ xuất thẻ <reasoning>."
        )
    return (
        "あなたは文書分析の専門家です。**正確性**を最優先し、推論は必ず段階的に行ってください。"
        "比較・差・最大/最小・合計・割合などが問われる場合は、**必ず数値を取り出して計算過程を明示**してください。"
        "情報が複数ページに分散している場合は、同じ年・同じ産業区分など**対応関係を揃えてから**統合してください。"
        "GDP比率と就業者比率のように別表の数値を比べるときは、各区分ごとに差の絶対値を計算し、"
        "設問が求める「最も大きい」等を**計算結果に基づいて**決めてください（単一表の最大値と混同しない）。"
        "外部知識は使わず、与えられた抽出テキストのみに依拠してください。"
        "出力は <reasoning> タグのみ。"
    )


def calc_hint_block(question: str, lang: str) -> str:
    if not needs_explicit_calculation(question):
        return ""
    if (lang or "").strip().lower() == "vi":
        return (
            "\n【Lưu ý tính toán】Câu hỏi yêu cầu so sánh/tính toán. "
            "Hãy: (1) liệt kê số liệu theo từng trang; (2) căn chỉnh cùng năm/phân loại; "
            "(3) tính từng bước (ví dụ |A−B|, tổng, max/min); (4) kết luận dựa trên kết quả tính.\n"
        )
    return (
        "\n【計算・比較の注意】本設問は数値の比較・差・最大/最小を含みます。"
        "必ず次を実行してください：\n"
        "1) 各【ページ N】から使う数値・表の値を列挙\n"
        "2) 複数ページのデータは、**同じ年・同じ区分名**で対応付け（名称の違いに注意）\n"
        "3) 設問が「開き」「差」「最も大きい」等を求める場合、**各候補について計算式と結果**を示す"
        "（例：|就業者比率−GDP比率| を各産業で計算し、最大の産業を特定）\n"
        "4) 結論は計算結果にのみ基づく（推測で答えない）\n"
    )


def prompt_final_reasoning(
    question: str,
    answer_format: str,
    page_extractions: List[Tuple[int, str]],
    lang: str,
) -> str:
    """阶段 2a：逐步推理 + 显式计算。"""
    blocks = format_extraction_blocks(page_extractions, lang)
    fmt_hint = answer_format_hint(answer_format, lang)
    calc_hint = calc_hint_block(question, lang)
    if (lang or "").strip().lower() == "vi":
        return (
            "Dưới đây là nội dung trích xuất từ PDF (theo trang, đếm từ 1).\n\n"
            f"{blocks}\n\n"
            f"Câu hỏi:\n{question}\n\n"
            f"Định dạng câu trả lời cuối cùng sẽ là: {fmt_hint}\n"
            f"{calc_hint}\n"
            "Nhiệm vụ: suy luận từng bước trong <reasoning>:\n"
            "- Bước 1: câu hỏi đang hỏi gì\n"
            "- Bước 2: liệt kê sự kiện/số liệu theo từng trang (ghi rõ 【Trang N】)\n"
            "- Bước 3: ghép dữ liệu nhiều trang (cùng năm/phân loại)\n"
            "- Bước 4: nếu cần, tính toán/so sánh rõ ràng (bảng hoặc công thức)\n"
            "- Bước 5: kết luận cuối (một câu, khớp định dạng đáp án)\n"
            "Chỉ xuất:\n<reasoning>...</reasoning>\n"
        )
    return (
        "以下は PDF 各ページから抽出したテキストです（【ページ N】= PDF 上のページ番号、1始まり）。\n\n"
        f"{blocks}\n\n"
        f"設問：\n{question}\n\n"
        f"最終的に必要な回答形式：{fmt_hint}\n"
        f"{calc_hint}\n"
        "【タスク】<reasoning> 内で次の順に**必ず**推論してください：\n"
        "ステップ1：設問が何を求めているか（比較対象・対象年・単位など）\n"
        "ステップ2：各【ページ N】から使える事実・数値を列挙（ページ番号を明記）\n"
        "ステップ3：複数ページの情報を統合（同じ年・同じ産業区分など対応関係を確認）\n"
        "ステップ4：計算・比較が必要なら、各式と結果を明示（表形式推奨）。"
        "「最も大きい/小さい」「差/開き」は必ず全候補を計算してから決定\n"
        "ステップ5：最終結論（回答形式に合う短い答え）と、根拠ページ番号\n"
        "出力は <reasoning> のみ。Markdown 可。\n"
        "<reasoning>\n（ここに段階的推論）\n</reasoning>\n"
    )


def prompt_final_answer_from_reasoning(
    question: str,
    answer_format: str,
    reasoning: str,
    lang: str,
) -> str:
    """阶段 2b：从推理结论提取结构化答案。"""
    fmt_ja = {
        "string": "最終回答は1行の短文のみ（産業名・固有名詞など、設問が求める形式）。",
        "number": "問題が求める数値・単位のみを1行で。",
        "ordered_list": "順序付きリストを1行の Python リストリテラルで。",
        "unordered_list": "順不同リストを1行の Python リストリテラルで。",
    }
    fmt_vi = {
        "string": "Một dòng câu trả lời cuối cùng.",
        "number": "Một dòng: số/đơn vị theo đề bài.",
        "ordered_list": "Một dòng list Python có thứ tự.",
        "unordered_list": "Một dòng list Python không cần thứ tự.",
    }
    if (lang or "").strip().lower() == "vi":
        fmt = fmt_vi.get(answer_format, fmt_vi["string"])
        return (
            "Dưới đây là quá trình suy luận đã hoàn tất. "
            "Hãy trích xuất câu trả lời cuối và trang bằng chứng — "
            "**phải khớp với kết luận ở bước cuối của reasoning**, không đổi ý.\n\n"
            f"Câu hỏi:\n{question}\n\n"
            f"Định dạng: {fmt}\n\n"
            f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
            "Chỉ xuất hai thẻ:\n"
            "<answer>...</answer>\n"
            "<evidence>[các số trang PDF, đếm từ 1]</evidence>\n"
        )
    fmt = fmt_ja.get(answer_format, fmt_ja["string"])
    return (
        "以下は完了した推論過程です。**最終結論（ステップ5）と矛盾しない** "
        "answer と evidence を抽出してください。推論で計算した結論を変更しないでください。\n\n"
        f"設問：\n{question}\n\n"
        f"出力形式：{fmt}\n\n"
        f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
        "出力は次の2タグのみ：\n"
        "<answer>推論の最終結論</answer>\n"
        "<evidence>[根拠となった PDF ページ番号の整数リスト]</evidence>\n"
    )


def prompt_final_from_extractions(
    question: str,
    answer_format: str,
    page_extractions: List[Tuple[int, str]],
    lang: str,
) -> str:
    """单阶段汇总（TWO_STAGE_FINAL=0 时的后备）。"""
    blocks = format_extraction_blocks(page_extractions, lang)
    calc_hint = calc_hint_block(question, lang)
    fmt_ja = {
        "string": "最終回答は1行の短文のみ。",
        "number": "問題が求める数値・単位のみを1行で。",
        "ordered_list": "順序付きリストを1行の Python リストリテラルで、例: ['甲','乙']。",
        "unordered_list": "順不同リストを1行の Python リストリテラルで。",
    }
    fmt_vi = {
        "string": "Một dòng câu trả lời cuối cùng.",
        "number": "Một dòng: số/đơn vị theo đề bài.",
        "ordered_list": "Một dòng list Python có thứ tự.",
        "unordered_list": "Một dòng list Python không cần thứ tự.",
    }
    if (lang or "").strip().lower() == "vi":
        fmt = fmt_vi.get(answer_format, fmt_vi["string"])
        return (
            "Dưới đây là nội dung trích xuất từng trang PDF.\n"
            f"{calc_hint}\n"
            f"{blocks}\n\n"
            f"Câu hỏi:\n{question}\n\n"
            f"Yêu cầu định dạng: {fmt}\n"
            "<answer>...</answer>\n"
            "<evidence>[...]</evidence>\n"
        )
    fmt = fmt_ja.get(answer_format, fmt_ja["string"])
    return (
        "以下は PDF 各ページから抽出したテキストです。\n"
        f"{calc_hint}\n"
        f"{blocks}\n\n"
        f"設問：\n{question}\n\n"
        f"出力要件：{fmt}\n"
        "<answer>...</answer>\n"
        "<evidence>[...]</evidence>\n"
    )


def parse_reasoning_tag(text: str) -> str:
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text or "", flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def evidence_from_reasoning(reasoning: str) -> List[int]:
    """从推理文本中 salvage 页码引用。"""
    pages: List[int] = []
    seen: set[int] = set()
    for m in re.finditer(
        r"(?:【ページ|【Trang|ページ|trang)\s*(\d+)",
        reasoning or "",
        flags=re.I,
    ):
        p = int(m.group(1))
        if p > 0 and p not in seen:
            seen.add(p)
            pages.append(p)
    return pages


def run_final_answer(
    llm: LLM,
    processor: Any,
    question: str,
    answer_format: str,
    page_extractions: List[Tuple[int, str]],
    lang: str,
) -> Tuple[str, str, str]:
    """
    汇总作答。返回 (reasoning_text, final_raw, answer_raw)。
    final_raw 含 <answer>/<evidence>；单阶段时 reasoning_text 为空。
    """
    useful = filter_extractions_for_final(page_extractions)
    if not useful:
        useful = page_extractions

    if TWO_STAGE_FINAL:
        reason_user = prompt_final_reasoning(
            question, answer_format, useful, lang
        )
        reason_msgs = build_messages(
            system_msg_final_reasoning(lang), reason_user, image_paths=None
        )
        reason_prompt = apply_generation_prompt_with_brief_thinking(
            processor, reason_msgs
        )
        reasoning_out = llm.generate(
            [{"prompt": reason_prompt}],
            sampling_params=SAMPLING_REASONING,
        )[0].outputs[0].text
        reasoning_text = parse_reasoning_tag(reasoning_out) or reasoning_out

        ans_user = prompt_final_answer_from_reasoning(
            question, answer_format, reasoning_text, lang
        )
        ans_msgs = build_messages(
            system_msg_for_lang(lang), ans_user, image_paths=None
        )
        ans_prompt = apply_generation_prompt_without_thinking(processor, ans_msgs)
        final_out = llm.generate(
            [{"prompt": ans_prompt}],
            sampling_params=SAMPLING_FINAL,
        )[0].outputs[0].text
        return reasoning_text, final_out, final_out

    final_user = prompt_final_from_extractions(
        question, answer_format, useful, lang
    )
    final_msgs = build_messages(
        system_msg_for_lang(lang), final_user, image_paths=None
    )
    final_prompt = apply_generation_prompt_with_brief_thinking(
        processor, final_msgs
    )
    final_out = llm.generate(
        [{"prompt": final_prompt}],
        sampling_params=SAMPLING_REASONING,
    )[0].outputs[0].text
    return "", final_out, final_out


def _normalize_token(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())


def _is_empty_extract(extract: str) -> bool:
    s = (extract or "").strip()
    if not s:
        return True
    if s in _EMPTY_EXTRACT_MARKERS:
        return True
    if len(s) <= 3 and s.lower() in ("na", "n/a"):
        return True
    return False


def _extract_from_untagged(raw: str) -> str:
    """模型未按标签输出时，尽量 salvage 正文。"""
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"</?(?:relevant|extract|think|thinking)[^>]*>", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if _is_empty_extract(text):
        return ""
    return text


def parse_page_stage_output(text: str) -> Tuple[bool, str]:
    """解析逐页输出 → (是否相关, 抽取文本或「无」)。内容优先于标签。"""
    raw = (text or "").strip()
    rel_m = re.search(r"<relevant>\s*(.*?)\s*</relevant>", raw, flags=re.S | re.I)
    ext_m = re.search(r"<extract>\s*(.*?)\s*</extract>", raw, flags=re.S | re.I)
    rel_token = _normalize_token(rel_m.group(1) if rel_m else "")
    extract = ext_m.group(1).strip() if ext_m else ""

    if not extract:
        extract = _extract_from_untagged(raw)

    tag_relevant: Optional[bool] = None
    if rel_token in _RELEVANT_YES_EXACT:
        tag_relevant = True
    elif rel_token in _RELEVANT_NO_EXACT:
        tag_relevant = False
    elif rel_token.startswith("はい") or rel_token in ("yes", "có"):
        tag_relevant = True
    elif rel_token.startswith("いいえ") or rel_token.startswith("khong") or rel_token.startswith("không"):
        tag_relevant = False

    has_content = (
        bool(extract)
        and not _is_empty_extract(extract)
        and len(extract.strip()) >= _MIN_EXTRACT_CHARS
    )

    # 有足够抽取内容 → 一律视为相关（修复「标了いいえ但 extract 有字」的漏检）
    if has_content:
        return True, extract

    if tag_relevant is True and extract:
        return True, extract if not _is_empty_extract(extract) else NO_INFO

    if tag_relevant is False:
        return False, NO_INFO

    # 无标签、无足够内容
    return False, NO_INFO


def list_fix_system_msg() -> str:
    return (
        "你是一个 CSV 提交答案格式规范化助手。"
        "把 raw_answer 解析成符合 answer_format 的列表。"
        "只输出 <answer>...</answer>，内容为严格 JSON array。"
    )


def list_fix_user_msg(question: str, answer_format: str, raw_answer: str) -> str:
    return (
        f"question: {question}\n"
        f"answer_format: {answer_format}\n"
        f"raw_answer: {raw_answer}\n"
    )


def load_checkpoint(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("id")
            if qid:
                done[qid] = rec
    return done


def append_checkpoint(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_page_batches(
    llm: LLM,
    processor: Any,
    jobs: List[Dict[str, Any]],
    batch_size: int,
    *,
    lenient: bool = False,
) -> List[str]:
    """jobs: {page_idx, page_num, png_path, question, language, answer_format}"""
    outputs: List[str] = [""] * len(jobs)
    sys_cache: Dict[str, str] = {}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        inputs: List[Dict[str, Any]] = []
        for job in chunk:
            lang = job["language"]
            if lang not in sys_cache:
                sys_cache[lang] = system_msg_page_screen(lang)
            user_text = prompt_page_screen(
                job["question"],
                job["page_num"],
                job.get("answer_format", "string"),
                lang,
                lenient=lenient,
            )
            msgs = build_messages(sys_cache[lang], user_text, [job["png_path"]])
            prompt = apply_generation_prompt_without_thinking(processor, msgs)
            llm_in: Dict[str, Any] = {"prompt": prompt}
            mm = prepare_mm_data(msgs, [job["png_path"]])
            if mm:
                llm_in["multi_modal_data"] = mm
            inputs.append(llm_in)
        outs = llm.generate(inputs, sampling_params=SAMPLING_PAGE)
        for i, out in enumerate(outs):
            outputs[start + i] = out.outputs[0].text
    return outputs


def screen_all_pages(
    llm: LLM,
    processor: Any,
    page_jobs: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]], int]:
    """逐页筛查；若 0 命中则用更宽松 prompt 重跑一遍。"""
    page_raws = run_page_batches(llm, processor, page_jobs, batch_size, lenient=False)
    page_details, page_extractions, n_hit = _build_page_results(page_jobs, page_raws)

    if n_hit == 0 and page_jobs:
        print("    [warn] 相关页=0，启用宽松模式重跑全部页面…")
        retry_raws = run_page_batches(
            llm, processor, page_jobs, batch_size, lenient=True
        )
        page_details, page_extractions, n_hit = _build_page_results(
            page_jobs, retry_raws
        )
        for d in page_details:
            d["retried_lenient"] = True

    return page_details, page_extractions, n_hit


def _build_page_results(
    page_jobs: List[Dict[str, Any]],
    page_raws: List[str],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]], int]:
    page_details: List[Dict[str, Any]] = []
    page_extractions: List[Tuple[int, str]] = []
    for job, raw in zip(page_jobs, page_raws):
        relevant, extract = parse_page_stage_output(raw)
        page_details.append(
            {
                "page_num": job["page_num"],
                "relevant": relevant,
                "extract": extract,
                "raw": raw,
            }
        )
        page_extractions.append((job["page_num"], extract))
    n_hit = sum(1 for d in page_details if d["relevant"])
    return page_details, page_extractions, n_hit


def main() -> None:
    wall_start = time.time()
    t0 = time.perf_counter()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = _resolve(QUESTIONS_CSV, script_dir)
    pdf_dir = _resolve(PDF_DIR, script_dir)
    ckpt_path = _resolve(CHECKPOINT_JSONL, script_dir)
    sub_path = _resolve(SUBMISSION_CSV, script_dir)
    out_json_path = _resolve(OUT_JSON, script_dir)

    _print_run_banner("start", wall_start)
    set_random_seed(SEED)
    tp_size = resolve_tensor_parallel_size()
    visible_gpus = count_visible_gpus()
    if tp_size > 1 and visible_gpus < tp_size:
        raise ValueError(
            f"TENSOR_PARALLEL_SIZE={tp_size} 超过可见 GPU 数 {visible_gpus}"
        )

    print(f"MODEL_PATH={MODEL_PATH}")
    print(
        f"TWO_STAGE_FINAL={'开启（推理→作答）' if TWO_STAGE_FINAL else '关闭（单阶段）'} | "
        f"REASONING_MAX_TOKENS={SAMPLING_REASONING.max_tokens}"
    )
    print(
        f"TENSOR_PARALLEL_SIZE={tp_size} "
        f"(env={_TENSOR_PARALLEL_SIZE or 'auto'}, visible_gpus={visible_gpus})"
    )
    if visible_gpus > 0:
        for i in range(visible_gpus):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({mem_gb:.1f} GiB)")
    print(f"QUESTIONS_CSV={csv_path}")
    print(f"PDF_DIR={pdf_dir} | PDF_DPI={PDF_DPI}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if MAX_SAMPLES > 0:
        rows = rows[:MAX_SAMPLES]
    print(f"共 {len(rows)} 题")

    rows_by_fid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_fid[row["file_id"]].append(row)

    pdf_meta: Dict[str, Tuple[str, int]] = {}
    for fid in rows_by_fid:
        p = resolve_pdf_path(pdf_dir, fid)
        n = pdf_page_count(p)
        if n <= 0:
            raise ValueError(f"PDF 无页: {p}")
        pdf_meta[fid] = (p, n)
        print(f"  PDF {fid}: {n} 页 -> {p}")

    checkpoint = load_checkpoint(ckpt_path)
    if checkpoint:
        print(f"从 checkpoint 恢复: 已完成 {len(checkpoint)} 题 -> {ckpt_path}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    t_llm = time.perf_counter()
    print(
        f"加载 VLM: {MODEL_PATH} "
        f"(tensor_parallel_size={tp_size}, MAX_NUM_SEQS={MAX_NUM_SEQS})"
    )
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
        seed=SEED,
    )
    print(f"[timing] LLM 加载: {time.perf_counter() - t_llm:.3f}s")

    all_temp_png: List[str] = []
    results_by_id: Dict[str, Dict[str, Any]] = dict(checkpoint)

    for fid in sorted(rows_by_fid.keys()):
        pdf_path, page_count = pdf_meta[fid]
        print(f"\n=== file_id={fid} ({page_count} 页, {len(rows_by_fid[fid])} 题) ===")

        page_pngs = render_pdf_pages_to_png_paths(
            pdf_path, list(range(page_count)), PDF_DPI
        )
        all_temp_png.extend(page_pngs)
        if len(page_pngs) != page_count:
            raise RuntimeError(
                f"渲染页数不符: 期望 {page_count}, 得到 {len(page_pngs)}"
            )

        for row in rows_by_fid[fid]:
            qid = row["id"]
            if qid in results_by_id:
                print(f"  跳过已完成: {qid}")
                continue

            lang = (row.get("language") or "ja").strip().lower()
            question = row["question"]
            afmt = (row.get("answer_format") or "string").strip()
            t_q = time.perf_counter()

            page_jobs: List[Dict[str, Any]] = []
            for page_idx, png in enumerate(page_pngs):
                page_jobs.append(
                    {
                        "page_idx": page_idx,
                        "page_num": page_idx + 1,
                        "png_path": png,
                        "question": question,
                        "language": lang,
                        "answer_format": afmt,
                    }
                )

            print(f"  [{qid}] 逐页筛查 {len(page_jobs)} 页…")
            page_details, page_extractions, n_hit = screen_all_pages(
                llm, processor, page_jobs, PAGE_BATCH_SIZE
            )

            if n_hit == 0:
                print(f"  [{qid}] 警告: 宽松重试后仍 0 相关页，汇总阶段将仅依赖「无」标记页")
            print(f"  [{qid}] 相关页 {n_hit}/{page_count}，汇总作答…")

            reasoning_text, final_out, answer_raw = run_final_answer(
                llm,
                processor,
                question,
                afmt,
                page_extractions,
                lang,
            )

            parse_msgs = build_messages(
                parse_system_msg(),
                parse_user_msg(question, answer_raw),
                image_paths=None,
            )
            parse_prompt = apply_generation_prompt_without_thinking(
                processor, parse_msgs
            )
            parsed_out = llm.generate(
                [{"prompt": parse_prompt}],
                sampling_params=SAMPLING_PARSE,
            )[0].outputs[0].text

            ans = parse_answer_tag(parsed_out)
            evidence = parse_evidence(parsed_out)
            if not evidence:
                evidence = evidence_from_reasoning(reasoning_text)
            if not evidence:
                evidence = [
                    d["page_num"]
                    for d in page_details
                    if d["relevant"] and d["extract"].strip() not in (NO_INFO, "")
                ]

            row_lang = (row.get("language") or "ja").strip()
            if afmt in ("unordered_list", "ordered_list"):
                parsed_list = try_parse_list(ans)
                if parsed_list is not None:
                    ans = dump_list(parsed_list, row_lang)
                else:
                    fix_msgs = build_messages(
                        list_fix_system_msg(),
                        list_fix_user_msg(question, afmt, ans),
                        image_paths=None,
                    )
                    fix_prompt = apply_generation_prompt_without_thinking(
                        processor, fix_msgs
                    )
                    fix_out = llm.generate(
                        [{"prompt": fix_prompt}],
                        sampling_params=SAMPLING_LIST_FIX,
                    )[0].outputs[0].text
                    items = try_parse_list(parse_answer_tag(fix_out))
                    ans = dump_list(
                        items if items is not None else fallback_list(ans, row_lang),
                        row_lang,
                    )
            else:
                ans = normalize_submission_answer(ans, afmt, row_lang)

            record = {
                "id": qid,
                "file_id": fid,
                "question": question,
                "language": row_lang,
                "answer_format": afmt,
                "pdf_path": pdf_path,
                "page_count": page_count,
                "page_details": page_details,
                "page_extractions": [
                    {"page": p, "text": t} for p, t in page_extractions
                ],
                "raw_reasoning": reasoning_text,
                "raw_final": final_out,
                "raw_parse": parsed_out,
                "answer": ans,
                "evidence": evidence,
            }
            results_by_id[qid] = record
            append_checkpoint(ckpt_path, record)
            dt = time.perf_counter() - t_q
            print(
                f"  [{qid}] 完成 {dt:.1f}s | 相关页={n_hit} | "
                f"answer={ans[:80]}{'…' if len(ans) > 80 else ''} | evidence={evidence}"
            )

    del llm
    del processor
    release_torch_memory()

    for p in all_temp_png:
        try:
            os.unlink(p)
        except OSError:
            pass

    ordered_results = [results_by_id[r["id"]] for r in rows if r["id"] in results_by_id]
    submission_rows = [
        {
            "id": rec["id"],
            "answer": rec["answer"],
            "evidence_page_number": format_evidence_column(rec["evidence"]),
        }
        for rec in ordered_results
    ]

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidence_page_number"])
        w.writeheader()
        w.writerows(submission_rows)

    wall_end = time.time()
    elapsed = time.perf_counter() - t0
    run_meta = {
        "started_at": _wall_time_str(wall_start),
        "finished_at": _wall_time_str(wall_end),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_human": _format_duration(elapsed),
        "questions_total": len(rows),
        "questions_completed": len(ordered_results),
    }
    run_meta_path = _resolve(
        os.environ.get("RUN_META_JSON", "pdf_exhaustive_run_meta.json"),
        script_dir,
    )
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"\n已写入: {out_json_path}")
    print(f"已写入: {sub_path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"运行元信息: {run_meta_path}")
    _print_run_banner("end", wall_start, wall_end)


if __name__ == "__main__":
    main()
