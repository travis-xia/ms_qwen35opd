#!/usr/bin/env bash
# 将 submission.csv 打包为 Codabench 提交 zip
set -euo pipefail
cd "$(dirname "$0")"

CSV="${1:-submission.csv}"
ZIP="${2:-your_submission.zip}"

if [[ ! -f "${CSV}" ]]; then
  echo "错误: 找不到 ${CSV}" >&2
  exit 1
fi

rm -f "${ZIP}"
zip -j "${ZIP}" "${CSV}"

echo "已生成 ${ZIP}（包含 ${CSV}）"
