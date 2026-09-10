#!/usr/bin/env bash
# 前端内联脚本语法检查：从 static/index.html 提取 <script> 主体交 node --check。
# 提取前提：假定 static/index.html 只有一个内联 <script> 块（标签独占一行）。
# 运行：bash tests/check_frontend_js.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp "${TMPDIR:-/tmp}/llmspeed-js.XXXXXX.js")
trap 'rm -f "$tmp"' EXIT
sed -n '/^<script>$/,/^<\/script>$/p' static/index.html | sed '1d;$d' > "$tmp"
if [ ! -s "$tmp" ]; then
  echo "ERROR: 未从 static/index.html 提取到 <script> 主体（脚本标签形式变化？），拒绝假绿" >&2
  exit 1
fi
node --check "$tmp"
echo "frontend JS syntax OK"
