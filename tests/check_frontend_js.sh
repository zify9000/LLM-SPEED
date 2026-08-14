#!/usr/bin/env bash
# 前端内联脚本语法检查：从 static/index.html 提取 <script> 主体交 node --check。
# 运行：bash tests/check_frontend_js.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp --suffix=.js)
trap 'rm -f "$tmp"' EXIT
sed -n '/^<script>$/,/^<\/script>$/p' static/index.html | sed '1d;$d' > "$tmp"
node --check "$tmp"
echo "frontend JS syntax OK"
