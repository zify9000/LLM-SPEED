#!/usr/bin/env bash
# 前端展示层纯函数（§pure 区域）单测：
#   ① awk 从 static/index.html 的内联脚本中提取 §pure:start/end 之间的区域；
#   ② 与 tests/frontend/pure_cases.mjs 的用例拼成一个临时 .mjs；
#   ③ node --test 执行。
# 区域 DOM-free（不碰 document/window/state/localStorage/rAF），因此可在
# node 里直接跑，零依赖、无构建步骤。标记缺失或提取为空一律报错退出，拒绝假绿。
# 运行：bash tests/run_frontend_pure.sh
set -euo pipefail
cd "$(dirname "$0")/.."

region=$(mktemp "${TMPDIR:-/tmp}/llmspeed-pure-region.XXXXXX.js")
tmp=$(mktemp "${TMPDIR:-/tmp}/llmspeed-pure.XXXXXX.mjs")
trap 'rm -f "$region" "$tmp"' EXIT

# 标记必须各出现一次：防止区域被重构删掉/写重后测试静默失效
start_n=$(grep -c '§pure:start' static/index.html || true)
end_n=$(grep -c '§pure:end' static/index.html || true)
if [ "$start_n" != "1" ] || [ "$end_n" != "1" ]; then
  echo "ERROR: 期望 static/index.html 中 §pure:start/end 标记各 1 个，实际 start=$start_n end=$end_n（区域被删改？）" >&2
  exit 1
fi

awk '/§pure:start/{f=1} f{print} /§pure:end/{f=0}' static/index.html > "$region"
# 除两行标记外必须还有真实内容
if [ ! -s "$region" ] || ! grep -qvE '§pure:(start|end)' "$region"; then
  echo "ERROR: 未从 static/index.html 提取到 §pure 区域内容（标记形式变化？），拒绝假绿" >&2
  exit 1
fi
# 区域必须含已知纯函数：提取范围错位（比如只抓到一行）时及早失败
if ! grep -q 'hasMultiConc' "$region"; then
  echo "ERROR: §pure 区域缺少 hasMultiConc（提取范围错位？），拒绝假绿" >&2
  exit 1
fi

{
  echo 'import { test } from "node:test";'
  echo 'import assert from "node:assert/strict";'
  cat "$region"
  cat tests/frontend/pure_cases.mjs
} > "$tmp"

n_cases=$(grep -cE '^test\(' tests/frontend/pure_cases.mjs || true)
node --test "$tmp"
echo "frontend pure-function tests OK ($n_cases cases)"
