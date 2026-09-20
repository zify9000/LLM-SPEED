#!/usr/bin/env bash
# 前端内联脚本检查：
#   ① node --check 语法检查（从 static/index.html 提取 <script> 主体）
#   ② 两条注入门禁（纯文本匹配，专防已修过的属性逃逸类 XSS 回归）
# 提取前提：假定 static/index.html 只有一个内联 <script> 块（标签独占一行）。
# 运行：bash tests/check_frontend_js.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp "${TMPDIR:-/tmp}/llmspeed-js.XXXXXX.js")
trap 'rm -f "$tmp"' EXIT
script_tag_count=$(grep -c '^<script>$' static/index.html || true)
if [ "$script_tag_count" != "1" ]; then
  echo "ERROR: 期望恰好 1 个内联 <script> 块，实际 $script_tag_count 个（提取逻辑失效？）" >&2
  exit 1
fi
sed -n '/^<script>$/,/^<\/script>$/p' static/index.html | sed '1d;$d' > "$tmp"
if [ ! -s "$tmp" ]; then
  echo "ERROR: 未从 static/index.html 提取到 <script> 主体（脚本标签形式变化？），拒绝假绿" >&2
  exit 1
fi
node --check "$tmp"

# 门禁：单引号属性内不得出现模板插值（`='…${…}'`）。属性以单引号定界时
# JSON.stringify 不转义单引号，模型名/run_id 只要含 `'` 就能闭合属性并注入
# 事件处理器——即 2026-09 审计确认的存储型 XSS（改名端点可写入任意 ≤200 字符
# 模型名，网关返回的 model id 同样不可信）。结构化身份请走
# data-*="${esc(JSON.stringify(...))}" + 事件委托。
#
# 只做这一条而不做"属性内插值一律要 esc()"的通用门禁：现有代码里属性插值多为
# 自产数值（aria-label="${fmtK(v)}" 等），通用规则会大面积误报；外部字符串走
# esc() 的约定由 design/frontend.md 负责，这里只钉死已知可利用的那一类。
if grep -nE "='[^']*\\\$\{" "$tmp" >/dev/null; then
  echo "ERROR: 单引号属性中出现模板插值，存在属性逃逸风险：" >&2
  grep -nE "='[^']*\\\$\{" "$tmp" >&2
  exit 1
fi

echo "frontend JS syntax OK · 注入门禁通过"
