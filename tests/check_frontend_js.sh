#!/usr/bin/env bash
# 前端内联脚本检查：
#   ① node --check 语法检查（从 static/index.html 提取 <script> 主体）
#   ② 两条注入门禁（纯文本匹配，专防已修过的属性逃逸类 XSS 回归）
#   ③ §pure 纯函数区域存在且非空门禁（tests/run_frontend_pure.sh 依赖它提取用例）
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

# 门禁：§pure 纯函数区域必须存在且非空。tests/run_frontend_pure.sh 靠这对
# 标记提取区域做 DOM-free 单测；若标记被重构删掉（或只留空壳），纯函数测试
# 会失去覆盖却仍可能报绿，这里先钉死防静默失效。
# 注：用 grep -c 而非管道下游的 grep -q——后者提前退出会让 sed 吃 SIGPIPE，
# 在 pipefail 下被误判为失败。
pure_start=$(grep -c '§pure:start' "$tmp" || true)
pure_end=$(grep -c '§pure:end' "$tmp" || true)
if [ "$pure_start" != "1" ] || [ "$pure_end" != "1" ]; then
  echo "ERROR: 内联脚本中期望 §pure:start/end 标记各 1 个，实际 start=$pure_start end=$pure_end" >&2
  exit 1
fi
pure_lines=$(sed -n '/§pure:start/,/§pure:end/p' "$tmp" | grep -cvE '§pure:(start|end)' || true)
if [ "${pure_lines:-0}" -eq 0 ]; then
  echo "ERROR: §pure 区域为空（标记在但内容被删？），拒绝假绿" >&2
  exit 1
fi

echo "frontend JS syntax OK · 注入门禁通过"
