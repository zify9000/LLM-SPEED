#!/usr/bin/env bash
# 前端内联脚本检查：
#   ① node --check 语法检查（从 static/index.html 提取 <script> 主体）
#   ② 两条注入门禁（纯文本匹配，专防已修过的属性逃逸类 XSS 回归）
#   ③ §pure 纯函数区域存在且非空门禁（tests/run_frontend_pure.sh 依赖它提取用例）
#   ④ HTML 注释收尾形式门禁（注释体里写 */ 会让 DOM 结构静默错位）
#   ⑤ CSS 令牌存在性门禁（引用未定义的 var(--x) 会静默失效，底变透明）
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

# 门禁：页面不得加载任何**外部资源**（脚本/样式/字体/图片外链）。
# 图表库已本地 vendored（static/vendor/，见其 README）：既保证离线/内网可用，
# 也消掉"第三方脚本运行在能调用本机全部 API 的页面里"这条供应链面。
# 允许的是相对路径与 data: URI；http(s):// 只允许出现在注释/文档链接里——
# 因此只检查真正会发起请求的标签属性。
if grep -nE '<(script|link|img|iframe|source|video|audio)[^>]*[[:space:]](src|href)="https?://' \
     static/index.html >/dev/null; then
  echo "ERROR: 页面引用了外部资源（应本地 vendored，见 static/vendor/README.md）：" >&2
  grep -nE '<(script|link|img|iframe|source|video|audio)[^>]*[[:space:]](src|href)="https?://' \
    static/index.html >&2
  exit 1
fi

# 门禁：HTML 注释必须"包住的只是注释"。三类同一个后果——浏览器把中间的标记当
# 注释吞掉，DOM 结构整体错位，而**标签配平检查照样会过**（浏览器眼里那就是一个
# 超长注释），肉眼看页面常常"正常"：
#   ① 注释体内出现 CSS 风格收尾 */（HTML 不认它，注释一路吞到下一个 -->）
#      ——自定义档位添加器首版踩过：translateWrap 及后续分区全成了 #ctxList 的子节点
#   ② 注释未收尾（漏了 -->）——合并基础设置卡时踩过：#fProvider 与整张测试矩阵卡
#      的层级全错
#   ③ 注释吞掉了整行标记（行首只有空白紧跟 <tag>）——①② 的直接症状，也是可靠的
#      静态信号。判据只认"行首标记"：本文件的注释正文是中文说明，会从行首写标签；
#      而正文里提到 `<script>` 这类字样是合法的（只作说明），不算吞标记
# 只扫 <script> 之前的标记段：注释全在那儿，脚本里的字符串不参与。
if bad_comment=$(python3 - static/index.html <<'PYEOF'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
inc, opened_at = False, 0
for n, line in enumerate(lines, 1):
    if line.strip() == '<script>':
        break
    i = 0
    while i < len(line):
        if inc:
            if re.match(r'\s*</?[A-Za-z]', line[i:]) and line[i:].strip():
                print('static/index.html:%d: HTML 注释（起于第 %d 行）吞掉了整行标记'
                      '「%s」——收尾符写错或漏了 -->，中间标记会被浏览器当注释吃掉'
                      % (n, opened_at, line.strip()[:40]))
                sys.exit(1)
            e = line.find('-->', i)
            seg = line[i:] if e < 0 else line[i:e]
            if '*/' in seg:
                print('static/index.html:%d: HTML 注释体内出现 */'
                      '（注释会一路吞到下一个 -->，中间的标签被当注释吃掉）' % n)
                sys.exit(1)
            if e < 0:
                break
            i, inc = e + 3, False
        else:
            st = line.find('<!--', i)
            if st < 0:
                break
            i, inc, opened_at = st + 4, True, n
if inc:
    print('static/index.html:%d: HTML 注释未收尾（缺 -->），它会一路吞到下一个 -->，'
          '中间的标签全被当注释吃掉' % opened_at)
    sys.exit(1)
PYEOF
); then
  :
else
  echo "ERROR: $bad_comment" >&2
  exit 1
fi

# 门禁：CSS 自定义属性（设计令牌）引用必须**有定义或有兜底值**。写错令牌名是最阴的
# 一类错：浏览器不报错、控制台无警告，只是那条声明静默失效——历史记录「…」菜单首版
# 把底色写成 var(--surface)（本项目根本没有这个令牌），结果浮层没有卡面，三个菜单项
# 直接压在页面内容上（用户截图才发现）。参照上面「无外部资源」门禁的教训：能静态判的
# 就不要靠人眼。
# 判据：剥掉 CSS/HTML 注释后（注释正文里写 var(--x) 是说明，不是引用，会误报），收集
# 全部 `--name:` 定义，再找 `var(--name)`——**带兜底值**（`var(--x, #fff)`）的放过，
# 那本来就是合法的降级写法。
if bad_token=$(python3 - static/index.html <<'PYEOF'
import re, sys
raw = open(sys.argv[1], encoding='utf-8').read()
body = re.sub(r'/\*.*?\*/', '', re.sub(r'<!--.*?-->', '', raw, flags=re.S), flags=re.S)
defined = set(re.findall(r'(--[A-Za-z0-9-]+)\s*:', body))
for n, line in enumerate(body.split('\n'), 1):
    for m in re.finditer(r'var\((--[A-Za-z0-9-]+)(\s*,)?', line):
        if m.group(1) not in defined and not m.group(2):
            print('static/index.html（剥注释后第 %d 行）引用了未定义的 CSS 令牌 %s'
                  '，且没有兜底值——该声明会静默失效（底色变透明、边框消失这类），'
                  '请改用已定义的令牌或写 var(%s, 兜底值)' % (n, m.group(1), m.group(1)))
            sys.exit(1)
PYEOF
); then
  :
else
  echo "ERROR: $bad_token" >&2
  exit 1
fi

echo "frontend JS syntax OK · 注入门禁通过 · 无外部资源 · HTML 注释收尾形式 OK · CSS 令牌存在性 OK"
