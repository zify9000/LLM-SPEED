#!/usr/bin/env bash
# 前端明细表渲染基准 + 等价性回归（评审 backlog 性能项②）：
#   tests/frontend/bench_render.mjs —— 与 tests/check_frontend_js.sh 同口径提取
#   static/index.html 的内联脚本，在 Node + 极小 DOM 桩下整段求值（同时验证页面
#   加载不抛），测 renderResultTable() 的单次与 O(n²) 累计字符串构建耗时，并断言
#   渲染确定、addPoint 的 rAF 合帧不改变最终 innerHTML、终态/阶段切换不被排队帧吞掉。
#
# 离线、零依赖（不装任何 npm 包）、确定性：计时只打印不断言（CI 负载敏感，绝对
# ms 断言会 flake），唯一失败门是等价性断言。运行 < 20s。
# 运行：bash tests/run_frontend_render.sh
set -euo pipefail
cd "$(dirname "$0")/.."
node tests/frontend/bench_render.mjs
