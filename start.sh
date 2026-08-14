#!/usr/bin/env bash
# LLM-SPEED 一键启动（Linux / macOS）：
#   自动创建 .venv 虚拟环境 → 安装依赖 → 生成 .env（如缺失）→ 启动服务
# 环境变量：HOST（缺省 127.0.0.1，需局域网访问设 0.0.0.0）、PORT（缺省 8501）
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[错误] 未找到 python3（可用 PYTHON=/path/to/python 指定）" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[初始化] 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import fastapi, uvicorn, httpx" >/dev/null 2>&1; then
  echo "[初始化] 安装依赖（requirements.txt）..."
  pip install -q -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[提示] 已从 .env.example 生成 .env——请先填入各 provider 的 API Key"
  echo "       （也可以启动后在页面「Provider 管理」抽屉里配置）"
fi

if [ ! -f config.json ] && [ -f config.json.example ]; then
  cp config.json.example config.json
  echo "[提示] 已从 config.json.example 生成 config.json，请按需修改网关与部署映射"
fi

echo "[启动] http://127.0.0.1:${PORT:-8501}（HOST=${HOST:-127.0.0.1}，Ctrl+C 停止）"
exec python server.py
