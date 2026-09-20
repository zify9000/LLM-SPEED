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
# 凭据文件权限收紧：cp 出来的 .env 继承 umask（常见 0664，同机其他用户可读走
# 全部 API Key）；服务端写回时也只收紧不放宽，这里把历史遗留的宽权限一并纠正
chmod 600 .env 2>/dev/null || true

if [ ! -f config.json ] && [ -f config.json.example ]; then
  cp config.json.example config.json
  echo "[提示] 已从 config.json.example 生成 config.json，请按需修改网关与部署映射"
fi

# 生效监听地址：环境变量优先，其次 .env（与 server.py 的加载顺序一致），
# 最后默认值——旧版无条件打印 127.0.0.1，.env 里写了 HOST=0.0.0.0 时
# 会把"已对本机之外开放"的实情藏起来
env_val() {
  local k="$1" d="$2" v
  v=$(printenv "$k" || true)
  if [ -z "$v" ] && [ -f .env ]; then
    # 取值后剥掉行内注释、首尾引号与空白（与 server.py 的 .env 解析同口径）
    v=$(sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?${k}[[:space:]]*=[[:space:]]*//p" .env \
        | tail -1 | sed 's/[[:space:]]*#.*$//' | tr -d '\r')
    v="${v%\"}"; v="${v#\"}"
    v="${v%\'}"; v="${v#\'}"
    v=$(printf '%s' "$v" | tr -d '[:space:]')
  fi
  printf '%s' "${v:-$d}"
}
HOST_EFF=$(env_val HOST 127.0.0.1)
PORT_EFF=$(env_val PORT 8501)
case "$HOST_EFF" in
  0.0.0.0|::|"")
    echo "[提示] HOST=$HOST_EFF：服务对本机之外开放，而全端点无鉴权。"
    echo "       写接口有 Host/同源校验，需在 .env 用 ALLOWED_HOSTS=<访问用的主机名或IP> 显式放行，"
    echo "       否则请求会被拒绝；同时请自行评估局域网内他人可触发测速与删除历史的风险"
    ;;
esac
echo "[启动] http://${HOST_EFF}:${PORT_EFF}（Ctrl+C 停止）"
exec python server.py
