@echo off
rem LLM-SPEED 一键启动（Windows）：
rem   自动创建 .venv 虚拟环境 -> 安装依赖 -> 生成 .env（如缺失）-> 启动服务
rem 环境变量：set HOST=0.0.0.0 可开放局域网访问；set PORT=8501
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python，请先安装 Python 3.10+ 并勾选 Add to PATH
  pause
  exit /b 1
)

if not exist .venv (
  echo [初始化] 创建虚拟环境 .venv ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -c "import fastapi, uvicorn, httpx" >nul 2>nul
if errorlevel 1 (
  echo [初始化] 安装依赖（requirements.txt）...
  pip install -q -r requirements.txt
)

if not exist .env (
  copy .env.example .env >nul
  echo [提示] 已从 .env.example 生成 .env——请先填入各 provider 的 API Key
  echo        （也可以启动后在页面「Provider 管理」抽屉里配置）
)

if not exist config.json (
  if exist config.json.example (
    copy config.json.example config.json >nul
    echo [提示] 已从 config.json.example 生成 config.json，请按需修改网关与部署映射
  )
)

if not defined PORT set PORT=8501
echo [启动] http://127.0.0.1:%PORT%（Ctrl+C 停止）
python server.py
pause
