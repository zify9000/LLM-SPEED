@echo off
rem LLM-SPEED 一键启动（Windows）：
rem   自动创建 .venv 虚拟环境 -> 安装依赖 -> 生成 .env（如缺失）-> 启动服务
rem 环境变量：set HOST=0.0.0.0 可开放局域网访问；set PORT=8501
chcp 65001 >nul
cd /d "%~dp0"

rem 探测真实可用的 python：WindowsApps 里的 python.exe 可能是 Microsoft
rem Store 占位别名，where 找得到但一执行就失败，故除 where 外再跑最小脚本
where python >nul 2>nul
if errorlevel 1 goto no_python
python -c "print(1)" >nul 2>nul
if errorlevel 1 goto no_python

if not exist .venv (
  echo [初始化] 创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 goto venv_failed
)
if not exist .venv\Scripts\activate.bat goto venv_failed
call .venv\Scripts\activate.bat
if errorlevel 1 goto venv_failed

python -c "import fastapi, uvicorn, httpx" >nul 2>nul
if not errorlevel 1 goto deps_ok
echo [初始化] 安装依赖（requirements.txt）...
pip install -q -r requirements.txt
if errorlevel 1 goto pip_failed
:deps_ok

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
exit /b 0

:no_python
echo [错误] 未找到可用的 python：请安装 Python 3.10+ 并勾选 Add to PATH
echo        （Microsoft Store 的 python 占位别名不算，需真实安装）
pause
exit /b 1

:venv_failed
echo [错误] 创建或激活虚拟环境 .venv 失败，已中止（避免把依赖装进全局 Python）
pause
exit /b 1

:pip_failed
echo [错误] 依赖安装失败（pip install -r requirements.txt），已中止
pause
exit /b 1
