"""LLM-SPEED 服务端：FastAPI 单页测速服务（多 provider）。

启动：  uvicorn server:app --host 0.0.0.0 --port 8501
或：    python server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from bench import BenchRun, DEFAULT_CONCURRENCIES, DEFAULT_CTX_LIST, SCENARIOS

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
CONFIG_PATH = os.path.join(BASE, "config.json")


def _load_dotenv():
    """轻量 .env 加载（不覆盖已有环境变量，避免引入额外依赖）。"""
    path = os.path.join(BASE, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()
GLOBAL_API_KEY = os.environ.get("API_KEY", "")     # 兜底 key；仅存服务端，绝不下发前端
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")

app = FastAPI(title="LLM-SPEED")

RUNS: dict[str, BenchRun] = {}


def load_config_file() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                d = json.load(f)
                d.pop("api_key", None)   # 兼容旧配置：忽略其中的 key
                return d
        except Exception:  # noqa: BLE001
            pass
    return {}


def _env_key_name(provider_name: str) -> str:
    return "API_KEY_" + re.sub(r"[^A-Z0-9]", "_", provider_name.upper())


def load_providers() -> list[dict]:
    """多 provider：config.json 的 providers 列表 + .env 的分 provider key。

    key 查找顺序：API_KEY_<NAME> → 全局 API_KEY。全部缺失时回退单网关模式。
    """
    d = load_config_file()
    providers = []
    for p in d.get("providers") or []:
        if not p.get("gateway_url"):
            continue
        name = p.get("name") or p["gateway_url"]
        key = os.environ.get(_env_key_name(name), "") or GLOBAL_API_KEY
        providers.append({
            "name": name,
            "label": p.get("label") or name,
            "gateway_url": p["gateway_url"],
            "has_key": bool(key),
            "local": bool(p.get("local")),   # 本地 provider：跑并发测试；云端不跑
            "deployments": p.get("deployments") or [],   # 模型→部署环境（硬件/框架/参数）映射
        })
    if not providers:
        url = GATEWAY_URL or d.get("gateway_url", "http://127.0.0.1:4000")
        providers.append({"name": "default", "label": "默认网关",
                          "gateway_url": url, "has_key": bool(GLOBAL_API_KEY),
                          "local": True, "deployments": []})
    return providers


def resolve_provider(name: str | None) -> dict:
    providers = load_providers()
    if name:
        for p in providers:
            if p["name"] == name:
                return p
        raise HTTPException(400, f"未知 provider: {name}")
    return providers[0]


def match_deployments(provider: dict, models: list[str]) -> dict[str, dict]:
    """按被测模型匹配部署环境，返回 {model: {hardware, framework, params, deploy_label}}。

    一个 provider 可挂多套部署（deployments），每套声明其适用的 models；
    只有被选中的模型命中映射时才携带部署信息（本地多部署场景）。
    """
    out: dict[str, dict] = {}
    for m in models:
        for dep in provider.get("deployments") or []:
            if m in (dep.get("models") or []):
                out[m] = {
                    "deploy_label": dep.get("label") or "",
                    "quant": dep.get("quant") or "",   # 模型量化（如 Q8_0），卡片首格展示
                    "hardware": dep.get("hardware") or "",
                    "framework": dep.get("framework") or "",
                    "params": dep.get("params") or "",
                }
                break
    return out


def provider_key(p: dict) -> str:
    return os.environ.get(_env_key_name(p["name"]), "") or GLOBAL_API_KEY


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.get("/api/config")
async def get_config():
    d = load_config_file()
    providers = load_providers()
    return {
        "providers": [{k: v for k, v in p.items()} for p in providers],
        "api_key_configured": any(p["has_key"] for p in providers),
        "scenarios": [{"key": k, "label": v["label"]} for k, v in SCENARIOS.items()],
        "ctx_list": DEFAULT_CTX_LIST,
        "concurrencies": DEFAULT_CONCURRENCIES,
    }


@app.get("/api/models")
async def list_models(provider: str | None = None, gateway: str | None = None):
    """透传网关的模型列表；API Key 由服务端注入，不经前端。"""
    if provider:
        p = resolve_provider(provider)
        base_url, key = p["gateway_url"], provider_key(p)
    else:
        base_url, key = (gateway or GATEWAY_URL), GLOBAL_API_KEY
    if not base_url:
        raise HTTPException(400, "缺少网关地址")
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            base = base_url.rstrip("/")
            r = None
            for path in ("/v1/models", "/models"):   # LiteLLM 用 /v1，DeepSeek 无前缀
                r = await client.get(base + path, headers=headers)
                if r.status_code != 404:
                    break
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"无法连接网关 {base_url}: {e}") from e
    items = data.get("data") or data.get("models") or []
    ids = [m["id"] if isinstance(m, dict) else str(m) for m in items]
    return {"models": ids}


@app.post("/api/bench/start")
async def bench_start(cfg: dict):
    _purge_finished_runs()
    for field in ("models", "scenarios", "ctx_list", "concurrencies"):
        if not cfg.get(field):
            raise HTTPException(400, f"缺少配置项: {field}")
    # provider 解析：服务端注入网关地址与 key，客户端不接触凭据
    if cfg.get("provider"):
        p = resolve_provider(cfg["provider"])
        cfg["gateway_url"] = p["gateway_url"]
        cfg["provider_label"] = p["label"]
        # 本地/云端标记随存档与 cfg 事件下发：卡片对本地部署隐去网关地址
        cfg["provider_local"] = bool(p.get("local"))
        cfg["api_key"] = provider_key(p)
        # 部署环境（硬件/框架/参数）跟随被测模型，由服务端按映射注入并随结果存档
        cfg["model_info"] = match_deployments(p, cfg.get("models") or [])
        # 部署的实际上下文上限（deployments[].max_ctx）：超限档位由测速引擎跳过，
        # 避免构造出必然 400 的超窗 prompt（ADR-0013）
        cfg["model_max_ctx"] = {
            m: int(dep["max_ctx"])
            for dep in p.get("deployments") or [] if dep.get("max_ctx")
            for m in (dep.get("models") or []) if m in (cfg.get("models") or [])
        }
        if not p["local"]:
            # 云端不做并发/吞吐测试：响应不受控、可能调度到不同推理设备，
            # 服务端强制单发，不信任前端传参（ADR-0009）
            cfg["concurrencies"] = [1]
    elif cfg.get("gateway_url"):
        cfg["api_key"] = GLOBAL_API_KEY
        cfg["provider_local"] = True   # 自定义网关视为本地部署，卡片隐去地址
    else:
        raise HTTPException(400, "缺少 provider 或 gateway_url")
    run = BenchRun(cfg, results_dir=RESULTS_DIR)
    RUNS[run.run_id] = run
    asyncio.create_task(run.run())
    return {"run_id": run.run_id}


def _purge_finished_runs(ttl_s: float = 2 * 3600):
    """清理已结束超过 TTL 的运行：RUNS 与事件 history 驻留内存，不清会缓慢泄漏。"""
    now = time.time()
    for rid, r in list(RUNS.items()):
        if r.finished_at and now - r.finished_at > ttl_s:
            RUNS.pop(rid, None)


@app.post("/api/bench/stop/{run_id}")
async def bench_stop(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    run.stop()
    return {"ok": True}


@app.get("/api/bench/events/{run_id}")
async def bench_events(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")

    async def gen():
        # 订阅扇出（ADR-0021）：每连接独立队列，先注册再回放历史——注册与回放
        # 之间到达的新事件会重复，前端按 seq 去重；旧连接残留不再抢事件
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        try:
            for ev in run.history:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if run.finished_at is not None:
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # 心跳：探活死连接、防中间代理缓冲断流
                    continue
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            run.subs.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/bench/active")
async def bench_active():
    """进行中的测速任务（多标签页入口：任意页面可进入正在跑的任务，ADR-0021）。"""
    out = []
    for r in RUNS.values():
        if r.finished_at is not None:
            continue
        cfg = r.cfg
        total = (len(cfg.get("models") or []) * len(cfg.get("scenarios") or [])
                 * len(cfg.get("ctx_list") or []) * len(cfg.get("concurrencies") or []))
        out.append({"run_id": r.run_id, "started_at": r.started_at,
                    "provider": cfg.get("provider_label") or cfg.get("provider") or "",
                    "models": cfg.get("models") or [],
                    "total_points": total, "done_points": len(r.results)})
    return {"runs": out}


@app.get("/api/bench/history")
async def bench_history():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = sorted((f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")),
                   reverse=True)[:50]
    out = []
    for f in files:
        try:
            with open(os.path.join(RESULTS_DIR, f), encoding="utf-8") as fp:
                d = json.load(fp)
            out.append({"run_id": d.get("run_id", f), "started_at": d.get("started_at"),
                        "models": d.get("cfg", {}).get("models", []),
                        "n_points": len(d.get("results", []))})
        except Exception:  # noqa: BLE001
            continue
    return {"history": out}


@app.get("/api/bench/history/{run_id}")
async def bench_history_detail(run_id: str):
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.delete("/api/bench/history/{run_id}")
async def bench_history_delete(run_id: str):
    if any(c in run_id for c in "/\\") or ".." in run_id:
        raise HTTPException(400, "非法 run_id")
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    os.remove(path)
    return {"ok": True}


@app.delete("/api/bench/history")
async def bench_history_clear():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(RESULTS_DIR, f))
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8503)))
