"""LLM-SPEED 服务端：FastAPI 单页测速服务（多 provider）。

启动：  uvicorn server:app --host 127.0.0.1 --port 8501
或：    python server.py      # 缺省绑定 127.0.0.1:8501（HOST/PORT 环境变量可覆盖）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from bench import (AGENT_TURNS_DEFAULT, BenchRun, DEFAULT_CONCURRENCIES,
                   DEFAULT_CTX_LIST, SCENARIOS)

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
                k, v = k.strip(), v.strip()
                if v.startswith('"'):   # 写侧 json.dumps 加引号（值含 # 等），读侧对称解码
                    try:
                        v = json.loads(v)
                    except ValueError:
                        v = v.strip('"')
                else:
                    v = v.strip("'")
                os.environ.setdefault(k, v)
    except OSError:
        pass


_load_dotenv()
GLOBAL_API_KEY = os.environ.get("API_KEY", "")     # 兜底 key；仅存服务端，绝不下发前端

app = FastAPI(title="LLM-SPEED")

RUNS: dict[str, BenchRun] = {}
# _drive 后台任务强引用集：create_task 不保证持有引用，弱引用任务可能被 GC 提前回收
_BG_TASKS: set[asyncio.Task] = set()

# 网关能力记忆（进程级）：探测/400 回退得出的「不支持 thinking 参数」「模型
# 温度受限」按 provider 缓存，下次测速直接跳过——同一能力不再每轮拿一次
# 400 报错去试探（LiteLLM 后台的报错日志即来源于此）
_CAP: dict[str, dict] = {}


def load_config_file() -> tuple[dict, str | None]:
    """返回 (config, error)。解析失败不再静默吞掉：错误上浮到 /api/config，
    前端显式提示，避免用户面对「配置坏了却只看到空 provider 列表」的悬案。"""
    if not os.path.exists(CONFIG_PATH):
        return {}, None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("api_key", None)   # 兼容旧配置：忽略其中的 key
        return d, None
    except Exception as e:  # noqa: BLE001
        return {}, f"config.json 解析失败: {e}"


def _env_key_name(provider_name: str) -> str:
    return "API_KEY_" + re.sub(r"[^A-Z0-9]", "_", provider_name.upper())


def _provider_urls(p: dict) -> list[str]:
    """provider 的候选网关地址列表：gateway_urls 优先，兼容旧字段 gateway_url。"""
    urls = [str(u).strip() for u in (p.get("gateway_urls") or []) if str(u).strip()]
    if not urls and p.get("gateway_url"):
        urls = [str(p["gateway_url"]).strip()]
    return urls


def load_providers() -> list[dict]:
    """多 provider：config.json 的 providers 列表 + .env 的分 provider key。

    key 查找顺序：API_KEY_<NAME> → 全局 API_KEY。无 providers 时返回空列表
    （前端空态引导新增；不再合成「默认网关」——ADR-0001：测速目标必须是用户
    显式登记的 provider，凭空默认一个没有可服务的意图）。
    """
    d, _err = load_config_file()
    providers = []
    for p in d.get("providers") or []:
        urls = _provider_urls(p)
        if not urls:
            continue
        name = p.get("name") or urls[0]
        key = os.environ.get(_env_key_name(name), "") or GLOBAL_API_KEY
        providers.append({
            "name": name,
            "label": p.get("label") or name,
            "gateway_url": urls[0],          # 首选/回退地址（兼容字段）
            "gateway_urls": urls,            # 多地址候选（本地场景内网+隧道，择优绕中继）
            "has_key": bool(key),
            "local": bool(p.get("local")),   # 本地 provider：跑并发测试；云端不跑
            "deployments": p.get("deployments") or [],   # 模型→部署环境（硬件/框架/参数）映射
        })
    return providers


def resolve_provider(name: str | None) -> dict:
    providers = load_providers()
    if not providers:
        raise HTTPException(400, "尚未配置 provider：请在「Provider 管理」中新增")
    if name:
        for p in providers:
            if p["name"] == name:
                return p
        raise HTTPException(400, f"未知 provider: {name}")
    return providers[0]


def _resolve_deployment(provider: dict, model: str) -> dict | None:
    """模型的生效部署：一个模型允许映射多套部署环境（1 对多），但激活
    （active=true）的只有一套——激活者生效；无 active 标记回退首个命中
    （兼容旧配置）。"""
    hits = [d for d in provider.get("deployments") or []
            if model in (d.get("models") or [])]
    if not hits:
        return None
    for d in hits:
        if d.get("active"):
            return d
    return hits[0]


def match_deployments(provider: dict, models: list[str]) -> dict[str, dict]:
    """按被测模型匹配部署环境，返回 {model: {hardware, framework, params, deploy_label}}。

    一个 provider 可挂多套部署（deployments），每套声明其适用的 models；
    同一模型命中多套时激活的那套生效（无 active 标记回退首个命中）；
    只有被选中的模型命中映射时才携带部署信息（本地多部署场景）。
    """
    out: dict[str, dict] = {}
    for m in models:
        dep = _resolve_deployment(provider, m)
        if dep:
            out[m] = {
                "deploy_label": dep.get("label") or "",
                "quant": dep.get("quant") or "",   # 模型量化（如 Q8_0），卡片首格展示
                "hardware": dep.get("hardware") or "",
                "framework": dep.get("framework") or "",
                "params": dep.get("params") or "",
            }
    return out


def provider_key(p: dict) -> str:
    return os.environ.get(_env_key_name(p["name"]), "") or GLOBAL_API_KEY


async def _probe_url(url: str, headers: dict, timeout: float = 3.0) -> float | None:
    """探测候选网关地址的可达性与延迟（GET /models，/v1 两种前缀都试）。
    返回秒级延迟；不可达/服务端错误返回 None。"""
    try:
        async with httpx.AsyncClient(
                base_url=url.rstrip("/"), headers=headers,
                timeout=httpx.Timeout(connect=timeout, read=timeout,
                                      write=timeout, pool=timeout)) as c:
            t = time.perf_counter()
            for path in ("/v1/models", "/models"):
                r = await c.get(path)
                if r.status_code != 404:
                    return (time.perf_counter() - t) if r.status_code < 500 else None
    except Exception:  # noqa: BLE001
        return None
    return None


async def pick_gateway_url(p: dict) -> tuple[str, float | None]:
    """多地址择优：**按配置顺序取第一个可达地址**（用户把内网等优选地址排前，
    可达即绕开隧道等传输中继；不可达自动落到下一个）。并发探测、取可达者中
    顺序最前；全部不可达回退首个（由测速引擎报出真实错误）。
    返回 (选中地址, 延迟秒)。"""
    urls = p.get("gateway_urls") or [p["gateway_url"]]
    if len(urls) == 1:
        return urls[0], None
    key = provider_key(p)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    lats = await asyncio.gather(*(_probe_url(u, headers) for u in urls))
    for u, lat in zip(urls, lats):
        if lat is not None:
            return u, lat
    return urls[0], None


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.get("/api/config")
async def get_config():
    d, config_error = load_config_file()
    providers = load_providers()
    return {
        "providers": [{k: v for k, v in p.items()} for p in providers],
        "api_key_configured": any(p["has_key"] for p in providers),
        "scenarios": [{"key": k, "label": v["label"]} for k, v in SCENARIOS.items()],
        "ctx_list": DEFAULT_CTX_LIST,
        "concurrencies": DEFAULT_CONCURRENCIES,
        "config_error": config_error,   # config.json 坏了不静默：前端显式告警
    }


# ---------------------------------------------------------------------------
# 配置写入（ADR-0001）：页面编辑 provider / 部署环境，服务端校验后原子写回
# config.json；API Key 单向上行写入 .env，永不回显前端
# ---------------------------------------------------------------------------

_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _atomic_write(path: str, text: str, private: bool = False):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        if private:   # 凭据类文件（.env）：沿用原文件权限，新建收紧为 0o600
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode)
                     if os.path.exists(path) else 0o600)
        os.replace(tmp, path)   # 原子替换，写坏一半不会毁掉旧配置
    except BaseException:
        try:
            os.unlink(tmp)      # 失败路径清理 .tmp 残渣
        except OSError:
            pass
        raise


def _validate_providers(providers) -> list[dict]:
    """白名单化前端提交的 providers（config.json 的权威结构），拒绝非法值。"""
    if not isinstance(providers, list) or not providers:
        raise HTTPException(400, "providers 不能为空")
    out, seen = [], set()
    for p in providers:
        if not isinstance(p, dict):
            raise HTTPException(400, "provider 条目必须是对象")
        name = str(p.get("name") or "").strip()
        if not _PROVIDER_NAME_RE.fullmatch(name):
            raise HTTPException(400, f"provider 名称非法: {name!r}"
                                     f"（仅字母/数字/_/-，≤64 字符）")
        if name in seen:
            raise HTTPException(400, f"provider 名称重复: {name}")
        seen.add(name)
        # 网关地址：gateway_urls（多地址，本地场景内网+隧道择优）优先，
        # 兼容旧单地址字段 gateway_url；至少一个，全部必须 http(s)://
        urls_in = p.get("gateway_urls")
        if urls_in is None:
            gw = str(p.get("gateway_url") or "").strip()
            urls = [gw] if gw else []
        else:
            if not isinstance(urls_in, list):
                raise HTTPException(400, f"{name} 的 gateway_urls 必须是数组")
            urls = []
            for u in urls_in:
                u = str(u).strip()
                if u and u not in urls:
                    urls.append(u)
        if not urls:
            raise HTTPException(400, f"{name} 至少配置一个网关地址")
        if len(urls) > 8:
            raise HTTPException(400, f"{name} 网关地址最多 8 个")
        for u in urls:
            if not u.startswith(("http://", "https://")):
                raise HTTPException(400, f"{name} 的网关地址必须以 http(s):// 开头: {u!r}")
        deps = []
        for d in p.get("deployments") or []:
            if not isinstance(d, dict):
                raise HTTPException(400, f"{name} 的部署条目必须是对象")
            max_ctx = d.get("max_ctx")
            dep = {
                "label": str(d.get("label") or "").strip()[:64],
                "models": [str(m).strip() for m in (d.get("models") or []) if str(m).strip()],
                "quant": str(d.get("quant") or "").strip()[:64],
                "hardware": str(d.get("hardware") or "").strip()[:200],
                "framework": str(d.get("framework") or "").strip()[:100],
                "params": str(d.get("params") or "").strip()[:300],
            }
            if isinstance(max_ctx, (int, float)) and not isinstance(max_ctx, bool) \
                    and int(max_ctx) > 0:
                dep["max_ctx"] = int(max_ctx)   # 超窗钳制来源（ADR-0005），非正数视为未配置
            if d.get("active"):
                dep["active"] = True   # 激活标记仅在置位时落盘，旧配置保持零字段
            deps.append(dep)
        # 一个模型允许映射多套部署，但激活只能有一套：重叠激活拒绝写入，
        # 否则生效口径（_resolve_deployment）出现二义
        active_by_model: dict[str, str] = {}
        for dep in deps:
            if not dep.get("active"):
                continue
            for m in dep["models"]:
                if m in active_by_model:
                    raise HTTPException(
                        400, f"{name} 的模型 {m} 同时激活了多套部署环境"
                             f"（{active_by_model[m]}、{dep['label'] or '未命名部署'}），"
                             f"同一模型只能激活一套")
                active_by_model[m] = dep["label"] or "未命名部署"
        out.append({
            "name": name,
            "label": (str(p.get("label") or "").strip() or name)[:64],
            "gateway_url": urls[0],      # 兼容字段=首选地址
            "gateway_urls": urls,
            "local": bool(p.get("local")),
            "deployments": deps,
        })
    return out


@app.put("/api/config")
async def put_config(body: dict):
    """整表替换 config.json 的 providers（部署环境随 provider 内嵌）。"""
    providers = _validate_providers(body.get("providers"))
    d, err = load_config_file()
    if err:
        # 解析失败时 d={}：以空底整表覆盖会丢光其他顶层键，拒绝写入（ADR-0001）
        raise HTTPException(409, f"config.json 已损坏，拒绝覆盖写入（{err}），请先手工修复")
    d["providers"] = providers
    _atomic_write(CONFIG_PATH, json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    return {"ok": True, "providers": len(providers)}


@app.put("/api/config/key")
async def put_config_key(body: dict):
    """写/清除某 provider 的 API Key 到 .env（API_KEY_<名称> 行）。

    凭据边界不变量（ADR-0001）：key 只单向上行（前端输入 → 服务端 → .env），
    任何接口不回显 key 本体；写入后同步 os.environ 即时生效。
    """
    provider = str(body.get("provider") or "").strip()
    if not _PROVIDER_NAME_RE.fullmatch(provider):
        raise HTTPException(400, "provider 名称非法")
    if not any(p["name"] == provider for p in load_providers()):
        raise HTTPException(400, f"provider 不存在: {provider}")
    key = str(body.get("key") or "")
    clear = bool(body.get("clear"))
    env_name = _env_key_name(provider)
    env_path = os.path.join(BASE, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    # 按解析出的 key 名过滤旧行（容忍 `API_KEY_X = "old"` 空格写法）——
    # startswith 漏匹配会让旧行残留，重启后旧值复活（改 key 假生效）
    lines = [l for l in lines
             if (l.split("=", 1)[0].strip() if "=" in l else "") != env_name]
    if not clear and key:
        lines.append(f'{env_name}={json.dumps(key, ensure_ascii=False)}')  # 加引号，值含 # 等也不破坏解析
        os.environ[env_name] = key
    else:
        os.environ[env_name] = ""   # 空值回退全局 API_KEY（load_providers 语义）
    _atomic_write(env_path, "\n".join(lines) + "\n", private=True)
    return {"ok": True}


@app.get("/api/models")
async def list_models(provider: str | None = None):
    """透传网关的模型列表；API Key 由服务端注入，不经前端。"""
    # 旧 gateway= 自定义地址参数已删除（ADR-0001）：它把全局 key 发往任意提交地址
    p = resolve_provider(provider)   # 缺省取首个 provider
    base_url, key = (await pick_gateway_url(p))[0], provider_key(p)
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


def _validate_bench_cfg(cfg: dict):
    """测速矩阵四要素钳制（ADR-0001）：类型/范围/条数不合法一律 400——畸形 ctx
    （如 1e9）会让引擎按档位构造 GB 级 prompt，直接打爆内存。"""
    models, scens = cfg["models"], cfg["scenarios"]
    ctxs, concs = cfg.get("ctx_list") or [], cfg["concurrencies"]
    if not isinstance(models, list) or len(models) > 32 or any(
            not isinstance(m, str) or not m.strip() for m in models):
        raise HTTPException(400, "models 须为 ≤32 个非空字符串")
    if not isinstance(scens, list) or any(
            not isinstance(s, str) or s not in SCENARIOS for s in scens):
        raise HTTPException(400, f"scenarios 须取自白名单: {sorted(SCENARIOS)}")
    # 纯 Agent 场景不以上下文档位为变量（按任务轮次出点），ctx_list 可为空
    if not isinstance(ctxs, list) or len(ctxs) > 16 or any(
            not isinstance(c, (int, float)) or isinstance(c, bool)
            or not 0 <= c <= 4 * 1048576 for c in ctxs):
        raise HTTPException(400, "ctx_list 须为 ≤16 档、逐值 0~4M（对齐前端自定义档位上限）")
    if not isinstance(concs, list) or len(concs) > 8 or any(
            not isinstance(c, int) or isinstance(c, bool)
            or not 1 <= c <= 64 for c in concs):
        raise HTTPException(400, "concurrencies 须为 ≤8 个、逐值 1~64 的整数")
    # Agent 连续任务链参数（可选，缺省引擎用默认值）：轮 1 冷启动独立配置
    # agent_cold_ctx（默认 10K）；两阶段暖轮数 agent_turns_p1（短文本）/
    # agent_turns_p2（长文本 ladder）显式配置；旧配置 agent_turns 单值由
    # 引擎对半切兼容。阶段一每轮新增上下文为正态采样（中心 1K）的钳制区间
    # [min, max]；阶段二 ladder 默认 4K 起翻倍
    vc = cfg.get("agent_cold_ctx")
    if vc is not None and (not isinstance(vc, int) or isinstance(vc, bool)
                           or not 1024 <= vc <= 262144):
        raise HTTPException(400, "agent_cold_ctx 须为 1024~262144 的整数")
    if cfg.get("agent_turns_p1") is not None or cfg.get("agent_turns_p2") is not None:
        for name in ("agent_turns_p1", "agent_turns_p2"):
            v = cfg.get(name)
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 32:
                raise HTTPException(400, f"{name} 须为 0~32 的整数")
        if not 1 <= cfg["agent_turns_p1"] + cfg["agent_turns_p2"] <= 32:
            raise HTTPException(400, "agent_turns_p1 + agent_turns_p2 须在 1~32 之间")
    else:
        v = cfg.get("agent_turns")
        if v is not None and (not isinstance(v, int) or isinstance(v, bool)
                              or not 1 <= v <= 32):
            raise HTTPException(400, "agent_turns 须为 1~32 的整数")
    # 阶段二 ladder 起始增量（可选，默认 4K 逐轮翻倍）
    vb = cfg.get("agent_phase2_base")
    if vb is not None and (not isinstance(vb, int) or isinstance(vb, bool)
                           or not 1024 <= vb <= 65536):
        raise HTTPException(400, "agent_phase2_base 须为 1024~65536 的整数")
    dv = cfg.get("agent_turn_delta")
    if dv is not None:
        vals = dv if isinstance(dv, list) else [dv]
        if (len(vals) not in (1, 2) or any(
                not isinstance(x, int) or isinstance(x, bool)
                or not 256 <= x <= 32768 for x in vals)):
            raise HTTPException(400, "agent_turn_delta 须为 256~32768 的整数或 [min, max] 区间")
        if len(vals) == 2 and vals[0] > vals[1]:
            raise HTTPException(400, "agent_turn_delta 区间须 min ≤ max")


@app.post("/api/bench/start")
async def bench_start(cfg: dict):
    _purge_finished_runs()
    for field in ("models", "scenarios", "concurrencies"):
        if not cfg.get(field):
            raise HTTPException(400, f"缺少配置项: {field}")
    # 纯 Agent 场景按任务轮次出点，不以上下文档位为变量——ctx_list 可缺省
    if any(s != "agent" for s in cfg["scenarios"]) and not cfg.get("ctx_list"):
        raise HTTPException(400, "缺少配置项: ctx_list")
    cfg.setdefault("ctx_list", [])
    _validate_bench_cfg(cfg)
    # 任务备注：可选短文本，随 cfg 存档、展示在历史标题最前；非字符串/空白丢弃
    note = cfg.get("note")
    if isinstance(note, str) and note.strip():
        cfg["note"] = note.strip()[:80]
    else:
        cfg.pop("note", None)
    # provider 解析：服务端注入网关地址与 key，客户端不接触凭据
    if cfg.get("provider"):
        p = resolve_provider(cfg["provider"])
        url, url_lat = await pick_gateway_url(p)   # 多地址择优（内网可达即绕开隧道中继）
        cfg["gateway_url"] = url
        if len(p.get("gateway_urls") or []) > 1:
            cfg["url_choice"] = {"chosen": url, "candidates": len(p["gateway_urls"]),
                                 "latency_ms": (round(url_lat * 1000)
                                                if url_lat is not None else None)}
        cfg["provider_label"] = p["label"]
        # 本地/云端标记随存档与 cfg 事件下发：卡片对本地部署隐去网关地址
        cfg["provider_local"] = bool(p.get("local"))
        cfg["api_key"] = provider_key(p)
        # 部署环境（硬件/框架/参数）跟随被测模型，由服务端按映射注入并随结果存档
        cfg["model_info"] = match_deployments(p, cfg.get("models") or [])
        # 部署的实际上下文上限（deployments[].max_ctx）：超限档位由测速引擎跳过，
        # 避免构造出必然 400 的超窗 prompt（ADR-0005）。同一模型多套部署时
        # 取激活的那套（与 match_deployments 同口径）
        cfg["model_max_ctx"] = {
            m: int(dep["max_ctx"])
            for m in (cfg.get("models") or [])
            if (dep := _resolve_deployment(p, m)) and dep.get("max_ctx")
        }
        if not p["local"]:
            # 云端不做并发/吞吐测试：响应不受控、可能调度到不同推理设备，
            # 服务端强制单发，不信任前端传参（ADR-0003）
            cfg["concurrencies"] = [1]
        # 网关能力记忆下发：已知不支持 thinking / 温度受限的，不再试探（免 400）
        cap = _CAP.get(p["name"])
        if cap:
            cfg["thinking_unsupported"] = cap["thinking_unsupported"]
            cfg["temperature_locked"] = cap["temperature_locked"]
    else:
        # 旧 gateway_url 自定义地址分支已删除（ADR-0001）：它把全局 key 发往任意提交地址
        raise HTTPException(400, "缺少 provider")
    run = BenchRun(cfg, results_dir=RESULTS_DIR)
    RUNS[run.run_id] = run
    t = asyncio.create_task(_drive(run, cfg.get("provider")))
    _BG_TASKS.add(t)                        # 持强引用防 GC 提前回收任务
    t.add_done_callback(_BG_TASKS.discard)
    return {"run_id": run.run_id}


async def _drive(run: BenchRun, provider_name: str | None):
    """驱动一次测速，结束后收割网关能力结论（thinking/温度参数支持度），
    供下次测速免 400 试探。"""
    await run.run()
    if provider_name:
        _CAP[provider_name] = {
            "thinking_unsupported": run.thinking_unsupported,
            "temperature_locked": sorted(run.temperature_locked),
        }


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
        # 订阅扇出（ADR-0009）：每连接独立队列，先注册再回放历史——注册与回放
        # 之间到达的新事件会重复，前端按 seq 去重；旧连接残留不再抢事件。
        # 队列设上限：慢客户端积压撞顶由 bench.emit 摘除该订阅并补终止帧，
        # 前端 EventSource 重连后按 seq 去重回放历史兜底（ADR-0009）
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
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
    """进行中的测速任务（多标签页入口：任意页面可进入正在跑的任务，ADR-0009）。"""
    out = []
    for r in RUNS.values():
        if r.finished_at is not None:
            continue
        cfg = r.cfg
        # agent 场景按任务轮次出点（不以下文档位为变量）：1 轮冷启动 + 两阶段
        # 暖轮数之和，再 × 并发链数
        n_turns = (1 + (cfg["agent_turns_p1"] + cfg["agent_turns_p2"])
                   if cfg.get("agent_turns_p1") is not None
                   and cfg.get("agent_turns_p2") is not None
                   else 1 + int(cfg.get("agent_turns") or AGENT_TURNS_DEFAULT))
        n_scen = sum(n_turns if s == "agent"
                     else len(cfg.get("ctx_list") or [])
                     for s in cfg.get("scenarios") or [])
        total = len(cfg.get("models") or []) * n_scen * len(cfg.get("concurrencies") or [])
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
            cfg = d.get("cfg", {})
            out.append({"run_id": d.get("run_id", f), "started_at": d.get("started_at"),
                        "models": cfg.get("models", []),
                        "scenarios": cfg.get("scenarios", []),
                        "ctx_list": cfg.get("ctx_list", []),
                        "concurrencies": cfg.get("concurrencies", []),
                        "agent_turns": cfg.get("agent_turns"),
                        "agent_turns_p1": cfg.get("agent_turns_p1"),
                        "agent_turns_p2": cfg.get("agent_turns_p2"),
                        "agent_cold_ctx": cfg.get("agent_cold_ctx"),
                        "note": cfg.get("note") or "",
                        "n_points": len(d.get("results", []))})
        except Exception:  # noqa: BLE001
            continue
    return {"history": out}


@app.get("/api/bench/history/{run_id}")
async def bench_history_detail(run_id: str):
    if any(c in run_id for c in "/\\") or ".." in run_id:   # 与 delete 对称的路径穿越防护
        raise HTTPException(400, "非法 run_id")
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/bench/history/{run_id}/note")
async def bench_history_note(run_id: str, body: dict):
    """事后修改历史存档的任务备注（写回 cfg.note，与启动时填的同一字段）。"""
    if any(c in run_id for c in "/\\") or ".." in run_id:
        raise HTTPException(400, "非法 run_id")
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    note = body.get("note")
    if not isinstance(note, str):
        raise HTTPException(400, "note 须为字符串")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    note = note.strip()[:80]
    if note:
        d.setdefault("cfg", {})["note"] = note
    else:
        d.get("cfg", {}).pop("note", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    return {"ok": True, "note": note}


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
    # 缺省只绑本机回环：全端点无鉴权，绑 0.0.0.0 会让局域网内任何人可触发
    # 对云端付费网关的测速（烧额度）并删除历史；确需局域网访问时显式
    # HOST=0.0.0.0 并自行评估风险
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8501)))
