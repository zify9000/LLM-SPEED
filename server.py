"""LLM-SPEED 服务端：FastAPI 单页测速服务（多 provider）。

启动：  uvicorn server:app --host 127.0.0.1 --port 8501
或：    python server.py      # 缺省绑定 127.0.0.1:8501（HOST/PORT 环境变量可覆盖）
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.parse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from bench import (AGENT_CACHE_RANGE, AGENT_INST_RANGE, AGENT_LADDER_MAX,
                   ASR_LADDER_RANGE, BenchRun, CTX_HEADROOM,
                   DEFAULT_CONCURRENCIES, DEFAULT_CTX_LIST,
                   MEDIA_LADDER_MAX, OCR_LADDER_RANGE, SCENARIOS,
                   TTS_LADDER_RANGE)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
CONFIG_PATH = os.path.join(BASE, "config.json")


def _load_dotenv():
    """轻量 .env 加载（不覆盖已有环境变量，避免引入额外依赖）。

    兼容常见手写形态：`export K=V`、带引号的值（写侧 json.dumps 加引号，
    读侧对称 json 解码）、无引号值后的行内注释 `K=sk-x  # kimi`——旧解析会把
    `sk-x" # kimi` 整串当 key 发出去，表现为"刚填的 key 401"，极难自查。"""
    path = os.path.join(BASE, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v.startswith('"'):
                    try:
                        v = json.loads(v)   # 写侧 json.dumps，读侧对称解码
                    except ValueError:
                        v = v.strip('"')
                else:
                    v = v.split("#", 1)[0].strip().strip("'")
                if k:
                    os.environ.setdefault(k, v)
    except OSError:
        pass


_load_dotenv()
GLOBAL_API_KEY = os.environ.get("API_KEY", "")     # 兜底 key；仅存服务端，绝不下发前端

log = logging.getLogger("llm-speed")


# ---------------------------------------------------------------------------
# 本机写接口的同源 / Host 校验与请求体限额（ADR-0081）
#
# 威胁模型：全端点无鉴权（本地单用户工具，SOUL 边界），因此"谁能给本机发请求"
# 就是唯一防线。浏览器里两个已知的绕过口子必须堵：
#   ① DNS rebinding：恶意页面先解析到自己、再重解析到 127.0.0.1，于是对
#      http://evil.tld:8501 的请求变成"同源"，可读写全部接口。防法是校验 Host
#      必须是本机地址（TrustedHostMiddleware），rebinding 的 Host 过不去。
#   ② 无 Origin 校验的写接口：`PUT /api/config` 能整表替换 providers，把已有
#      provider 的 gateway_urls 改成攻击者地址，随后 `GET /api/models` 就把
#      .env 里的 API_KEY_<同名> 发给攻击者。FastAPI 不校验 Content-Type，
#      text/plain 的"简单请求"可绕过 CORS 预检，副作用无需读响应。防法是
#      拒绝跨源写请求（Origin/Sec-Fetch-Site），并对请求体设上限。
# 非浏览器客户端（curl/脚本）不带 Origin/Sec-Fetch-Site，照常可用。
# ---------------------------------------------------------------------------
_TRUSTED_HOSTS = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",")
                  if h.strip()] or ["127.0.0.1", "localhost", "[::1]", "::1",
                                    "testserver"]
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_MAX_BODY_BYTES = 2 * 1024 * 1024   # 写请求体上限：畸形 50MB cfg 曾被原样广播+落盘


def _origin_matches_host(origin: str, host: str) -> bool:
    """Origin 的 netloc 是否等于 Host（同源）。解析失败一律判不匹配。"""
    try:
        netloc = urllib.parse.urlsplit(origin).netloc
    except ValueError:
        return False
    return bool(netloc) and netloc == host


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """启动周期清理任务：RUNS 与事件 history 驻留内存，只在下次 bench_start
    清理会漏掉"起一轮就不再动"的场景（ADR-0009/0081）。"""
    task = asyncio.create_task(_purge_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="LLM-SPEED", lifespan=_lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_TRUSTED_HOSTS)


@app.middleware("http")
async def _guard_write_requests(request: Request, call_next):
    if request.method in _UNSAFE_METHODS:
        origin = request.headers.get("origin")
        if origin and not _origin_matches_host(origin, request.headers.get("host", "")):
            return JSONResponse(
                {"detail": "跨源写请求被拒绝（本机写接口只接受同源请求；"
                           "确需跨源请在服务端显式放行）"}, status_code=403)
        site = request.headers.get("sec-fetch-site")
        if site and site not in ("same-origin", "none"):
            return JSONResponse(
                {"detail": "跨站写请求被拒绝（Sec-Fetch-Site 非 same-origin）"},
                status_code=403)
        raw_len = request.headers.get("content-length")
        if raw_len and raw_len.isdigit() and int(raw_len) > _MAX_BODY_BYTES:
            return JSONResponse(
                {"detail": f"请求体过大（上限 {_MAX_BODY_BYTES // 1024} KB）"},
                status_code=413)
    return await call_next(request)


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
        # 合法 JSON 不代表结构合法：顶层非对象 / providers 非数组会让
        # load_providers 抛未捕获异常，与解析失败同走 config_error 降级
        if not isinstance(d, dict):
            return {}, f"config.json 结构非法: 顶层必须是对象（当前为 {type(d).__name__}）"
        if "providers" in d and not isinstance(d["providers"], list):
            return {}, (f"config.json 结构非法: providers 必须是数组"
                        f"（当前为 {type(d['providers']).__name__}）")
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
            "deployment_groups": p.get("deployment_groups") or [],   # 部署环境分组（仅组织用）
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


# 模型类型白名单（ADR-0020）：部署的 kind/kinds 取值空间
_KINDS = ("llm", "asr", "ocr", "tts")


def _dep_kinds(dep: dict) -> list[str]:
    """部署声明的模型能力集（多选，如文本模型支持视觉输入 = ["llm", "ocr"]）：
    kinds 数组优先，兼容旧单值 kind 字段，缺省 ["llm"]。"""
    kinds = dep.get("kinds")
    if isinstance(kinds, list) and kinds:
        return [str(k) for k in kinds]
    return [str(dep["kind"])] if dep.get("kind") else ["llm"]


def _dep_quants(dep: dict) -> list[str]:
    """部署的量化多选（如 ["Q8_0", "Q4_K_M"]）。
    quants 数组优先，兼容旧单值 quant 字段（按逗号拆分），缺省 []。"""
    quants = dep.get("quants")
    if isinstance(quants, list) and quants:
        return [str(q) for q in quants]
    return [q.strip() for q in str(dep.get("quant") or "").split(",") if q.strip()]


def _dep_kv_quants(dep: dict) -> list[str]:
    """部署的 KV 缓存量化多选（如 ["Q8_0"]），缺省 []。"""
    kv = dep.get("kv_quants")
    if isinstance(kv, list) and kv:
        return [str(q) for q in kv]
    return []


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
    """按被测模型匹配部署环境，返回 {model: {deploy_label, quant, kv_quant,
    max_ctx, hardware, framework, params, kinds}}。

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
                # 本次测速生效的模型量化（quants 首项，兼容旧 quant 单值回退）
                # 与 KV 缓存量化（kv_quants 首项）：一次测速只有一个生效值
                "quant": (_dep_quants(dep) or [""])[0],
                "kv_quant": (_dep_kv_quants(dep) or [""])[0],
                # 部署上下文上限（tokens）：测速卡片 meta「模型」格的上下文段
                # 来源；未配置为 None（前端省略该段，与 model_max_ctx 同门槛）
                "max_ctx": int(dep["max_ctx"]) if dep.get("max_ctx") else None,
                "hardware": dep.get("hardware") or "",
                "framework": dep.get("framework") or "",
                "params": dep.get("params") or "",
                # 部署的模型能力集（llm/asr/ocr/tts 多选，缺省 [llm]）：场景按
                # kind 匹配（ADR-0020），随结果存档、供前端分组与引擎校验
                "kinds": _dep_kinds(dep),
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
    返回 (选中地址, 延迟秒)。

    探测**不带 Authorization**（ADR-0081）：候选里可能混着 http 明文内网地址，
    旧实现把同一把 key 并发发往全部候选——包括最终不会选中的那些。可达性与
    是否带凭据无关（无鉴权探测通常得 401，非 404 即视为可达），所以去掉凭据
    不损失判别力，只把 key 的暴露面收敛到"最终选中的那一个地址"（由引擎下发）。"""
    urls = p.get("gateway_urls") or [p["gateway_url"]]
    if len(urls) == 1:
        return urls[0], None
    lats = await asyncio.gather(*(_probe_url(u, {}) for u in urls))
    for u, lat in zip(urls, lats):
        if lat is not None:
            return u, lat
    return urls[0], None


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


# 静态资源（vendored 的 ECharts / html2canvas，见 static/vendor/README.md）：
# 本地化后页面不再请求任何外部资源——离线/内网可用，且第三方脚本这条供应链
# 面直接消失（此前 CDN 脚本运行在"能调用本机全部 API"的页面里）。
# 只挂 static/ 且仅 GET：写接口守卫（ADR-0081）不受影响。
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")),
          name="static")


@app.get("/api/config")
async def get_config():
    d, config_error = load_config_file()
    providers = load_providers()
    return {
        "providers": [{k: v for k, v in p.items()} for p in providers],
        "api_key_configured": any(p["has_key"] for p in providers),
        "scenarios": [{"key": k, "label": v["label"], "kind": v.get("kind", "llm"),
                       "ladder": v.get("ladder"), "unit": v.get("unit"),
                       # agent 缓存×指令矩阵默认阶梯（前端档位区默认值）
                       "cache_ladder": v.get("cache_ladder"),
                       "inst_ladder": v.get("inst_ladder")}
                      for k, v in SCENARIOS.items()],
        "ctx_list": DEFAULT_CTX_LIST,
        "concurrencies": DEFAULT_CONCURRENCIES,
        "config_error": config_error,   # config.json 坏了不静默：前端显式告警
    }


# ---------------------------------------------------------------------------
# 配置写入（ADR-0001）：页面编辑 provider / 部署环境，服务端校验后原子写回
# config.json；API Key 单向上行写入 .env，永不回显前端
# ---------------------------------------------------------------------------

_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# 模型名（历史存档改名端点用）：禁止控制字符与 HTML/属性注入相关字符。
# 不用"正向白名单字母数字"是因为真实网关模型名含 CJK、点、斜杠、冒号、@ 等，
# 一律拒绝会挡住正常改名；这里只挡真正会造成展示层/请求头问题的字符。
_MODEL_NAME_RE = re.compile(r"^[^\x00-\x1f\x7f\"'<>\\&]{1,200}$")


def _atomic_write(path: str, text: str, private: bool = False):
    """原子写：同目录临时文件 + fsync + os.replace。

    fsync 不是可有可无：只替换不落盘时，掉电后可能留下 0 字节/半截的
    config.json 或 .env（凭据文件被清空 = 全部 provider 失去 key）。
    private=True 用于凭据文件（.env）：无论旧文件权限如何都收紧到 0o600——
    旧实现"沿用原权限"会把 `cp .env.example .env` 带来的 0664 一直传下去，
    同机其他用户可直接读走 API Key。"""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if private:
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)   # 原子替换，写坏一半不会毁掉旧配置
        try:                    # 目录项落盘，rename 本身也需要持久化
            dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)      # 失败路径清理临时文件残渣
        except OSError:
            pass
        raise


# 历史列表摘要缓存：文件名 → (mtime, size, 摘要)。只缓存列表页真正需要的字段，
# 不缓存 results（那是 MB 级）；存档被 note/env/rename/retest 改写时 mtime 变，
# 缓存自然失效。超上限整体清空（列表页最多 50 项，够用且实现最简）。
_HIST_CACHE: dict[str, tuple[float, int, dict]] = {}
_HIST_CACHE_MAX = 200


def _archive_summary(path: str, name: str) -> dict | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    prev = _HIST_CACHE.get(name)
    if prev and prev[0] == st.st_mtime and prev[1] == st.st_size:
        return prev[2]
    try:
        with open(path, encoding="utf-8") as fp:
            d = json.load(fp)
    except (OSError, json.JSONDecodeError, ValueError):
        return None   # 损坏/半截存档不进列表，也不阻塞其余条目
    cfg = d.get("cfg", {})
    summary = {
        "run_id": d.get("run_id", name), "started_at": d.get("started_at"),
        "models": cfg.get("models", []),
        "scenarios": cfg.get("scenarios", []),
        "ctx_list": cfg.get("ctx_list", []),
        "concurrencies": cfg.get("concurrencies", []),
        "agent_cache_ladder": cfg.get("agent_cache_ladder"),
        "agent_inst_ladder": cfg.get("agent_inst_ladder"),
        # 旧存档（连续任务链口径）标题展示回退用
        "agent_turns": cfg.get("agent_turns"),
        "agent_turns_p1": cfg.get("agent_turns_p1"),
        "agent_turns_p2": cfg.get("agent_turns_p2"),
        "agent_cold_ctx": cfg.get("agent_cold_ctx"),
        "note": cfg.get("note") or "",
        "n_points": len(d.get("results", [])),
        # 口径版本（无该字段 = 本机制引入前的遗留口径，见 bench.METRIC_VERSION）
        "metric_version": d.get("metric_version"),
    }
    if len(_HIST_CACHE) >= _HIST_CACHE_MAX:
        _HIST_CACHE.clear()
    _HIST_CACHE[name] = (st.st_mtime, st.st_size, summary)
    return summary


def _dump_archive(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=1) + "\n"


async def _read_json(path: str, what: str = "存档") -> dict:
    """异步读 JSON（线程池执行，ADR-0082）。

    11MB 级存档的 json.load 要占住事件循环数十毫秒，而 TTFT/decode 就是同一
    循环里的 perf_counter 测量——历史面板一开就能把正在跑的测速读数测歪。
    缺失/损坏统一 404（与 _load_archive 同口径）。"""
    def _do():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        return await asyncio.to_thread(_do)
    except FileNotFoundError:
        raise HTTPException(404, "not found") from None
    except json.JSONDecodeError:
        raise HTTPException(404, f"{what}文件已损坏（JSON 无法解析）") from None


async def _write_json(path: str, d: dict) -> None:
    """异步原子写 JSON（线程池执行）：同步 dumps(11MB) ≈ 66ms 同样会卡住测速。"""
    await asyncio.to_thread(_atomic_write, path, _dump_archive(d))


def _provider_deploy_groups(provider: dict) -> list[str]:
    """provider 的部署环境分组名（有序去重，仅组织用；≤40 字符，空值剔除）。"""
    groups_in = provider.get("deployment_groups") or []
    if not isinstance(groups_in, list):
        raise HTTPException(400, f"{provider.get('name') or '?'} 的 deployment_groups "
                                 f"必须是数组（分组名有序去重）")
    groups = []
    for g in groups_in:
        g = str(g).strip()
        if not g:
            continue
        if len(g) > 40:
            raise HTTPException(400, f"{provider.get('name') or '?'} 的分组名 "
                                     f"≤40 字符: {g!r}")
        if g not in groups:
            groups.append(g)
    return groups


def _validate_providers(providers) -> list[dict]:
    """白名单化前端提交的 providers（config.json 的权威结构），拒绝非法值。

    允许空数组：文档化的空态（gateway-integration.md / web-config.md）就是
    "providers 为空 + 前端空态引导新增"。旧实现拒绝空数组，导致用户删掉最后
    一个 provider 后保存 400、除手改 config.json 无路可走。"""
    if not isinstance(providers, list):
        raise HTTPException(400, "providers 必须是数组")
    if not providers:
        return []
    out, seen = [], set()
    env_owners: dict[str, str] = {}
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
        # env 键名冲突检查：非 A-Z0-9 字符统一映射为 _（编码规则不动，兼容既有
        # .env 条目），但 gpt-4o 与 gpt_4o 会落到同一个 API_KEY_GPT_4O——两个
        # provider 静默共用/互清同一 Key，可能把 key 发给错误网关，拒绝写入
        env_name = _env_key_name(name)
        owner = env_owners.get(env_name)
        if owner:
            raise HTTPException(
                400, f"provider「{owner}」与「{name}」的 API Key 存储名冲突"
                     f"（同映射到 {env_name}），会共用同一把 Key——请重命名其中一个")
        env_owners[env_name] = name
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
                "hardware": str(d.get("hardware") or "").strip()[:200],
                "framework": str(d.get("framework") or "").strip()[:100],
                "params": str(d.get("params") or "").strip()[:300],
            }
            # 环境分组：仅组织/排序用（纯前端归类），不影响解析与 max_ctx 钳制；
            # 空值 = 未分组，不落盘保持零字段
            g = str(d.get("group") or "").strip()[:40]
            if g:
                dep["group"] = g
            # 量化多选（如 ["Q8_0", "Q4_K_M"]）：quants 数组优先，兼容旧单值
            # quant 字段（逗号分隔拆分，与前端解析口径一致）；非空才落盘，
            # 不再写 quant（与 kinds/kind 先例同口径，旧配置读取见 _dep_quants）
            quants_in = d.get("quants")
            if quants_in is None:
                quants_in = str(d.get("quant") or "").split(",")
            if not isinstance(quants_in, list):
                raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                         f"quants 须为数组（量化多选）")
            quants = []
            for q in quants_in:
                q = str(q).strip()
                if not q:
                    continue
                if len(q) > 64:
                    raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                             f"quants 单项 ≤64 字符: {q!r}")
                if q not in quants:
                    quants.append(q)
            if len(quants) > 8:
                raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                         f"quants 最多 8 项")
            if quants:
                dep["quants"] = quants
            # KV 缓存量化多选（如 ["Q8_0"]）：校验口径与 quants 完全一致；
            # 非空才落盘（一次测速生效值取首项，见 match_deployments）
            kv_in = d.get("kv_quants") or []
            if not isinstance(kv_in, list):
                raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                         f"kv_quants 须为数组（KV 量化多选）")
            kv_quants = []
            for q in kv_in:
                q = str(q).strip()
                if not q:
                    continue
                if len(q) > 64:
                    raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                             f"kv_quants 单项 ≤64 字符: {q!r}")
                if q not in kv_quants:
                    kv_quants.append(q)
            if len(kv_quants) > 8:
                raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                         f"kv_quants 最多 8 项")
            if kv_quants:
                dep["kv_quants"] = kv_quants
            # 非有限值（NaN/Infinity，json.loads 会接受）同非正数口径：视为未配置
            if isinstance(max_ctx, (int, float)) and not isinstance(max_ctx, bool) \
                    and math.isfinite(max_ctx) and int(max_ctx) > 0:
                dep["max_ctx"] = int(max_ctx)   # 超窗钳制来源（ADR-0005）
            if d.get("active"):
                dep["active"] = True   # 激活标记仅在置位时落盘，旧配置保持零字段
            # 模型能力集（多选：文本模型支持视觉输入时同时挂 ocr）：kinds 数组
            # 优先，兼容旧单值 kind 字段；缺省 [llm] 不落盘，旧配置保持零字段
            kinds_in = d.get("kinds")
            if kinds_in is None:
                kinds_in = [d["kind"]] if d.get("kind") else ["llm"]
            if not isinstance(kinds_in, list) or not kinds_in:
                raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                         f"kinds 须为非空数组（{'/'.join(_KINDS)} 多选）")
            kinds = []
            for k in kinds_in:
                k = str(k).strip().lower()
                if k not in _KINDS:
                    raise HTTPException(400, f"{name} 的部署 {dep['label'] or '未命名部署'} "
                                             f"kinds 元素须取自 {'/'.join(_KINDS)}: {k!r}")
                if k not in kinds:
                    kinds.append(k)
            if kinds != ["llm"]:
                dep["kinds"] = kinds
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
            "deployment_groups": _provider_deploy_groups(p),
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
    await asyncio.to_thread(_atomic_write, CONFIG_PATH,
                            json.dumps(d, ensure_ascii=False, indent=2) + "\n")
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
    if not clear and key:
        # 粘贴常带尾随换行/制表符：进了 os.environ 就会拼进 Authorization 头，
        # httpx 直接抛 LocalProtocolError("Illegal header value")，之后每个请求
        # 都失败且报错与 key 无关，极难定位；长度上限防畸形超长值撑爆 .env 与头
        if len(key) > 512:
            raise HTTPException(400, "API Key 过长（≤512 字符）")
        if any(c in key for c in "\r\n\x00"):
            raise HTTPException(400, "API Key 不得包含换行或空字符")
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
    await asyncio.to_thread(_atomic_write, env_path, "\n".join(lines) + "\n", True)
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
    # 网关返回非预期结构不 500：顶层须为对象、data/models 须为数组，按本端点
    # 既有口径转 502 并说明；单条缺 id 跳过（不因个别畸形条目丢整个列表）
    if not isinstance(data, dict):
        raise HTTPException(502, f"网关 {base_url} 返回了非预期的数据结构"
                                 f"（顶层不是对象），无法解析模型列表")
    items = data.get("data") or data.get("models") or []
    if not isinstance(items, list):
        raise HTTPException(502, f"网关 {base_url} 返回了非预期的数据结构"
                                 f"（data/models 不是数组），无法解析模型列表")
    ids = []
    for m in items:
        if isinstance(m, dict):
            if m.get("id"):
                ids.append(str(m["id"]))
        else:
            ids.append(str(m))
    return {"models": ids}


# 测速请求允许客户端提交的配置键（ADR-0081）。其余键（gateway_url/api_key/
# model_info/model_max_ctx/...）一律由服务端在下面自行注入，不接受客户端指定。
_BENCH_CFG_KEYS = frozenset({
    "provider", "models", "scenarios", "ctx_list", "concurrencies",
    "translate_ladder", "agent_cache_ladder", "agent_inst_ladder",
    "asr_ladder", "ocr_ladder", "tts_ladder",
    "max_tokens", "repeats", "timeout_s", "thinking", "reply_mode",
    "cache", "temperature", "cpt", "note",
})
_MODEL_NAME_MAX = 256


def _clean_bench_cfg(cfg: dict) -> dict:
    """按白名单重建测速配置（未知键丢弃），并给自由文本设长度上限。

    不做类型/范围校验（那是 _validate_bench_cfg 的职责），只管"能进来的键"
    与"字符串能有多长"——模型名会进请求体、SSE 帧、存档与前端 DOM，无上限时
    一个 10MB 的模型名就能把整条链路撑爆。"""
    if not isinstance(cfg, dict):
        raise HTTPException(400, "测速配置必须是对象")
    clean = {k: cfg[k] for k in _BENCH_CFG_KEYS if k in cfg}
    models = clean.get("models")
    if isinstance(models, list) and any(
            isinstance(m, str) and len(m) > _MODEL_NAME_MAX for m in models):
        raise HTTPException(400, f"模型名过长（单个 ≤{_MODEL_NAME_MAX} 字符）")
    if "provider" in clean:
        if not isinstance(clean["provider"], str):
            raise HTTPException(400, "provider 须为字符串")
        clean["provider"] = clean["provider"].strip()[:64]
    return clean


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
    # 纯 Agent 场景不以上下文档位为变量（按缓存×指令矩阵出点），ctx_list 可为空
    if not isinstance(ctxs, list) or len(ctxs) > 16 or any(
            not isinstance(c, (int, float)) or isinstance(c, bool)
            or not 0 <= c <= 4 * 1048576 for c in ctxs):
        raise HTTPException(400, "ctx_list 须为 ≤16 档、逐值 0~4M（对齐前端自定义档位上限）")
    # 翻译场景原文字长阶梯（可选，缺省用场景默认值）：逐值为正整数（字原文）
    tl = cfg.get("translate_ladder")
    if tl is not None and (not isinstance(tl, list) or len(tl) > 16 or any(
            not isinstance(v, int) or isinstance(v, bool)
            or not 1 <= v <= 65536 for v in tl)):
        raise HTTPException(400, "translate_ladder 须为 ≤16 档、逐值 1~65536 字原文")
    if not isinstance(concs, list) or len(concs) > 8 or any(
            not isinstance(c, int) or isinstance(c, bool)
            or not 1 <= c <= 64 for c in concs):
        raise HTTPException(400, "concurrencies 须为 ≤8 个、逐值 1~64 的整数")
    # 回复模式（ADR-0021）：free 自由回答（默认）/ echo 模板续写（创意/代码
    # 场景换用改写指令 + assistant 预填，decode 对齐高复用真实场景）
    rm = cfg.get("reply_mode")
    if rm is not None and rm not in ("free", "echo"):
        raise HTTPException(400, "reply_mode 须为 free 或 echo")
    # Agent 缓存×指令矩阵阶梯（可选，缺省用场景默认值）：缓存档 = 已写入前缀
    # 缓存的上下文 tokens（0 起，与 ctx_list 同界）；指令档 = 缓存基础上的
    # 单步新输入 tokens（16~65536）；各 ≤16 档
    cl = cfg.get("agent_cache_ladder")
    if cl is not None and (not isinstance(cl, list) or len(cl) > AGENT_LADDER_MAX
            or any(not isinstance(v, int) or isinstance(v, bool)
                   or not AGENT_CACHE_RANGE[0] <= v <= AGENT_CACHE_RANGE[1]
                   for v in cl)):
        raise HTTPException(400, f"agent_cache_ladder 须为 ≤{AGENT_LADDER_MAX} 档、"
                                 f"逐值 {AGENT_CACHE_RANGE[0]}~{AGENT_CACHE_RANGE[1]} tokens")
    il = cfg.get("agent_inst_ladder")
    if il is not None and (not isinstance(il, list) or len(il) > AGENT_LADDER_MAX
            or any(not isinstance(v, int) or isinstance(v, bool)
                   or not AGENT_INST_RANGE[0] <= v <= AGENT_INST_RANGE[1]
                   for v in il)):
        raise HTTPException(400, f"agent_inst_ladder 须为 ≤{AGENT_LADDER_MAX} 档、"
                                 f"逐值 {AGENT_INST_RANGE[0]}~{AGENT_INST_RANGE[1]} tokens")
    # 媒体场景阶梯（可选，缺省用场景默认值）：asr=时长（秒）/ ocr=张数 /
    # tts=字符（字数）；各 ≤MEDIA_LADDER_MAX 档，范围随单位而定
    for field, rng, unit in (("asr_ladder", ASR_LADDER_RANGE, "秒"),
                             ("ocr_ladder", OCR_LADDER_RANGE, "张"),
                             ("tts_ladder", TTS_LADDER_RANGE, "字")):
        ml = cfg.get(field)
        if ml is not None and (not isinstance(ml, list) or len(ml) > MEDIA_LADDER_MAX
                or any(not isinstance(v, int) or isinstance(v, bool)
                       or not rng[0] <= v <= rng[1] for v in ml)):
            raise HTTPException(400, f"{field} 须为 ≤{MEDIA_LADDER_MAX} 档、"
                                     f"逐值 {rng[0]}~{rng[1]} {unit}")
    # 复测/超时/输出上限/思考模式：bench_start 直接透传引擎，这里一并白名单化
    # （repeats 上限对齐引擎钳制 1~5；thinking 枚举 = 引擎三值 auto/enabled/disabled）
    rv = cfg.get("repeats")
    if rv is not None and (not isinstance(rv, int) or isinstance(rv, bool)
                           or not 1 <= rv <= 5):
        raise HTTPException(400, "repeats 须为 1~5 的整数")
    tv = cfg.get("timeout_s")
    if tv is not None and (not isinstance(tv, (int, float)) or isinstance(tv, bool)
                           or not 0 < tv <= 3600):
        raise HTTPException(400, "timeout_s 须为 1~3600 的数值（秒）")
    mt = cfg.get("max_tokens")
    if mt is not None and (not isinstance(mt, int) or isinstance(mt, bool)
                           or not 1 <= mt <= 32768):
        raise HTTPException(400, "max_tokens 须为 1~32768 的整数")
    th = cfg.get("thinking")
    if th is not None and th not in ("auto", "enabled", "disabled"):
        raise HTTPException(400, "thinking 须为 auto/enabled/disabled")
    # 白名单收进来的自由参数也一并钳制：畸形值会直接进请求体发给网关，
    # 或在引擎里参与构造（ADR-0081 同批）
    cm = cfg.get("cache")
    if cm is not None and cm not in ("bust", "stable"):
        raise HTTPException(400, "cache 须为 bust 或 stable")
    tv2 = cfg.get("temperature")
    if tv2 is not None and (not isinstance(tv2, (int, float)) or isinstance(tv2, bool)
                            or not 0 <= tv2 <= 2):
        raise HTTPException(400, "temperature 须为 0~2 的数值")
    cp = cfg.get("cpt")
    if cp is not None and (not isinstance(cp, (int, float)) or isinstance(cp, bool)
                           or not 0.3 <= cp <= 20):
        raise HTTPException(400, "cpt 须为 0.3~20 的数值（chars/token）")


def _agent_matrix_max_total(cfg: dict) -> int:
    """agent 矩阵最大组合目标的保守上界（tokens）：最大缓存档 + 最大指令档
    （部署预算告警用；引擎对超预算组合整档跳过）。"""
    sc = SCENARIOS["agent"]
    cache = [int(v) for v in (cfg.get("agent_cache_ladder") or sc["cache_ladder"])]
    inst = [int(v) for v in (cfg.get("agent_inst_ladder") or sc["inst_ladder"])]
    return (max(cache) if cache else 0) + (max(inst) if inst else 0)


@app.post("/api/bench/start")
async def bench_start(cfg: dict):
    _purge_finished_runs()
    # 白名单重建：请求体整份进 RUNS[].cfg，随后被 SSE cfg 帧广播（bench.py
    # emit）并写入存档。旧实现只校验已知字段、未知键原样保留，于是
    # `{"junk": "A"*50MB}` 会被常驻内存 + 广播 + 落盘（ADR-0081）。
    cfg = _clean_bench_cfg(cfg)
    for field in ("models", "scenarios", "concurrencies"):
        if not cfg.get(field):
            raise HTTPException(400, f"缺少配置项: {field}")
    # 纯文本场景按上下文档位出点；agent 场景按缓存×指令矩阵；媒体场景（asr/ocr/tts）
    # 与翻译场景（translate）按阶梯出点（时长/张数/字符/原文字数），默认阶梯可被
    # cfg 的 <场景>_ladder 覆盖 ——后三类 ctx_list 均可缺省
    if any(SCENARIOS[s].get("kind", "llm") == "llm" and s != "agent"
           and "ladder" not in SCENARIOS[s]
           for s in cfg["scenarios"]) and not cfg.get("ctx_list"):
        raise HTTPException(400, "缺少配置项: ctx_list")
    cfg.setdefault("ctx_list", [])
    _validate_bench_cfg(cfg)
    warnings: list[str] = []   # 启动前的显式告警：随 status 事件下发（见 run 装配处）
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
        # 模型类型能力集映射（ADR-0020 多选扩展）：按部署的 kinds 下发（未映射
        # 部署的模型按 [llm]），引擎据此把模型分发到对应 kind 的场景
        cfg["model_kinds"] = {
            m: list((cfg["model_info"].get(m) or {}).get("kinds") or ["llm"])
            for m in (cfg.get("models") or [])
        }
        # 一次运行一种 kind 的边界不因多能力模型打破：所选场景须同属一种类型
        scen_kinds = {SCENARIOS[s].get("kind", "llm") for s in cfg["scenarios"]}
        if len(scen_kinds) > 1:
            raise HTTPException(400, "一次运行只允许一种类型的场景"
                                     "（文本生成/语音转写/图像识别/语音合成不可混选）")
        bad = sorted({(m, s) for s in cfg["scenarios"] for m in cfg["models"]
                      if SCENARIOS[s].get("kind", "llm")
                      not in cfg["model_kinds"].get(m, ["llm"])})
        if bad:
            hint = "; ".join(
                f"{m}（{'/'.join(cfg['model_kinds'].get(m, ['llm']))}）× 场景「"
                f"{SCENARIOS[s]['label']}」（{SCENARIOS[s].get('kind', 'llm')}）"
                for m, s in bad[:3])
            raise HTTPException(400, f"模型类型与场景不匹配：{hint}"
                                     f"{' 等' if len(bad) > 3 else ''}"
                                     "——模型选择器只列出同类型场景")
        # 部署的实际上下文上限（deployments[].max_ctx）：超限档位由测速引擎跳过，
        # 避免构造出必然 400 的超窗 prompt（ADR-0005）。同一模型多套部署时
        # 取激活的那套（与 match_deployments 同口径）
        cfg["model_max_ctx"] = {
            m: int(dep["max_ctx"])
            for m in (cfg.get("models") or [])
            if (dep := _resolve_deployment(p, m)) and dep.get("max_ctx")
        }
        # 未命中显式告警（不静默）：provider 配了 deployments 但被测模型一个都
        # 没命中时，model_info/model_max_ctx 为空、引擎失去 max_ctx 钳制保护，
        # 超窗 prompt 会直接打爆模型上限（事故：部署映射模型名与网关实际 id
        # 不一致，静默丢保护）——这里显式提示用户去补映射
        if p.get("deployments"):
            for m in (cfg.get("models") or []):
                if not _resolve_deployment(p, m):
                    warnings.append(
                        f"⚠ {m}：模型未映射部署环境，max_ctx 钳制不可用"
                        "（请在「部署环境」中核对模型名后补映射）")
        # agent 矩阵最大组合校验（不阻断）：最大缓存档 + 最大指令档超出部署
        # 预算（max_ctx − max_tokens − 余量）时引擎会整档跳过（point_skipped），
        # 预先告知用户会有组合被跳过
        if "agent" in (cfg.get("scenarios") or []):
            max_total = _agent_matrix_max_total(cfg)
            max_tokens = int(cfg.get("max_tokens")
                             or SCENARIOS["agent"]["max_tokens_default"])
            for m in (cfg.get("models") or []):
                budget = (cfg["model_max_ctx"].get(m) or 0) \
                    - max_tokens - CTX_HEADROOM
                if budget > 0 and max_total > budget:
                    warnings.append(
                        f"⚠ {m}：agent 矩阵最大组合（缓存+指令）~{max_total} tokens 超出"
                        f"部署预算 {budget}（max_ctx − max_tokens − 余量），"
                        "超出上限的组合将被跳过")
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
    # 告警先入 run.history 再起测：订阅连接（含中途进入的标签页）回放即得，
    # 复用现有 status 事件协议，不新造事件类型
    for msg in warnings:
        await run.emit({"type": "status", "msg": msg})
    RUNS[run.run_id] = run
    t = asyncio.create_task(_drive(run, cfg.get("provider")))
    _BG_TASKS.add(t)                        # 持强引用防 GC 提前回收任务
    t.add_done_callback(_on_drive_done)
    return {"run_id": run.run_id}


def _on_drive_done(task: asyncio.Task):
    """后台任务收口：必须取回异常。旧实现只 discard 引用，_save 抛错
    （磁盘满/权限）时异常被静默吞掉——客户端只看到 SSE 流无声结束，
    服务端也没有任何日志，事后完全无从查起（ADR-0081）。"""
    _BG_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.exception("测速后台任务异常终止", exc_info=exc)


async def _drive(run: BenchRun, provider_name: str | None):
    """驱动一次测速，结束后收割网关能力结论（thinking/温度参数支持度），
    供下次测速免 400 试探。收尾阶段（存档/日志）自身的异常不得再抛出。"""
    try:
        await run.run()
    except Exception:   # noqa: BLE001  引擎内部已收口为 error 事件，这里只兜底存档/收尾
        log.exception("测速运行 %s 未捕获异常", run.run_id)
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


_PURGE_INTERVAL_S = 600


async def _purge_loop():
    """周期清理（lifespan 启动）：不依赖"下一次 bench_start"触发。"""
    while True:
        await asyncio.sleep(_PURGE_INTERVAL_S)
        try:
            _purge_finished_runs()
        except Exception:   # noqa: BLE001  清理失败不得拖垮服务
            log.exception("清理已结束运行失败")



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
        # agent 场景按缓存×指令矩阵出点（不以下文档位为变量）：组合数 =
        # 缓存档数 × 指令档数，再 × 并行会话数
        n_agent = (len(cfg.get("agent_cache_ladder") or SCENARIOS["agent"]["cache_ladder"])
                   * len(cfg.get("agent_inst_ladder") or SCENARIOS["agent"]["inst_ladder"]))
        n_scen = sum(n_agent if s == "agent"
                     else len(cfg.get("translate_ladder") or SCENARIOS[s]["ladder"])
                     if s == "translate"   # 翻译场景：矩阵自定义阶梯优先于场景默认
                     else len(cfg.get(f"{s}_ladder") or SCENARIOS[s]["ladder"])
                     if "ladder" in SCENARIOS[s]   # 媒体场景：cfg 覆盖优先于场景默认
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
    """历史列表：只回摘要，且带 (mtime,size) 缓存 + 线程池执行。

    旧实现每次请求都同步 json.load 最多 50 份完整存档（实测单份 11MB、
    全量 ~70ms 占住事件循环），把"打开历史面板"变成对正在跑的测速的计时干扰
    （ADR-0082）。摘要缓存让同一份未变动的存档只解析一次。"""
    def _collect():
        os.makedirs(RESULTS_DIR, exist_ok=True)
        files = sorted((f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")),
                       reverse=True)[:50]
        return [_archive_summary(os.path.join(RESULTS_DIR, f), f) for f in files]

    out = await asyncio.to_thread(_collect)
    return {"history": [s for s in out if s]}


@app.get("/api/bench/history/{run_id}")
async def bench_history_detail(run_id: str):
    _valid_run_id(run_id)
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    # 与「不存在」区分：损坏时给可行动提示（_read_json 统一措辞）
    return await _read_json(path, "存档")


@app.post("/api/bench/history/{run_id}/note")
async def bench_history_note(run_id: str, body: dict):
    """事后修改历史存档的任务备注（写回 cfg.note，与启动时填的同一字段）。"""
    _valid_run_id(run_id)
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    note = body.get("note")
    if not isinstance(note, str):
        raise HTTPException(400, "note 须为字符串")
    d = await _read_json(path, "存档")
    note = note.strip()[:80]
    if note:
        d.setdefault("cfg", {})["note"] = note
    else:
        d.get("cfg", {}).pop("note", None)
    # 原子替换写回：直接覆写会因进程被杀留下写了一半的存档，永久损坏既有测速记录
    await _write_json(path, d)
    return {"ok": True, "note": note}


@app.post("/api/bench/history/{run_id}/rename")
async def bench_history_rename(run_id: str, body: dict):
    """存档模型改名适配：本地部署模型改名后，把历史存档里的旧模型名改写为新名。

    变更所有引用位——cfg.models、model_info/model_kind(s)/model_max_ctx 的键、
    temperature_locked 名单、逐测速点 model 字段；并在存档顶层追加
    model_renames 审计留痕（from/to/at），原子写回（与 note 端点同口径）。
    """
    _valid_run_id(run_id)
    old = str(body.get("old") or "").strip()
    new = str(body.get("new") or "").strip()
    if not old or not new:
        raise HTTPException(400, "old/new 均须为非空字符串")
    if len(old) > 200 or len(new) > 200:
        raise HTTPException(400, "模型名过长（≤200 字符）")
    # 模型名会进前端 DOM（含属性/内联事件）与网关请求体：限制为可打印常见字符，
    # 既避免属性逃逸类注入，也避免控制字符破坏存档与展示
    if not _MODEL_NAME_RE.fullmatch(new):
        raise HTTPException(
            400, "新模型名含非法字符（仅字母/数字/._:@+/-/空格/中文等可打印字符，"
                 "不得含引号、尖括号、反斜杠与控制字符）")
    if old == new:
        raise HTTPException(400, "新旧模型名相同，无需改名")
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    d = await _read_json(path, "存档")
    cfg = d.get("cfg")
    if not isinstance(cfg, dict):
        raise HTTPException(404, "存档缺少 cfg 段（结构不完整），无法改名")
    models = cfg.get("models") or []
    if old not in models:
        raise HTTPException(400, f"存档中不存在模型 {old!r}（本次测速模型：{models}）")
    if new in models:
        raise HTTPException(400, f"存档中已存在模型 {new!r}，改名会让两个模型同名混淆")
    # 目标名已作为快照键存在时一并拒绝：静默覆盖会让新名的部署环境快照被旧名的
    # 顶掉，卡片上的硬件/量化信息与实际测速条件不再对应
    for key in ("model_info", "model_kind", "model_kinds", "model_max_ctx"):
        mp = cfg.get(key)
        if isinstance(mp, dict) and new in mp:
            raise HTTPException(
                400, f"存档的 {key} 中已存在模型 {new!r}，改名会覆盖其快照，"
                     f"请先清理或改用其他名字")
    cfg["models"] = [new if m == old else m for m in models]
    for key in ("model_info", "model_kind", "model_kinds", "model_max_ctx"):
        mp = cfg.get(key)
        if isinstance(mp, dict) and old in mp:
            mp[new] = mp.pop(old)
    tl = cfg.get("temperature_locked")
    if isinstance(tl, list):   # 温度受限名单按模型名记录（bench 侧 set[str]）
        cfg["temperature_locked"] = [new if m == old else m for m in tl]
    n = 0
    for p in d.get("results") or []:
        if isinstance(p, dict) and p.get("model") == old:
            p["model"] = new
            n += 1
    # 审计数组同步改名：retests/deletions/splices 里的 model 是身份五元组的一部分，
    # 不改会让"改名前留的审计"指向一个已不存在的模型名，后续删除点/复测匹配不上
    renamed_audit = 0
    for key in ("retests", "deletions", "splices"):
        for item in d.get(key) or []:
            if not isinstance(item, dict):
                continue
            if item.get("model") == old:
                item["model"] = new
                renamed_audit += 1
            for rep in item.get("replaced") or []:
                if isinstance(rep, dict) and rep.get("model") == old:
                    rep["model"] = new
                    renamed_audit += 1
    d.setdefault("model_renames", []).append(
        {"from": old, "to": new, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "points": n, "audit_entries": renamed_audit})
    await _write_json(path, d)
    return {"ok": True, "old": old, "new": new, "points": n}


@app.post("/api/bench/history/{run_id}/env")
async def bench_history_env(run_id: str, body: dict):
    """事后修改历史存档的部署环境快照（写回 cfg.model_info[model] 的字段）。

    在原条目基础上更新 deploy_label/quant/kv_quant/max_ctx/hardware/framework/
    params（量化存单值 = 该次测速实际生效值；max_ctx 为上下文上限 tokens，
    省略键保留原值 / null 清空 / 正整数落盘），未提交的 kinds 等字段保留不
    抹；旧 quants 数组键只读兼容，重写时移除。
    """
    _valid_run_id(run_id)
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    model = body.get("model")
    info = body.get("info")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, "model 须为非空字符串")
    if not isinstance(info, dict):
        raise HTTPException(400, "info 须为对象")
    d = await _read_json(path, "存档")
    cfg = d.get("cfg")
    if not isinstance(cfg, dict):
        raise HTTPException(404, "存档缺少 cfg 段（结构不完整），无法修改环境")
    model = model.strip()
    models = cfg.get("models") or []
    if model not in models:
        raise HTTPException(400, f"存档中不存在模型 {model!r}（本次测速模型：{models}）")
    def _q(key):
        v = info.get(key)
        if v is None:
            return ""
        if not isinstance(v, str):
            raise HTTPException(400, f"{key} 须为字符串")
        v = v.strip()
        if len(v) > 64:
            raise HTTPException(400, f"{key} 过长（≤64 字符）")
        return v

    def _s(key, limit):
        v = info.get(key)
        if v is None:
            return ""
        if not isinstance(v, str):
            raise HTTPException(400, f"{key} 须为字符串")
        return v.strip()[:limit]   # 长度上限镜像 _validate_providers 的部署校验

    mi = cfg.setdefault("model_info", {})
    if not isinstance(mi, dict):
        raise HTTPException(404, "存档 model_info 段结构异常，无法修改环境")
    entry = mi.get(model)
    if entry is None:
        entry = {}
    elif not isinstance(entry, dict):
        raise HTTPException(400, f"模型 {model!r} 的 model_info 条目结构异常，无法修改环境")
    entry.update({
        "deploy_label": _s("deploy_label", 64),
        "quant": _q("quant"),
        "kv_quant": _q("kv_quant"),
        "hardware": _s("hardware", 200),
        "framework": _s("framework", 100),
        "params": _s("params", 300),
    })
    # max_ctx：上下文上限（tokens，正整数）。省略键保留原条目值（旧客户端不
    # 回写时不误抹），null 显式清空，正整数落盘；数值校验口径与
    # _validate_providers 的部署 max_ctx 校验一致（bool/0/负/非数值一律拒
    # 绝）。只改存档显示快照，超窗钳制由部署配置单独承担（ADR-0005）
    if "max_ctx" in info:
        mv = info.get("max_ctx")
        ok = mv is None or (
            not isinstance(mv, bool) and isinstance(mv, (int, float))
            and math.isfinite(mv) and int(mv) > 0)
        if not ok:
            raise HTTPException(400, "max_ctx 须为正整数或 null（清空）")
        entry["max_ctx"] = int(mv) if mv is not None else None
    entry.pop("quants", None)   # 旧数组键不再写出（镜像 kinds/kind 先例）
    # 原子替换写回：与 note 端点同口径
    await _write_json(path, d)
    return {"ok": True, "model_info": entry}


def _point_identity(p: dict) -> tuple:
    """测速点身份五元组（单点复测的请求/存档/复测产出三方匹配口径）。"""
    return (p.get("model"), p.get("scenario"), p.get("ctx_target"),
            p.get("inst_tokens"), p.get("concurrency"))


@app.post("/api/bench/history/{run_id}/retest")
async def bench_history_retest(run_id: str, body: dict):
    """历史存档单点复测：按存档 cfg 窄化到被复测点（同模型/场景/档位/并发，
    agent 矩阵点含指令档）重跑一次完整测速（含 repeats 复测轮次），完成后
    由 _merge_retest 守望任务把新点合并替换回原存档——缓解「个别点失败
    整轮结果不可用」：失败点事后单独补测，无需重跑全矩阵。

    合并留痕：点级 retested_at + 存档顶层 retests 审计数组；复测运行自身
    的存档文件合并成功后删除，未产出可合并点（停止/跳档/identity 漂移）时
    保留为独立记录。旧版 agent 连续任务链测点（turn 字段）的链上下文无法
    独立重建，不支持复测。
    """
    _valid_run_id(run_id)
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    d = await _read_json(path, "存档")
    cfg0 = d.get("cfg")
    if not isinstance(cfg0, dict):
        raise HTTPException(404, "存档缺少 cfg 段（结构不完整），无法复测")
    model = body.get("model")
    scenario = body.get("scenario")
    ctx_target = body.get("ctx_target")
    conc = body.get("concurrency")
    inst = body.get("inst_tokens")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, "model 须为非空字符串")
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise HTTPException(400, f"scenario 须取自白名单: {sorted(SCENARIOS)}")
    if (not isinstance(ctx_target, (int, float)) or isinstance(ctx_target, bool)
            or not 0 <= ctx_target <= 4 * 1048576):
        raise HTTPException(400, "ctx_target 须为 0~4M 的数值")
    if not isinstance(conc, int) or isinstance(conc, bool) or not 1 <= conc <= 64:
        raise HTTPException(400, "concurrency 须为 1~64 的整数")
    ctx_target = int(ctx_target)
    ident = (model, scenario, ctx_target, inst, conc)
    target = next((p for p in d.get("results") or []
                   if isinstance(p, dict) and _point_identity(p) == ident), None)
    if target is None:
        raise HTTPException(404, "存档中找不到该测试点（模型/场景/档位/并发不匹配）")
    if target.get("turn") is not None:
        raise HTTPException(400, "旧版 agent 连续任务链测点不支持单点复测"
                                 "（链上下文无法独立重建）")
    # 窄化 cfg：沿用原运行全部参数（repeats/max_tokens/thinking/reply_mode/
    # 部署快照等），只收敛到被复测点本身；贴边裁减档（ctx_edge）按裁前原
    # 档位下发，引擎重新走贴边逻辑（max_ctx 以当前部署配置为准重新解析）
    cfg = dict(cfg0)
    cfg.pop("note", None)   # 复测运行不继承任务备注（其存档合并后即删除）
    cfg["models"] = [model]
    cfg["scenarios"] = [scenario]
    cfg["concurrencies"] = [conc]
    sc = SCENARIOS[scenario]
    rung = int(target["ctx_edge"]) if target.get("ctx_edge") else ctx_target
    if scenario == "agent":
        if not isinstance(inst, int) or isinstance(inst, bool) or inst < 1:
            raise HTTPException(400, "agent 矩阵点复测需要合法的 inst_tokens")
        cfg["agent_cache_ladder"] = [ctx_target]
        cfg["agent_inst_ladder"] = [inst]
        cfg["ctx_list"] = []
    elif "ladder" in sc:   # 翻译/媒体固定阶梯场景：档位值回对应阶梯字段
        cfg["translate_ladder" if scenario == "translate"
            else f"{scenario}_ladder"] = [rung]
        cfg["ctx_list"] = []
    else:
        cfg["ctx_list"] = [rung]
    res = await bench_start(cfg)   # 复用启动链路：provider 解析/校验/告警/登记
    t = asyncio.create_task(_merge_retest(run_id, res["run_id"], ident))
    _BG_TASKS.add(t)
    t.add_done_callback(_on_drive_done)
    return {"ok": True, "run_id": res["run_id"], "archive": run_id}


async def _merge_retest(archive_id: str, new_run_id: str, ident: tuple):
    """复测运行收尾守望：运行结束（含停止/异常）后把产出的对应测点合并
    替换回原存档（原子写回 + retests 审计留痕），并删除复测运行自身的
    存档文件；未产出可合并点或原存档失踪/损坏时不合并，复测存档保留为
    独立记录（数据不丢）。mock 网关运行不落档也不合并。"""
    run = RUNS.get(new_run_id)
    while run is not None and run.finished_at is None:
        await asyncio.sleep(0.5)
    if run is None or run.mock_seen:
        return
    new_point = next((p for p in run.results
                      if isinstance(p, dict) and _point_identity(p) == ident),
                     None)
    if new_point is None:
        return   # 未产出对应点（停止/超窗跳档等）：复测存档保留为独立记录
    apath = os.path.join(RESULTS_DIR, f"{archive_id}.json")
    try:
        d = await _read_json(apath, "存档")
    except HTTPException:
        return   # 原存档失踪/损坏：不动它，复测存档保留
    results = d.get("results")
    if not isinstance(results, list):
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for i, p in enumerate(results):
        if isinstance(p, dict) and _point_identity(p) == ident:
            new_point["retested_at"] = now
            d.setdefault("retests", []).append({
                "model": ident[0], "scenario": ident[1], "ctx_target": ident[2],
                "inst_tokens": ident[3], "concurrency": ident[4],
                "at": now, "prev_all_ok": p.get("all_ok"),
                "new_all_ok": new_point.get("all_ok"),
                # 口径版本留痕：新旧不同说明这次复测换了口径，读数不可直接比
                "prev_metric_version": p.get("metric_version"),
                "metric_version": new_point.get("metric_version"),
                "retest_run_id": new_run_id})
            results[i] = new_point
            await _write_json(apath, d)
            # 合并成功才删复测运行的独立存档；finished_at 先于 _save 置位，
            # 文件可能尚未落盘，短轮询重试
            rpath = os.path.join(RESULTS_DIR, f"{new_run_id}.json")
            for _ in range(10):
                try:
                    os.remove(rpath)
                    break
                except OSError:
                    await asyncio.sleep(0.3)
            return


def _valid_run_id(run_id: str):
    """run_id 白名单（与生成侧 _run_id 同字符集）：只允许时间戳 + 短随机后缀。

    黑名单式（挡 `/ \\ ..`）够挡穿越，但放开任意字符；白名单更省心——存档
    路由是全部按 id 落盘/删除的入口，且让"非 run_id 文件名"天然不可达。"""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise HTTPException(400, "非法 run_id")


async def _load_archive(run_id: str) -> dict:
    """读存档（删除点/拼接共用）：缺失或 JSON 损坏统一 404（线程池执行）。"""
    _valid_run_id(run_id)
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    return await _read_json(path, "存档")


def _ver_txt(v) -> str:
    """口径版本的人类可读写法（None/缺字段 = 本机制引入前的遗留口径）。"""
    return "遗留" if v is None else str(v)


def _check_point_ident(body: dict) -> tuple:
    """测点身份校验（删除点用，口径与复测一致 + turn 消歧）：返回
    (model, scenario, ctx_target, inst, conc, turn)。turn 用于旧版 agent
    链测点——同五元组按 turn 区分，缺省 None 匹配矩阵/文本/媒体点。"""
    model = body.get("model")
    scenario = body.get("scenario")
    ctx_target = body.get("ctx_target")
    conc = body.get("concurrency")
    inst = body.get("inst_tokens")
    turn = body.get("turn")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, "model 须为非空字符串")
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise HTTPException(400, f"scenario 须取自白名单: {sorted(SCENARIOS)}")
    if (not isinstance(ctx_target, (int, float)) or isinstance(ctx_target, bool)
            or not 0 <= ctx_target <= 4 * 1048576):
        raise HTTPException(400, "ctx_target 须为 0~4M 的数值")
    if not isinstance(conc, int) or isinstance(conc, bool) or not 1 <= conc <= 64:
        raise HTTPException(400, "concurrency 须为 1~64 的整数")
    if inst is not None and (not isinstance(inst, int)
                             or isinstance(inst, bool) or inst < 1):
        raise HTTPException(400, "inst_tokens 须为正整数或 null")
    if turn is not None and (not isinstance(turn, int)
                             or isinstance(turn, bool) or turn < 0):
        raise HTTPException(400, "turn 须为非负整数或 null")
    return (model, scenario, int(ctx_target), inst, conc, turn)


@app.post("/api/bench/history/{run_id}/delete-point")
async def bench_history_delete_point(run_id: str, body: dict):
    """历史存档单点删除：按身份五元组 + turn 移除一个测速点（剔除失败/
    不可信点位），原子写回并在存档顶层 deletions 审计数组留痕（身份、
    删除前 all_ok、时间）。删空 results 允许（存档留壳，整档删除走
    DELETE /api/bench/history/{run_id}）；删除不可恢复，确认由前端兜底。"""
    d = await _load_archive(run_id)
    ident = _check_point_ident(body)
    results = d.get("results")
    if not isinstance(results, list):
        raise HTTPException(404, "存档缺少 results 段（结构不完整）")
    hits = [i for i, p in enumerate(results)
            if isinstance(p, dict)
            and _point_identity(p) == ident[:5]
            and p.get("turn") == ident[5]]
    if not hits:
        raise HTTPException(404, "存档中找不到该测试点（模型/场景/档位/并发/turn 不匹配）")
    if len(hits) > 1:   # 身份六元组仍不唯一：不猜，拒绝并要求人工核查
        raise HTTPException(409, "身份命中多个测点（存档异常），未删除")
    old = results.pop(hits[0])
    d.setdefault("deletions", []).append({
        "model": ident[0], "scenario": ident[1], "ctx_target": ident[2],
        "inst_tokens": ident[3], "concurrency": ident[4], "turn": ident[5],
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prev_all_ok": old.get("all_ok")})
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    await _write_json(path, d)
    return {"ok": True, "removed": 1, "remaining": len(results)}


@app.post("/api/bench/history/{run_id}/splice")
async def bench_history_splice(run_id: str, body: dict):
    """历史存档拼接：把来源存档的全部测速点合并进目标存档——同身份点
    （五元组+turn）以来源覆盖目标、新身份点追加（用途：个别档位在
    原运行测不出，另跑单点运行后拼回主存档）。拼接点打 spliced_from/
    spliced_at 溯源标（前端徽标展示），目标存档顶层 splices 审计留痕
    （来源、新增/覆盖清单、覆盖前后 all_ok）。两边测量口径参数
    （max_tokens/reply_mode/thinking）不一致不阻断，随响应 warnings
    返回由前端提示人工判断。"""
    src = body.get("source_run_id")
    if not isinstance(src, str) or not src.strip():
        raise HTTPException(400, "source_run_id 须为非空字符串")
    src = src.strip()
    _valid_run_id(src)
    _valid_run_id(run_id)
    if src == run_id:
        raise HTTPException(400, "来源与目标不能是同一存档")
    d = await _load_archive(run_id)
    sd = await _load_archive(src)
    results = d.get("results")
    sresults = sd.get("results")
    if not isinstance(results, list):
        raise HTTPException(404, "目标存档缺少 results 段（结构不完整）")
    if not isinstance(sresults, list):
        raise HTTPException(404, "来源存档缺少 results 段（结构不完整）")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    index = {}   # 身份六元组 → results 下标（重复身份后者覆盖前者，与拼接同口径）
    for i, p in enumerate(results):
        if isinstance(p, dict):
            index[(_point_identity(p), p.get("turn"))] = i
    added, replaced = 0, []
    for sp in sresults:
        if not isinstance(sp, dict):
            continue
        np = {**sp, "spliced_from": src, "spliced_at": now}
        key = (_point_identity(sp), sp.get("turn"))
        if key in index:
            old = results[index[key]]
            replaced.append({
                "model": sp.get("model"), "scenario": sp.get("scenario"),
                "ctx_target": sp.get("ctx_target"),
                "inst_tokens": sp.get("inst_tokens"),
                "concurrency": sp.get("concurrency"),
                "turn": sp.get("turn"),
                "prev_all_ok": old.get("all_ok") if isinstance(old, dict) else None,
                "new_all_ok": sp.get("all_ok")})
            results[index[key]] = np
        else:
            index[key] = len(results)
            results.append(np)
            added += 1
    d.setdefault("splices", []).append({
        "source_run_id": src, "at": now, "added": added, "replaced": replaced})
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    await _write_json(path, d)
    # 口径参数差异提示（不阻断）：同图混排不同口径的读数由用户裁决。
    # provider / 部署快照 / 上下文上限必须一起比：把云端点拼进本地存档时，
    # 点身份五元组里没有 provider，"同模型同档位"看着能合并，但卡片头部的
    # 部署信息只来自目标存档，会把两套硬件的读数混在一台机器名下。
    warnings = []
    cfg, scfg = d.get("cfg") or {}, sd.get("cfg") or {}
    for k, label in (("max_tokens", "输出预算"), ("reply_mode", "回复模式"),
                     ("thinking", "思考模式"), ("temperature", "温度")):
        if cfg.get(k) != scfg.get(k):
            warnings.append(f"{label}不一致：目标 {cfg.get(k)} / 来源 {scfg.get(k)}")
    for k, label in (("provider", "provider"), ("provider_label", "provider 名称"),
                     ("gateway_url", "网关地址"), ("model_max_ctx", "上下文上限")):
        if cfg.get(k) != scfg.get(k):
            warnings.append(f"{label}不一致：目标 {cfg.get(k)} / 来源 {scfg.get(k)}")
    # 口径版本差异（ADR-0083）：顶层代表"本次运行的版本"，但点级可能因复测/
    # 拼接而与顶层不同——两边都要比，否则把不同口径的点拼到一起却毫无提示
    dv, sv = d.get("metric_version"), sd.get("metric_version")
    if dv != sv:
        warnings.append(f"口径版本不一致：目标 v{_ver_txt(dv)} / 来源 v{_ver_txt(sv)}"
                        f"——同一张图上混排前请确认这些读数可以这么比")
    src_vers = sorted({_ver_txt(p.get("metric_version")) for p in sresults
                       if isinstance(p, dict)})
    if len(src_vers) > 1:
        warnings.append(f"来源存档内测点口径版本不统一（{', '.join(src_vers)}）")
    tgt_info, src_info = cfg.get("model_info") or {}, scfg.get("model_info") or {}
    for m in sorted(set(tgt_info) & set(src_info)):
        if tgt_info[m] != src_info[m]:
            warnings.append(f"模型 {m} 的部署快照不一致（硬件/框架/量化/参数），"
                            f"拼接后卡片按目标存档展示")
            break
    return {"ok": True, "added": added, "replaced": len(replaced),
            "warnings": warnings}


@app.delete("/api/bench/history/{run_id}")
async def bench_history_delete(run_id: str):
    _valid_run_id(run_id)
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    await asyncio.to_thread(os.remove, path)
    return {"ok": True}


@app.delete("/api/bench/history")
async def bench_history_clear():
    def _clear():
        os.makedirs(RESULTS_DIR, exist_ok=True)
        removed = 0
        for f in os.listdir(RESULTS_DIR):
            p = os.path.join(RESULTS_DIR, f)
            # 只删普通文件：目录名恰好以 .json 结尾（或符号链接）时
            # os.remove 会抛 IsADirectoryError，让整个清空请求 500
            if f.endswith(".json") and os.path.isfile(p) and not os.path.islink(p):
                os.remove(p)
                removed += 1
        return removed
    return {"ok": True, "removed": await asyncio.to_thread(_clear)}


if __name__ == "__main__":
    import uvicorn
    # 缺省只绑本机回环：全端点无鉴权，绑 0.0.0.0 会让局域网内任何人可触发
    # 对云端付费网关的测速（烧额度）并删除历史；确需局域网访问时显式
    # HOST=0.0.0.0 并自行评估风险
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8501)))
