"""MCP 服务：LLM-SPEED 测速台的 agent 入口（stdio）。

定位：与浏览器平级的另一个客户端——薄包装，全部能力经由运行中的 Web 服务
REST API（provider 解析 / API Key 注入 / cfg 白名单钳制都在服务端），本进程
只做 JSON 往返与 markdown 渲染：不 import bench、不接触 .env 凭据、不复制
校验逻辑。前提：Web 服务已启动（./start.sh，缺省 http://127.0.0.1:8501）。

环境变量：
  LLM_SPEED_BASE   测速台服务地址，缺省 http://127.0.0.1:8501
                   （到该地址的请求**一律直连**，不吃 HTTP_PROXY/ALL_PROXY，原因见 _KW）

工具一览：
  list_providers   已配置的 provider：部署环境（硬件/框架/量化）+ 模型声明（上下文上限/类型）
  list_models      某 provider 网关后的模型列表
  list_scenarios   场景白名单与默认阶梯（上下文档位/并发/agent 矩阵/媒体档位）
  bench_start      发起测速（异步，立即返回 run_id）
  bench_status     查询运行进度与最近状态
  bench_result     取完成运行的数据卡片（markdown）
  bench_stop       停止运行
  bench_history    历史存档列表
  bench_compare    多次运行同场景对比表（跨 provider 亦可）

客户端配置示例（Claude Code / Kimi Code 等）：
  {"mcpServers": {"llm-speed": {"command": "<repo>/.venv/bin/python",
                                "args": ["<repo>/mcp_server.py"]}}}
"""
import asyncio
import json
import os
import time

import httpx
from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("LLM_SPEED_BASE", "http://127.0.0.1:8501").rstrip("/")

# 两个 HTTP 客户端共用的构造参数。**trust_env=False 是必需的**：MCP 客户端把本进程
# 当子进程拉起，宿主环境里常挂着代理变量（HTTP_PROXY/ALL_PROXY）；httpx 默认
# trust_env=True 就会把对 127.0.0.1 的请求交给代理，代理要么回 502（表现为「服务端
# 拒绝」——服务明明活着），要么连不上代理（表现为「无法连接」）。靠 NO_PROXY 躲不掉：
# 常见写法是 CIDR（127.0.0.0/8），httpx 不认。测速台就是本机服务（要够远程的人自然
# 会自己搭隧道），不需要代理；真需要代理的人拿 LLM_SPEED_BASE 指向自己的入口。
_KW = {"trust_env": False}

mcp = FastMCP(
    "llm-speed",
    instructions=(
        "LLM-SPEED 本地 LLM 测速台：对 OpenAI 兼容网关后的模型做 prefill/decode "
        "测速（只测速度不测精度）。典型流程：list_providers/list_models 选模型 → "
        "bench_start 发起（异步，立即返回 run_id）→ bench_status 轮询进度 → "
        "bench_result 取数据卡片；多次运行的横向对比用 bench_compare。"
        "测速是分钟~十分钟级的真实打流，不要在未确认参数时盲目发起。"
    ),
)


# ---------------------------------------------------------------------------
# HTTP：统一错误面——连不上提示先启动服务；4xx/5xx 透出服务端 detail（服务端
# 的 400 文案本身就是可行动的参数修正指引，原样上交）
# ---------------------------------------------------------------------------

async def _api(method: str, path: str, *, timeout: float = 15.0, **kw):
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=timeout, **_KW) as c:
            r = await c.request(method, path, **kw)
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"无法连接 LLM-SPEED 服务 {BASE}（请先 ./start.sh 启动；"
            f"非默认地址用环境变量 LLM_SPEED_BASE 指定）: {e}") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail") or r.text[:300]
        except ValueError:
            detail = r.text[:300]
        raise RuntimeError(f"服务端拒绝（HTTP {r.status_code}）：{detail}")
    return r.json()


async def _sse_snapshot(run_id: str, budget_s: float = 3.0,
                        max_events: int = 5000) -> list[dict]:
    """一次性抓取 SSE 回放缓冲（cfg/status/point/skipped/done/stopped/error）。

    已结束的运行：服务端回放完即关闭流，自然返回；进行中的运行：回放后进入
    心跳等待，按总预算掐断——快照只为状态面，不跟随实时流。"""
    events: list[dict] = []
    t0 = time.monotonic()
    try:
        to = httpx.Timeout(5.0, read=min(2.0, budget_s))
        async with httpx.AsyncClient(base_url=BASE, timeout=to, **_KW) as c:
            async with c.stream("GET", f"/api/bench/events/{run_id}") as r:
                if r.status_code != 200:
                    return events
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            events.append(json.loads(line[6:]))
                        except ValueError:
                            continue
                        if len(events) >= max_events:
                            break
                    if time.monotonic() - t0 > budget_s:
                        break
    except httpx.HTTPError:
        pass
    return events


# ---------------------------------------------------------------------------
# 格式化：纯函数（输入为服务端 JSON dict），单测直接喂存档结构
# ---------------------------------------------------------------------------

def _num(v) -> str:
    return "—" if v is None else str(v)


def _gear_of(p: dict):
    """逻辑档位：贴边裁减点归到原档位（与前端 §pure cmpGearOf 同口径——
    同一逻辑档位若按裁后值取会被拆成两格，对比表出现错位空行）。"""
    return p.get("ctx_edge") or p.get("ctx_target")


def _gear_label(scenario: str, gear, unit: str | None) -> str:
    if gear is None:
        return "—"
    if scenario in ("asr",):
        return f"{gear}s"
    if scenario in ("ocr",):
        return f"{gear} 张"
    if scenario in ("translate", "tts"):
        return f"{gear} 字"
    # llm / agent：tokens 折 K
    g = int(gear)
    if g <= 0:
        return "0K"
    return f"{g // 1024}K" if g % 1024 == 0 else f"{g / 1024:.1f}K"


def _status_of(p: dict) -> str:
    if p.get("all_ok") is False:
        return "✗ " + str(p.get("err") or p.get("anomaly") or "失败")[:40]
    if p.get("anomaly"):
        return "⚠ " + str(p["anomaly"])[:40]
    if p.get("stopped_mid_point"):
        return "✗ 中途停止"
    return "✓"


def _hit_of(p: dict) -> str:
    """缓存命中三态：API 回传真值 / 估算（≈）/ 无读数（—），宁缺毋假。"""
    n = p.get("cache_hit_tokens")
    if n is None:
        return "—"
    if p.get("cache_reported") is False:
        return f"≈{n}"
    return str(n)


def _dep_line(model: str, info: dict | None) -> str:
    if not info:
        return f"**{model}**（未映射部署环境）"
    parts = [x for x in (info.get("deploy_label"), info.get("hardware"),
                         info.get("framework"), info.get("params"),
                         info.get("quant") and f"量化 {info['quant']}",
                         info.get("kv_quant") and f"KV {info['kv_quant']}",
                         info.get("max_ctx") and f"ctx {info['max_ctx']}")
             if x]
    return f"**{model}** · " + (" · ".join(str(p) for p in parts) if parts
                                else "（部署信息未填）")


def _point_rows_llm(pts: list[dict], scenario: str, unit: str | None):
    rows = []
    for p in pts:
        rows.append([
            _gear_label(scenario, _gear_of(p), unit),
            _num(p.get("concurrency")),
            _num(p.get("prompt_tokens")),
            _num(p.get("ttft_s")),
            _num(p.get("prefill_tok_s")),
            _num(p.get("decode_tok_s")),
            _num(p.get("total_span_s") or p.get("total_s") or p.get("batch_time_s")),
            _hit_of(p),
            _status_of(p),
        ])
    return rows


def _point_rows_media(pts: list[dict], scenario: str, unit: str | None):
    rows = []
    for p in pts:
        if scenario == "ocr":
            rows.append([_gear_label(scenario, _gear_of(p), unit),
                         _num(p.get("concurrency")), _num(p.get("ttft_s")),
                         _num(p.get("prefill_tok_s")), _num(p.get("decode_tok_s")),
                         _num(p.get("total_span_s") or p.get("batch_time_s")),
                         _status_of(p)])
        else:   # asr / tts
            rows.append([_gear_label(scenario, _gear_of(p), unit),
                         _num(p.get("concurrency")), _num(p.get("audio_s")),
                         _num(p.get("ttfa_s") if scenario == "tts"
                              else p.get("elapsed_s")),
                         _num(p.get("rtf")), _num(p.get("speed_x")),
                         _num(p.get("audio_min_per_min")), _status_of(p)])
    return rows


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


_LLM_HEADER = ["档位", "并发", "输入 tok", "TTFT s", "prefill tok/s",
               "decode tok/s", "端到端 s", "缓存命中", "状态"]
_OCR_HEADER = ["档位", "并发", "TTFT s", "prefill tok/s", "decode tok/s",
               "端到端 s", "状态"]
_ASR_HEADER = ["档位", "并发", "音频 s", "耗时 s", "RTF", "倍速",
               "吞吐 音频分/分", "状态"]
_TTS_HEADER = ["档位", "并发", "音频 s", "TTFA s", "RTF", "倍速",
               "吞吐 音频分/分", "状态"]


def _scenario_sections(model: str, scen: str, pts: list[dict],
                       label: str, kind: str, unit: str | None) -> str:
    """单模型单场景的卡片小节：llm/媒体用明细表，agent 用缓存×指令矩阵。"""
    out = [f"### {label}（{scen}）"]
    if scen == "agent":
        concs = sorted({p.get("concurrency") for p in pts})
        for conc in concs:
            sub = [p for p in pts if p.get("concurrency") == conc]
            caches = sorted({_gear_of(p) for p in sub})
            insts = sorted({p.get("inst_tokens") for p in sub})
            out.append(f"并发 {conc}（单元格 = TTFT s / decode tok/s；"
                       f"缓存命中见明细存档）：")
            header = ["缓存 \\ 指令"] + [str(i) for i in insts]
            rows = []
            for c in caches:
                row = [_gear_label(scen, c, unit)]
                for i in insts:
                    p = next((q for q in sub
                              if _gear_of(q) == c and q.get("inst_tokens") == i),
                             None)
                    if p is None:
                        row.append("—")
                    elif p.get("all_ok") is False:
                        row.append("✗")
                    else:
                        row.append(f"{_num(p.get('ttft_s'))} / "
                                   f"{_num(p.get('decode_tok_s'))}")
                rows.append(row)
            out.append(_md_table(header, rows))
        return "\n\n".join(out)
    if kind == "llm":
        out.append(_md_table(_LLM_HEADER, _point_rows_llm(pts, scen, unit)))
    elif scen == "ocr":
        out.append(_md_table(_OCR_HEADER, _point_rows_media(pts, scen, unit)))
    else:
        hdr = _ASR_HEADER if scen == "asr" else _TTS_HEADER
        out.append(_md_table(hdr, _point_rows_media(pts, scen, unit)))
    return "\n\n".join(out)


def render_card(archive: dict, scen_meta: dict | None = None) -> str:
    """一次运行的数据卡片（markdown）：头部 = 运行/provider/部署/参数，
    正文按 模型 × 场景 分节。"""
    meta = scen_meta or {}
    cfg = archive.get("cfg") or {}
    pts = archive.get("results") or []
    label_of = lambda s: (meta.get(s) or {}).get("label") or s   # noqa: E731
    kind_of = lambda s: (meta.get(s) or {}).get("kind") or "llm"  # noqa: E731
    unit_of = lambda s: (meta.get(s) or {}).get("unit")

    title = cfg.get("provider_label") or cfg.get("provider") or "未知 provider"
    out = [f"# ⚡ 测速数据卡片 · {title}", ""]
    line = f"**运行** `{archive.get('run_id')}` · {archive.get('started_at')}"
    if cfg.get("note"):
        line += f" · 备注：{cfg['note']}"
    out.append(line)
    # 云端展示网关地址，本地部署隐去（与前端卡片同口径）
    if cfg.get("gateway_url") and cfg.get("provider_local") is False:
        out.append(f"**网关** {cfg['gateway_url']}")
    params = []
    for k, label in (("max_tokens", "max_tokens"), ("repeats", "repeats"),
                     ("reply_mode", "reply_mode"), ("temperature", "temperature"),
                     ("thinking", "thinking"), ("timeout_s", "timeout_s")):
        if cfg.get(k) is not None:
            params.append(f"{label}={cfg[k]}")
    if params:
        out.append("**参数** " + " · ".join(str(p) for p in params))
    out.append("")
    if not pts:
        out.append("（无测速点——运行可能被停止或全部跳档；"
                   "可用 bench_status 查看事件明细）")
        return "\n".join(out)

    models = sorted({p.get("model") for p in pts})
    info = cfg.get("model_info") or {}
    for m in models:
        out.append(f"## {_dep_line(m, info.get(m))}")
        out.append("")
        mpts = [p for p in pts if p.get("model") == m]
        for scen in dict.fromkeys(p.get("scenario") for p in mpts):
            spts = sorted((p for p in mpts if p.get("scenario") == scen),
                          key=lambda p: ((_gear_of(p) or 0),
                                         p.get("inst_tokens") or 0,
                                         p.get("concurrency") or 0))
            out.append(_scenario_sections(m, scen, spts, label_of(scen),
                                          kind_of(scen), unit_of(scen)))
            out.append("")
    return "\n".join(out).rstrip()


def render_compare(runs: list[dict], scenario: str,
                   scen_meta: dict | None = None) -> str:
    """多次运行同场景对比表：行 = 逻辑档位（×指令/并发），列 = 每 run×model
    的 TTFT / prefill / decode（媒体场景为 RTF / 倍速）。跨 provider 对齐靠
    _gear_of 的贴边归一。"""
    meta = scen_meta or {}
    label = (meta.get(scenario) or {}).get("label") or scenario
    kind = (meta.get(scenario) or {}).get("kind") or "llm"
    unit = (meta.get(scenario) or {}).get("unit")

    # 列头：run_id 取尾段（短），同 run 多模型逐模型一列
    cols = []   # (标题, run_idx, model, {行键: point})
    head = [f"# ⚖ 对比测速 · {label}（{scenario}）", ""]
    for i, arc in enumerate(runs):
        cfg = arc.get("cfg") or {}
        rid = str(arc.get("run_id") or f"run{i}")
        prov = cfg.get("provider_label") or cfg.get("provider") or ""
        head.append(f"- **{rid[-4:]}** `{rid}` · {prov} · "
                    f"{arc.get('started_at')}"
                    + (f" · 备注：{cfg['note']}" if cfg.get("note") else ""))
        info = cfg.get("model_info") or {}
        models = sorted({p.get("model") for p in arc.get("results") or []
                         if p.get("scenario") == scenario})
        for m in models:
            dep = info.get(m) or {}
            dep_short = dep.get("deploy_label") or dep.get("quant") or ""
            cols.append((f"{m}（{rid[-4:]}{'·' + dep_short if dep_short else ''}）",
                         m, {(_gear_of(p),
                              p.get("inst_tokens") if scenario == "agent"
                              else p.get("concurrency")): p
                             for p in (arc.get("results") or [])
                             if p.get("scenario") == scenario
                             and p.get("model") == m}))
    head.append("")

    row_keys = sorted({k for _, _, pts in cols for k in pts},
                      key=lambda k: ((k[0] or 0), (k[1] or 0)))
    if not row_keys:
        return "\n".join(head) + "\n（各运行在该场景下均无测速点）"

    media = kind in ("asr", "tts")
    sub = ["RTF", "倍速"] if media else ["TTFT s", "prefill tok/s", "decode tok/s"]
    dim = "指令" if scenario == "agent" else "并发"
    header = ["档位", dim] + [f"{t} · {s}" for t, _, _ in cols for s in sub]
    rows = []
    for gear, sec in row_keys:
        row = [_gear_label(scenario, gear, unit),
               _num(sec) if scenario == "agent" else _num(sec)]
        for _, _, pts in cols:
            p = pts.get((gear, sec))
            if p is None:
                row += ["—"] * len(sub)
            elif p.get("all_ok") is False:
                row += ["✗"] * len(sub)
            elif media:
                row += [_num(p.get("rtf")), _num(p.get("speed_x"))]
            else:
                row += [_num(p.get("ttft_s")), _num(p.get("prefill_tok_s")),
                        _num(p.get("decode_tok_s"))]
        rows.append(row)
    return "\n".join(head) + "\n" + _md_table(header, rows)


def _scen_meta(cfg_resp: dict) -> dict:
    return {s["key"]: s for s in cfg_resp.get("scenarios") or [] if s.get("key")}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_providers() -> str:
    """列出测速台已配置的 provider：名称、标签、本地/云端、API Key 是否已配置、
    部署环境映射（模型 → 硬件/框架/参数/量化）与模型声明（上下文上限/模型类型）。"""
    d = await _api("GET", "/api/config")
    out = []
    for p in d.get("providers") or []:
        loc = "本地" if p.get("local") else "云端"
        key = "Key✓" if p.get("has_key") else "Key✗"
        out.append(f"## {p.get('name')}（{p.get('label') or p.get('name')}）· "
                   f"{loc} · {key}")
        urls = p.get("gateway_urls") or ([p["gateway_url"]] if p.get("gateway_url")
                                         else [])
        for u in urls:
            out.append(f"- 网关：{u}")
        for dep in p.get("deployments") or []:
            out.append(
                f"- 部署「{dep.get('label') or ''}」："
                f"{', '.join(dep.get('models') or [])} · "
                f"{dep.get('hardware') or ''} · {dep.get('framework') or ''} · "
                f"{dep.get('params') or ''} · 量化 {','.join(dep.get('quants') or [])}"
                f" · KV {','.join(dep.get('kv_quants') or [])}"
                f"{'（激活）' if dep.get('active') else ''}")
        # 上限与类型的唯一事实源是 `model_profiles`（ADR-0118/0122）：`deployments` 里的
        # 同名字段已废弃且不再落盘，读它只会得到 "?"（服务端 GET 已把旧值折算进声明）
        for f in p.get("model_profiles") or []:
            if not isinstance(f, dict):
                continue
            out.append(f"- 声明「{f.get('model') or ''}」："
                       f"max_ctx {f.get('max_ctx') or '不限'} · "
                       f"类型 {'/'.join(f.get('kinds') or ['llm'])}")
        if not p.get("model_profiles"):
            out.append("- 声明：无（未声明 = 不设上下文上限、类型按 llm）")
        out.append("")
    if not out:
        return ("尚未配置任何 provider——在页面「Provider 管理」抽屉添加，"
                "或编辑 config.json。")
    return "\n".join(out).rstrip()


@mcp.tool()
async def list_models(provider: str | None = None) -> str:
    """列出某 provider 网关后的可选模型（透传网关 /v1/models，凭据由服务端注入）。
    provider 为 config.json 里的名称；缺省取首个 provider。"""
    q = f"?provider={provider}" if provider else ""
    d = await _api("GET", f"/api/models{q}")
    ids = d.get("models") or []
    head = f"provider={provider or '（首个）'} · 共 {len(ids)} 个模型："
    return head + "\n" + "\n".join(f"- {m}" for m in ids)


@mcp.tool()
async def list_scenarios() -> str:
    """列出测速场景白名单与默认阶梯：场景 key（bench_start 的 scenarios 参数用）、
    类型、默认上下文档位/并发、agent 缓存×指令矩阵、媒体场景档位。"""
    d = await _api("GET", "/api/config")
    out = ["场景（key · 名称 · 类型 · 默认阶梯）："]
    for s in d.get("scenarios") or []:
        if s.get("key") == "agent":
            ladder = (f"缓存 {s.get('cache_ladder')} × 指令 {s.get('inst_ladder')}"
                      f" tokens")
        elif s.get("ladder"):
            ladder = f"{s['ladder']}{s.get('unit') or ''}"
        else:
            ladder = f"ctx_list {d.get('ctx_list')} tokens"
        out.append(f"- `{s.get('key')}` · {s.get('label')} · "
                   f"{s.get('kind', 'llm')} · {ladder}")
    out.append(f"\n默认并发：{d.get('concurrencies')}")
    out.append("注意：一次运行只允许一种类型（llm/asr/ocr/tts 不可混选）；"
               "纯 agent 场景不需要 ctx_list。")
    return "\n".join(out)


@mcp.tool()
async def bench_start(
    provider: str,
    models: list[str],
    scenarios: list[str],
    ctx_list: list[int] | None = None,
    concurrencies: list[int] | None = None,
    max_tokens: int | None = None,
    repeats: int | None = None,
    timeout_s: int | None = None,
    reply_mode: str | None = None,
    thinking: str | None = None,
    temperature: float | None = None,
    cache: str | None = None,
    cpt: float | None = None,
    note: str | None = None,
    agent_cache_ladder: list[int] | None = None,
    agent_inst_ladder: list[int] | None = None,
    translate_ladder: list[int] | None = None,
    asr_ladder: list[int] | None = None,
    ocr_ladder: list[int] | None = None,
    tts_ladder: list[int] | None = None,
) -> str:
    """发起一轮测速（异步，立即返回 run_id）。测速是分钟~十分钟级的真实打流。

    provider：config.json 里的 provider 名（list_providers 可见）；
    models：网关模型 id，可多选（同一次运行内自然形成同档位对比）；
    scenarios：场景 key（list_scenarios 可见），一次运行限一种类型；
    ctx_list：上下文档位 tokens（文本场景必填，如 [0,4096,32768,131072]；
    纯 agent 场景不需要）；
    concurrencies：并发档（云端 provider 服务端强制单发）；
    reply_mode：free（默认）/ echo（模板续写，对齐高复用编辑场景）；
    thinking：auto（默认）/ enabled / disabled；
    cache：bust / stable（前缀缓存利用策略，缺省走服务端默认）；
    timeout_s：连续无输出判断开的秒数（不是单请求总时长，长 prefill 档位要给足）；
    agent_cache_ladder / agent_inst_ladder：agent 矩阵自定义阶梯；
    *_ladder：媒体/翻译场景自定义档位。其余缺省走服务端默认。"""
    cfg = {"provider": provider, "models": models, "scenarios": scenarios}
    for k, v in (("ctx_list", ctx_list), ("concurrencies", concurrencies),
                 ("max_tokens", max_tokens), ("repeats", repeats),
                 ("timeout_s", timeout_s), ("reply_mode", reply_mode),
                 ("thinking", thinking), ("temperature", temperature),
                 ("cache", cache), ("cpt", cpt), ("note", note),
                 ("agent_cache_ladder", agent_cache_ladder),
                 ("agent_inst_ladder", agent_inst_ladder),
                 ("translate_ladder", translate_ladder),
                 ("asr_ladder", asr_ladder), ("ocr_ladder", ocr_ladder),
                 ("tts_ladder", tts_ladder)):
        if v is not None:
            cfg[k] = v
    d = await _api("POST", "/api/bench/start", timeout=120, json=cfg)
    run_id = d["run_id"]
    out = [f"已启动测速 `{run_id}`",
           f"provider={provider} · models={models} · scenarios={scenarios}"]
    # 启动即抓一次事件快照：服务端注入的显式告警（模型未映射部署、agent 矩阵
    # 超预算将跳档等）随 status 事件下发，启动时就要让调用方看见
    await asyncio.sleep(0.5)
    warns = [e.get("msg") for e in await _sse_snapshot(run_id)
             if e.get("type") == "status" and str(e.get("msg") or "").startswith("⚠")]
    if warns:
        out.append("服务端告警：")
        out += [f"  {w}" for w in warns]
    out.append(f"进度用 bench_status(\"{run_id}\") 查询；"
               f"完成后 bench_result(\"{run_id}\") 取数据卡片。")
    return "\n".join(out)


@mcp.tool()
async def bench_status(run_id: str) -> str:
    """查询测速运行状态：进行中（已完成点/总点数 + 最近状态消息）/
    已完成（测点数与取数指引）/ 已停止 / 出错。"""
    active = (await _api("GET", "/api/bench/active")).get("runs") or []
    run = next((r for r in active if r.get("run_id") == run_id), None)
    events = await _sse_snapshot(run_id)
    msgs = [e for e in events if e.get("type") == "status"]
    tail = []
    for e in events:
        if e.get("type") in ("done", "stopped", "error"):
            tail.append(e)
    if run:
        out = [f"运行 `{run_id}` · 进行中",
               f"{run.get('provider')} · {', '.join(run.get('models') or [])}",
               f"进度：{run.get('done_points')}/{run.get('total_points')} 测点"]
    elif tail and tail[-1].get("type") == "done":
        return (f"运行 `{run_id}` · 已完成（{sum(1 for e in events if e.get('type') == 'point')} "
                f"个测点）\n用 bench_result(\"{run_id}\") 取数据卡片。")
    elif tail and tail[-1].get("type") == "stopped":
        out = [f"运行 `{run_id}` · 已被停止"]
    elif tail and tail[-1].get("type") == "error":
        out = [f"运行 `{run_id}` · 出错：{tail[-1].get('msg')}"]
    else:
        try:
            arc = await _api("GET", f"/api/bench/history/{run_id}", timeout=60)
        except RuntimeError:
            return (f"运行 `{run_id}` · 未在进行中列表，也未找到存档——"
                    "可能刚结束仍在落盘（稍后再试），或 run_id 有误/已过期清理。")
        return (f"运行 `{run_id}` · 已完成（{len(arc.get('results') or [])} 个测点）"
                f"\n用 bench_result(\"{run_id}\") 取数据卡片。")
    if msgs:
        out.append("最近状态：")
        for e in msgs[-5:]:
            ts = time.strftime("%H:%M:%S", time.localtime(e.get("ts") or 0))
            out.append(f"  · {ts} {e.get('msg')}")
    return "\n".join(out)


@mcp.tool()
async def bench_result(run_id: str) -> str:
    """取已完成运行的数据卡片（markdown）：部署环境 + 参数 + 每模型每场景的
    测速读数表（TTFT/prefill/decode/缓存命中/状态；agent 为缓存×指令矩阵）。
    运行未结束时返回进度提示。完整的测速卡片 PNG 请在 Web 页面历史面板生成
    （卡片渲染在前端，服务端产出的是同口径数据）。"""
    active = (await _api("GET", "/api/bench/active")).get("runs") or []
    if any(r.get("run_id") == run_id for r in active):
        return (f"运行 `{run_id}` 仍在进行，先用 bench_status 查看进度，"
                "完成后再取结果。")
    arc = await _api("GET", f"/api/bench/history/{run_id}", timeout=60)
    meta = _scen_meta(await _api("GET", "/api/config"))
    return render_card(arc, meta)


@mcp.tool()
async def bench_stop(run_id: str) -> str:
    """停止进行中的测速运行（在途测点以中断形态留痕，已完成轮次保留存档）。"""
    await _api("POST", f"/api/bench/stop/{run_id}")
    return f"已请求停止 `{run_id}`——在途测点收尾后存档，可用 bench_result 查看已得部分。"


@mcp.tool()
async def bench_history() -> str:
    """列出历史测速存档（最近 50 条）：run_id、开始时间、模型、场景、测点数、备注。"""
    d = await _api("GET", "/api/bench/history")
    meta = _scen_meta(await _api("GET", "/api/config"))
    rows = []
    for h in d.get("history") or []:
        scens = "/".join((meta.get(s) or {}).get("label") or s
                         for s in h.get("scenarios") or [])
        rows.append([f"`{h.get('run_id')}`", str(h.get("started_at")),
                     ", ".join(h.get("models") or []), scens,
                     str(h.get("n_points")), h.get("note") or ""])
    if not rows:
        return "暂无历史存档。"
    return _md_table(["run_id", "开始", "模型", "场景", "点数", "备注"], rows)


@mcp.tool()
async def bench_compare(run_ids: list[str], scenario: str | None = None) -> str:
    """对比多次运行在同一场景下的读数：行 = 逻辑档位（贴边裁减点归一到原档位），
    列 = 每运行×模型的 TTFT/prefill/decode（媒体场景为 RTF/倍速）。
    跨 provider 对比亦可。scenario 缺省时要求各运行只有一个共同场景。
    同 provider 的多模型对比更省事的做法：bench_start 一次带多个 models。"""
    if len(run_ids) < 2:
        raise RuntimeError("bench_compare 至少需要 2 个 run_id")
    runs = [await _api("GET", f"/api/bench/history/{rid}", timeout=60)
            for rid in run_ids]
    meta = _scen_meta(await _api("GET", "/api/config"))
    if scenario is None:
        common = set((runs[0].get("cfg") or {}).get("scenarios") or [])
        for r in runs[1:]:
            common &= set((r.get("cfg") or {}).get("scenarios") or [])
        if len(common) != 1:
            raise RuntimeError(
                f"各运行的共同场景不唯一（{sorted(common)}），"
                "请显式指定 scenario 参数")
        scenario = common.pop()
    missing = [str(r.get("run_id")) for r in runs
               if not any(p.get("scenario") == scenario
                          for p in r.get("results") or [])]
    if missing:
        raise RuntimeError(f"运行 {missing} 在场景 {scenario} 下无测速点")
    return render_compare(runs, scenario, meta)


if __name__ == "__main__":
    mcp.run()   # stdio 传输：由 MCP 客户端以子进程方式拉起
