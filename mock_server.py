"""假 OpenAI 兼容网关，用于验证测速链路（无需真实模型）。

模拟特性：
- prefill 速度 ≈ MOCK_PP tok/s（TTFT 随 prompt 长度线性增长）
- decode 速度 ≈ MOCK_TG tok/s（流式逐 token 输出）
- 返回 usage（模拟 stream_options.include_usage 行为）

启动：MOCK_PP=1200 MOCK_TG=60 python mock_server.py
DeepSeek 风格（chat 端点不带 /v1 前缀）：MOCK_NO_V1=1 python mock_server.py
模拟分词偏差（验证输入系数校准）：MOCK_CPT=3.9；模拟严格校验网关：MOCK_REJECT_THINKING=1
模拟 decode 起步突发（首批 token 一次性吐出）：MOCK_BURST=15
模拟投机解码周期性交付（每事件 k token、间隔 k/tg）：MOCK_ACCEPT=4
模拟 prefill 期 SSE 注释心跳（秒间隔，如 0.3）：MOCK_KEEPALIVE=0.3
模拟响应头延迟到首个内容 token（prefill 期间无任何字节，LiteLLM→vLLM 链路行为）：MOCK_DELAY_HEADERS=1
模拟 decode 中段一次停滞（秒，验证空窗诊断）：MOCK_STALL=4
模拟流末不发 usage 帧（测速端走估算兜底）：MOCK_NO_USAGE=1
模拟每 token 多字符输出（英文形态：1 事件/token、chars/token≈4）：MOCK_CHUNK_CHARS=4
模拟服务端并发 prefill 串行化（全局排队）：MOCK_SERIALIZE=1
模拟 /models 慢响应（秒，验证 RTT 基线期停止）：MOCK_MODELS_DELAY=5
模拟模型上下文窗口上限（tokens，prompt+max_tokens 超限返回 400 context exceeded）：MOCK_MAX_CTX=4096
模拟 fastllm 系「软超窗」（超限不报错，200 + 正文 "prompt too long"）：MOCK_SOFT_MAX_CTX=4096
  占位串返回通道（content=正文，reasoning=走 reasoning_content，复现思考模型形态）：MOCK_SOFT_MAX_CTX_FIELD=reasoning
  占位串自定义（默认 "prompt too long"；自定义串用于验证测速端输出塌缩守卫）：MOCK_SOFT_TEXT="..."
模拟逐请求交替 decode 速度（tok/s 逗号分隔循环，双峰复现）：MOCK_TG_PATTERN="60,120"
模拟服务端前缀缓存（append-only 增长的 prompt 只增量 prefill，usage 回传命中）：MOCK_CACHE=1
模拟缓存命中但读取慢（缓存生效、不回传命中字段，命中前缀 TTFT = 命中tokens/N；N≫PP 时走平判别失效，验证全量预期/佐证通道判别）：MOCK_CACHE_READ_TOK_S=10000
模拟忽略 max_tokens（只认 max_completion_tokens，opencode zen 行为；旧名请求输出跑飞 4× 上限）：MOCK_IGNORE_MAX_TOKENS=1
模拟严格校验网关拒收 max_completion_tokens（400 Unrecognized request argument）：MOCK_REJECT_MCT=1
模拟首测提前停止（一次性脚本钩子：下一发 chat 请求 finish=stop、仅 N tokens、带真 usage，之后恢复正常满预算输出；验证测速端弃测重测全链路 ADR-0032）：MOCK_EARLY_STOP=4
模拟 agent 测量批提前停止（接下来 N 发「带指令材料」的 chat 请求逐发提前停止、每次仅 4 tokens（实测 4/512 形态）、用后归零；只命中末条消息 >64 字符的请求，agent 预热/基线的超短任务不命中——脚本化 agent 矩阵首测异常/重测正常或持续异常）：MOCK_EARLY_STOP_LONG=1
模拟 ASR 处理速度（音频秒/秒，50=50 倍实时）：MOCK_ASR_PP=50
模拟 TTS 合成语速（字/秒，决定音频时长）：MOCK_TTS_RATE=4.5
模拟 TTS 合成速度（相对实时的倍速）：MOCK_TTS_XR=20
模拟网关瞬时抖动（接下来 N 发 chat 请求直接 HTTP 500，用后归零恢复正常；验证测速端兜底重试）：MOCK_FLAKY_FAIL=1
模拟未自报超窗的秒拒（prompt+max_tokens 超 N 返回 500，报文不含 context/exceed 超窗关键词——fastllm 显存/KV 不足形态；验证贴边档回退裁减）：MOCK_HARD_FAIL_CTX=4096
"""
import asyncio
import io
import json
import math
import os
import wave
from array import array

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

PP = float(os.environ.get("MOCK_PP", "1200"))    # 模拟 prefill 速度 tok/s
TG = float(os.environ.get("MOCK_TG", "60"))     # 模拟单请求 decode 速度 tok/s
CPT = float(os.environ.get("MOCK_CPT", "1.8"))  # 字符/token，需与测速端估算一致才准
CH = int(os.environ.get("MOCK_CHUNK_CHARS", "1"))  # 每 token 事件的字符数：置 4 模拟英文输出（1 事件/token 但 chars/token≈4，验证 chunk 直计不虚发）
REASON = int(os.environ.get("MOCK_REASON", "0"))   # 正文前先输出多少个 reasoning_content token
NO_CONTENT = os.environ.get("MOCK_NO_CONTENT", "")  # 置 1 则只输出思考、无正文
REJECT_THINKING = os.environ.get("MOCK_REJECT_THINKING", "")  # 置 1 则带 thinking 参数的请求 400
REJECT_TEMPERATURE = os.environ.get("MOCK_REJECT_TEMPERATURE", "")  # 置 1 则带 temperature 参数的请求 400
BURST = int(os.environ.get("MOCK_BURST", "0"))  # 首个内容事件一次性携带的 token 数（模拟服务端缓冲/投机采样突发）
ACCEPT = max(1, int(os.environ.get("MOCK_ACCEPT", "1")))  # 每个 decode 事件携带的 token 数（>1 模拟投机解码 draft+verify 周期交付，间隔 k/tg）
KA = float(os.environ.get("MOCK_KEEPALIVE", "0"))  # prefill 期间 SSE 注释心跳间隔秒（0=关闭；模拟 vLLM/代理类网关的高频 keepalive）
DELAY_HDR = os.environ.get("MOCK_DELAY_HEADERS", "")  # 置 1 则响应头延迟到首个内容 token（prefill 期间无任何字节）
STALL = float(os.environ.get("MOCK_STALL", "0"))  # decode 中段插入一次停滞秒数（模拟传输/调度空窗）
NO_V1 = os.environ.get("MOCK_NO_V1", "")  # 置 1 则 /v1/chat/completions 404（chat 端点无前缀，DeepSeek 风格）
NO_USAGE = os.environ.get("MOCK_NO_USAGE", "")  # 置 1 则流末不发 usage 帧（测速端 usage_real=False，走估算兜底）
SERIALIZE = os.environ.get("MOCK_SERIALIZE", "")  # 置 1 则 prefill 全局串行（模拟服务端并发 prefill 串行化）
MODELS_DELAY = float(os.environ.get("MOCK_MODELS_DELAY", "0"))  # /models 响应前 sleep 秒数（模拟 RTT 基线期慢网关）
MAX_CTX = int(os.environ.get("MOCK_MAX_CTX", "0"))  # 置 N 则 prompt+max_tokens 超 N 返回 400 超窗（验证删减上下文重试）
SOFT_MAX_CTX = int(os.environ.get("MOCK_SOFT_MAX_CTX", "0"))  # 置 N 则 prompt+max_tokens 超 N 时 200 + 正文 "prompt too long"（fastllm 系软超窗，验证测速端识别兜底）
SOFT_FIELD = os.environ.get("MOCK_SOFT_MAX_CTX_FIELD", "content")  # 软超窗占位串通道：content（默认）/reasoning（走 reasoning_content，复现思考模型形态）
SOFT_TEXT = os.environ.get("MOCK_SOFT_TEXT", "prompt too long")  # 软超窗占位串自定义（默认已知文案；自定义串验证测速端输出塌缩守卫的通用兜底）
TG_PATTERN = [float(x) for x in os.environ.get("MOCK_TG_PATTERN", "").split(",") if x.strip()]  # 逐请求交替 decode 速度 tok/s（循环取值，双峰复现）
CACHE = os.environ.get("MOCK_CACHE", "")  # 置 1 模拟前缀缓存：与已见 prompt 的公共前缀部分不计 prefill 耗时（验证 agent 连续任务链）
CACHE_NOREPORT = os.environ.get("MOCK_CACHE_NOREPORT", "")  # 置 1 则缓存生效但 usage 不回传命中字段（验证缓存迹象判别：TTFT 走平 → ≈差分估算）
CACHE_READ_S = float(os.environ.get("MOCK_CACHE_READ_TOK_S", "0"))  # >0 则缓存生效（机制同 MOCK_CACHE）但不回传命中字段，且命中前缀的读取 TTFT = 命中tokens/N 秒（N ≫ PP 模拟「缓存命中但读取慢」，走平判别失效、验证全量预期/佐证通道判别）
IGNORE_MT = os.environ.get("MOCK_IGNORE_MAX_TOKENS", "")  # 置 1 则忽略 max_tokens（只认 max_completion_tokens，opencode zen 行为；旧名请求输出跑飞 4× 上限）
REJECT_MCT = os.environ.get("MOCK_REJECT_MCT", "")  # 置 1 则带 max_completion_tokens 参数的请求 400（严格校验的老规范网关）
EARLY_STOP = int(os.environ.get("MOCK_EARLY_STOP", "0"))  # 置 N 则下一发 chat 请求一次性 early-stop（finish=stop、仅 N tokens、带真 usage，用后归零恢复正常；脚本化「首测异常、重测正常」，验证弃测重测全链路）
EARLY_STOP_LONG = int(os.environ.get("MOCK_EARLY_STOP_LONG", "0"))  # 置 N 则接下来 N 发「带指令材料」的 chat 请求（末条消息 >64 字符，agent 预热/基线超短任务不命中）逐发 early-stop（每次仅 4 tokens，实测 4/512 形态）、用后归零（agent 矩阵弃测重测专项：N=1 首测异常重测正常，N 大 持续异常按实留档）
ASR_PP = float(os.environ.get("MOCK_ASR_PP", "50"))   # 模拟 ASR 处理速度（音频秒/秒）
TTS_RATE = float(os.environ.get("MOCK_TTS_RATE", "4.5"))  # 模拟 TTS 合成语速（字/秒 → 音频时长）
TTS_XR = float(os.environ.get("MOCK_TTS_XR", "20"))   # 模拟 TTS 合成速度（×实时）
FLAKY_FAIL = int(os.environ.get("MOCK_FLAKY_FAIL", "0"))  # 置 N 则接下来 N 发 chat 请求一次性 500（模拟网关瞬时抖动，验证测速端兜底重试；脚本化用后归零）
HARD_FAIL_CTX = int(os.environ.get("MOCK_HARD_FAIL_CTX", "0"))  # 置 N 则 prompt+max_tokens 超 N 返回 500「CUDA out of memory」（未自报超窗的秒拒：报文不含超窗关键词，验证贴边档回退裁减）
_SR = 16000   # 合成/解析音频的采样率
LAST_CHAT_ROLES: list | None = None   # 最近一次 chat 请求的消息角色序列（e2e 断言回复模式用）
_REQ_SEQ = 0   # chat 请求计数（TG_PATTERN 轮转取值用）


def _wav_seconds(data: bytes) -> float | None:
    """stdlib wave 读 wav 时长（秒）；非 wav/损坏返回 None。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        return None


def _pcm_sine(seconds: float) -> bytes:
    """生成正弦 PCM16（220Hz + 谐波，16kHz 单声道）：1s 段循环拼接——TTS mock
    只需要合法音频载荷，段重复不影响客户端的时长解析。"""
    n = int(_SR * seconds)
    seg = array("h", bytes(2 * _SR))
    for k in range(_SR):
        ph = 2 * math.pi * 220 * k / _SR
        seg[k] = int(9000 * (0.7 * math.sin(ph) + 0.3 * math.sin(2 * ph)))
    return (seg.tobytes() * (n // _SR + 1))[: n * 2]


def _wav_bytes(seconds: float) -> bytes:
    """完整合法 wav（头 + PCM），时长精确到帧。"""
    pcm = _pcm_sine(seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(pcm)
    return buf.getvalue()


def _multipart_first_file(body: bytes, boundary: bytes) -> bytes:
    """极简 multipart 解析：取第一个带 filename 的字段内容（自测 mock 专用，
    不引 python-multipart 依赖；生产服务端有自己的 multipart 实现）。"""
    for part in body.split(b"--" + boundary):
        head, _, payload = part.partition(b"\r\n\r\n")
        if b"filename=" in head:
            return payload.rstrip(b"\r\n-")
    return b""

_SEEN: list[str] = []   # 已见 prompt 全文（前缀缓存匹配源），容量 cap 32


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

_PREFILL_SEM = asyncio.Semaphore(1)   # MOCK_SERIALIZE 的全局 prefill 串行闸

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def mock_marker(request, call_next):
    """所有响应带标记头，测速端据此识别 mock 运行（不落历史记录）。"""
    resp = await call_next(request)
    resp.headers["x-mock-server"] = "1"
    return resp


async def _models_payload():
    if MODELS_DELAY:
        await asyncio.sleep(MODELS_DELAY)   # 慢网关：拖住 RTT 基线测量
    return {"data": [{"id": "mock-llm-7b"}, {"id": "mock-llm-14b"}]}


@app.get("/v1/models")
async def models():
    return await _models_payload()


@app.get("/models")
async def models_noprefix():
    return await _models_payload()


async def _prefill_sleep(prompt_tokens: int, cache_read_s: float = 0.0):
    """prefill 耗时等待（CACHE_READ_S 置位时叠加命中前缀的缓存读取耗时）；
    MOCK_SERIALIZE=1 时全局串行——模拟真实服务端并发 prefill 串行化，并发
    请求的 decode 区间因此先后发生、不再重叠。"""
    total = prompt_tokens / PP + cache_read_s
    if SERIALIZE:
        async with _PREFILL_SEM:
            await asyncio.sleep(total)
    else:
        await asyncio.sleep(total)


async def _chat_impl(req: Request):
    global LAST_CHAT_ROLES, _REQ_SEQ, EARLY_STOP, EARLY_STOP_LONG, FLAKY_FAIL
    body = await req.json()
    if FLAKY_FAIL > 0:
        # 一次性脚本钩子（瞬时失败兜底重试 e2e 专项）：本请求直接 HTTP 500、
        # 不进任何计时/角色记录，用后归零恢复正常——脚本化「首发抖动、
        # 兜底重试成功」，验证测速端 5xx 重试后 reqs 留 retried 痕
        FLAKY_FAIL -= 1
        return JSONResponse({"error": {"message":
            "upstream temporarily unavailable (flaky)"}}, status_code=500)
    tg = TG   # 本请求 decode 速度（TG_PATTERN 置位时逐请求轮转交替）
    if TG_PATTERN:
        tg = TG_PATTERN[_REQ_SEQ % len(TG_PATTERN)]
        _REQ_SEQ += 1
    # 记录最近一次 chat 请求的消息角色序列：e2e 断言回复模式（自由回答
    # =[system,user] / 模板续写末尾带 assistant 预填）用，不影响响应行为
    LAST_CHAT_ROLES = [m.get("role") for m in body.get("messages", [])]
    if REJECT_THINKING and "thinking" in body:
        return JSONResponse({"error": {"message": "unknown parameter: thinking"}},
                            status_code=400)
    if REJECT_TEMPERATURE and "temperature" in body:
        return JSONResponse({"error": {"message":
            "invalid temperature: only 0.6 is allowed for this model"}},
            status_code=400)
    if REJECT_MCT and "max_completion_tokens" in body:
        return JSONResponse({"error": {"message":
            "Unrecognized request argument supplied: max_completion_tokens"}},
            status_code=400)
    def _text_of(content) -> str:
        """消息 content 兼容字符串与多模态 parts 列表（VLM 读图，ADR-0020）：
        只计 text 部分；image_url 部分由服务端按分辨率计入，mock 不计。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content
                           if isinstance(p, dict) and p.get("type") == "text")
        return ""
    text = "".join(_text_of(m.get("content", "")) for m in body.get("messages", []))
    # 末条消息文本（EARLY_STOP_LONG 的命中判据：agent 测量请求带指令材料、
    # 预热/基线的超短任务 "Reply with OK." 不命中）
    _msgs = body.get("messages") or [{}]
    last_text = _text_of(_msgs[-1].get("content", ""))
    prompt_tokens = max(1, int(len(text) / CPT))
    max_tokens = int(body.get("max_tokens") or 256)
    if IGNORE_MT:
        # opencode zen 行为：旧名 max_tokens 静默忽略、输出跑飞到 4× 上限；
        # 新名 max_completion_tokens 正常执行截断
        if "max_completion_tokens" in body:
            max_tokens = int(body["max_completion_tokens"])
        else:
            max_tokens *= 4
    if MAX_CTX and prompt_tokens + max_tokens > MAX_CTX:
        # OpenAI 风格超窗报文（code=context_length_exceeded）
        return JSONResponse({"error": {"message":
            f"This model's maximum context length is {MAX_CTX} tokens. "
            f"However, you requested {prompt_tokens + max_tokens} tokens "
            f"({prompt_tokens} in the messages, {max_tokens} in the completion).",
            "code": "context_length_exceeded"}}, status_code=400)
    if HARD_FAIL_CTX and prompt_tokens + max_tokens > HARD_FAIL_CTX:
        # 未自报超窗的秒拒（fastllm 显存/KV 不足形态）：500 报文刻意不含
        # context/exceed 等超窗关键词，测速端 _is_ctx_overflow 不识别——
        # 验证贴边档回退裁减（估算贴窗军备 + CTX_RETRY_KEEP 删减阶梯）
        return JSONResponse({"error": {"message":
            "CUDA out of memory: failed to allocate KV cache block"}},
            status_code=500)
    if SOFT_MAX_CTX and prompt_tokens + max_tokens > SOFT_MAX_CTX:
        # fastllm 系软超窗：不报 4xx，HTTP 200 流式返回、正文替换为占位串
        # "prompt too long"、finish=stop、带 usage（与真实 fastllm 行为一致）；
        # 占位串通道/文案可配（reasoning 通道复现思考模型形态，自定义文案
        # 验证测速端输出塌缩守卫的通用兜底）
        soft_delta = ({"reasoning_content": SOFT_TEXT} if SOFT_FIELD == "reasoning"
                      else {"content": SOFT_TEXT})

        async def gen_soft():
            yield "data: " + json.dumps({"choices":
                [{"delta": soft_delta, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices":
                [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": 1}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen_soft(), media_type="text/event-stream")
    hit_tokens = 0
    cache_read_s = 0.0
    if CACHE or CACHE_NOREPORT or CACHE_READ_S > 0:
        # 前缀缓存：与任一已见 prompt 的最长公共前缀即命中（append-only
        # 增长的会话链命中上一轮全文），命中部分不计 prefill 耗时
        hit_chars = max((_common_prefix_len(text, s) for s in _SEEN), default=0)
        hit_tokens = int(hit_chars / CPT)
        if CACHE_READ_S > 0:
            # 缓存命中但读取慢：命中 tokens ÷ N 秒的读取耗时（N ≫ PP 时
            # 读取远慢于全量 prefill，TTFT 随缓存规模线性增长不走平）
            cache_read_s = hit_tokens / CACHE_READ_S
        _SEEN.append(text)
        del _SEEN[:-32]
    prefill_tokens = max(prompt_tokens - hit_tokens, 1)
    n_early = 0
    if EARLY_STOP > 0:
        n_early = EARLY_STOP
        EARLY_STOP = 0
    elif EARLY_STOP_LONG > 0 and len(last_text) > 64:
        # agent 测量批专用旋钮：逐发递减计数（N=1 首测异常重测正常，N 大
        # 持续异常按实留档），预热/基线超短任务不命中照常满预算输出；
        # 每次仅吐 4 tokens（实测「材料截断」短答 4/512 形态）
        n_early = 4
        EARLY_STOP_LONG -= 1
    if n_early > 0:
        # 一次性脚本钩子（ADR-0032 弃测重测 e2e 专项）：本请求照常 prefill，
        # 正文只吐 N 个 token 即 finish=stop、usage 回传真值 completion_tokens=N
        # ——脚本化「首测提前停止、后续正常」的异常形态（echo 续写口径下
        # 的 early_stop 实测案例：4K 档仅输出 4/512 tokens）。用后归零：
        # 后续请求（弃测重测批）恢复正常满预算输出，默认 0 时此分支死代码，
        # 不影响任何既有用例

        async def gen_early():
            await _prefill_sleep(prefill_tokens, cache_read_s)
            left = n_early
            while left > 0:
                k = min(ACCEPT, left)   # 交付形态与正常流同构（逐事件 k token）
                await asyncio.sleep(k / tg)
                yield "data: " + json.dumps({"choices":
                    [{"delta": {"content": "测" * (k * CH)}}]}) + "\n\n"
                left -= k
            final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": prompt_tokens,
                               "completion_tokens": n_early}}
            yield "data: " + json.dumps(final) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen_early(), media_type="text/event-stream")

    async def gen():
        if DELAY_HDR:
            pass   # prefill 等待已在端点返回前完成（响应头随首个内容块才发出）
        elif KA > 0:   # prefill 期高频注释心跳：淹没读行超时（回归：估值帧曾因此全程停摆）
            t_wait, t = prefill_tokens / PP + cache_read_s, 0.0
            while t < t_wait:
                await asyncio.sleep(KA)
                t += KA
                yield ": keepalive\n\n"
        else:
            await _prefill_sleep(prefill_tokens, cache_read_s)   # prefill 耗时（缓存命中部分不计；CACHE_READ_S 时叠加读取耗时）
        n_content = 0 if NO_CONTENT else max_tokens
        for _ in range(REASON):
            await asyncio.sleep(1.0 / tg)
            yield "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "思"}}]}) + "\n\n"
        burst = min(BURST, n_content)
        if burst:   # 首批 token 一次性吐出（不占 decode 时间轴上的间隔）
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "测" * (burst * CH)}}]}) + "\n\n"
        left = n_content - burst
        mid = left // 2   # 停滞定位：以剩余 token 计数的中点为准（事件步进下落在某事件的 token 跨度内）
        i = 0
        pending: list[str] = []   # 亚毫秒间隔事件的攒批缓冲（见下）
        while left > 0:
            k = min(ACCEPT, left)                        # 每事件 k 个 token（末事件携带余数），间隔 k/tg → 有效速率仍 ≈ tg
            payload = "data: " + json.dumps({"choices": [{"delta": {"content": "测" * (k * CH)}}]}) + "\n\n"
            iv = k / tg
            # 低于事件循环计时器分辨率（~1ms）的间隔不进定时器堆：sleep(80µs) 会被
            # 放大到 ~1ms/事件且随机器抖动，客户端到达间隔随之越过 1ms 冲刷线
            # （decode_burst 冲刷用例曾因此确定性失败）。改攒批到同一次写——同帧
            # 多事件是合法 SSE，也更贴近「网关缓冲合并冲刷」的真实形态
            if iv >= 0.001:
                if pending:
                    yield "".join(pending); pending = []
                await asyncio.sleep(iv)
            if STALL and i <= mid < i + k:
                if pending:
                    yield "".join(pending); pending = []
                await asyncio.sleep(STALL)               # 中段一次停滞（空窗诊断）
            if iv >= 0.001:
                yield payload
            else:
                pending.append(payload)
            i += k
            left -= k
        if pending:
            yield "".join(pending)
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        if not NO_USAGE:
            final["usage"] = {"prompt_tokens": prompt_tokens,
                              "completion_tokens": REASON + n_content}
            if CACHE:
                final["usage"]["prompt_cache_hit_tokens"] = hit_tokens
                final["usage"]["prompt_cache_miss_tokens"] = prompt_tokens - hit_tokens
            # CACHE_NOREPORT：缓存生效但不下发命中字段（模拟剥离 usage 扩展的网关）
        yield "data: " + json.dumps(final) + "\n\n"
        yield "data: [DONE]\n\n"

    if DELAY_HDR:   # prefill 等待放在端点返回前：响应头随首个内容块才发出
        await _prefill_sleep(prefill_tokens, cache_read_s)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_v1(req: Request):
    if NO_V1:
        # DeepSeek 风格：chat 端点不带 /v1 前缀（对齐 FastAPI 默认 404 报文）。
        # /v1/models 保留可达：RTT 基线测量若先在 /v1/models 撞 404 会静默
        # 锁定前缀，chat 路径的 404 回退与提示就永远走不到，e2e 无法覆盖
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return await _chat_impl(req)


@app.post("/chat/completions")
async def chat_noprefix(req: Request):
    return await _chat_impl(req)


@app.post("/audio/transcriptions")
@app.post("/v1/audio/transcriptions")
async def transcriptions(req: Request):
    """OpenAI 兼容语音转写：按音频时长/MOCK_ASR_PP 模拟处理耗时。
    MOCK_MAX_CTX 置位时按 ~50 tokens/音频秒 判超窗（验证引擎的媒体超窗保护）。"""
    ctype = req.headers.get("content-type", "")
    body = await req.body()
    wav = b""
    if "boundary=" in ctype:
        wav = _multipart_first_file(body, ctype.split("boundary=")[-1].encode())
    seconds = _wav_seconds(wav)
    if seconds is None:
        if wav:
            seconds = len(wav) / 32000   # 非 wav 按 16k 16bit 粗估
        else:
            return JSONResponse({"error": {"message":
                "missing audio file (multipart field 'file')"}}, status_code=400)
    prompt_tokens = int(seconds * 50)
    if MAX_CTX and prompt_tokens > MAX_CTX:
        return JSONResponse({"error": {"message":
            f"This model's maximum context length is {MAX_CTX} tokens. "
            f"However, you requested {prompt_tokens} tokens in the audio.",
            "code": "context_length_exceeded"}}, status_code=400)
    await asyncio.sleep(seconds / ASR_PP)
    return {"text": "模拟识别结果。" * max(1, min(60, int(seconds)))}


@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def speech(req: Request):
    """OpenAI 兼容语音合成：音频时长 = 文本字符/MOCK_TTS_RATE，按 MOCK_TTS_XR
    倍实时节奏流式吐 wav chunk（首 chunk 即刻到达，TTFA ≈ 0）。"""
    body = await req.json()
    text = body.get("input", "")
    seconds = max(0.05, len(text) / TTS_RATE)
    data = _wav_bytes(seconds)
    hdr_end = data.find(b"data") + 8   # RIFF 头与 PCM 分界（头内长度已知且正确）
    hdr, pcm = data[:hdr_end], data[hdr_end:]
    chunk_bytes = _SR // 2             # 0.5s 音频一块

    async def gen():
        first = True
        for off in range(0, len(pcm), chunk_bytes):
            piece = pcm[off:off + chunk_bytes]
            if first:
                yield hdr + piece   # 首个 chunk 携带完整 RIFF 头
                first = False
            else:
                yield piece
            await asyncio.sleep(0.5 / TTS_XR)

    return StreamingResponse(gen(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8901)))
