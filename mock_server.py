"""假 OpenAI 兼容网关，用于验证测速链路（无需真实模型）。

模拟特性：
- prefill 速度 ≈ MOCK_PP tok/s（TTFT 随 prompt 长度线性增长）
- decode 速度 ≈ MOCK_TG tok/s（流式逐 token 输出）
- 返回 usage（模拟 stream_options.include_usage 行为）

启动：MOCK_PP=1200 MOCK_TG=60 python mock_server.py
DeepSeek 风格（chat 端点不带 /v1 前缀）：MOCK_NO_V1=1 python mock_server.py
模拟分词偏差（验证输入系数校准）：MOCK_CPT=3.9；模拟严格校验网关：MOCK_REJECT_THINKING=1
模拟 decode 起步突发（首批 token 一次性吐出）：MOCK_BURST=15
模拟 prefill 期 SSE 注释心跳（秒间隔，如 0.3）：MOCK_KEEPALIVE=0.3
模拟响应头延迟到首个内容 token（prefill 期间无任何字节，LiteLLM→vLLM 链路行为）：MOCK_DELAY_HEADERS=1
模拟 decode 中段一次停滞（秒，验证空窗诊断）：MOCK_STALL=4
模拟流末不发 usage 帧（测速端走估算兜底）：MOCK_NO_USAGE=1
模拟流出 N 个 decode token 后裸断连（不发 [DONE]，直接关流）：MOCK_DIE_AT=5
模拟服务端并发 prefill 串行化（全局排队）：MOCK_SERIALIZE=1
模拟 /models 慢响应（秒，验证 RTT 基线期停止）：MOCK_MODELS_DELAY=5
"""
import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

PP = float(os.environ.get("MOCK_PP", "1200"))    # 模拟 prefill 速度 tok/s
TG = float(os.environ.get("MOCK_TG", "60"))     # 模拟单请求 decode 速度 tok/s
CPT = float(os.environ.get("MOCK_CPT", "1.8"))  # 字符/token，需与测速端估算一致才准
REASON = int(os.environ.get("MOCK_REASON", "0"))   # 正文前先输出多少个 reasoning_content token
NO_CONTENT = os.environ.get("MOCK_NO_CONTENT", "")  # 置 1 则只输出思考、无正文
REJECT_THINKING = os.environ.get("MOCK_REJECT_THINKING", "")  # 置 1 则带 thinking 参数的请求 400
REJECT_TEMPERATURE = os.environ.get("MOCK_REJECT_TEMPERATURE", "")  # 置 1 则带 temperature 参数的请求 400
BURST = int(os.environ.get("MOCK_BURST", "0"))  # 首个内容事件一次性携带的 token 数（模拟服务端缓冲/投机采样突发）
KA = float(os.environ.get("MOCK_KEEPALIVE", "0"))  # prefill 期间 SSE 注释心跳间隔秒（0=关闭；模拟 vLLM/代理类网关的高频 keepalive）
DELAY_HDR = os.environ.get("MOCK_DELAY_HEADERS", "")  # 置 1 则响应头延迟到首个内容 token（prefill 期间无任何字节）
STALL = float(os.environ.get("MOCK_STALL", "0"))  # decode 中段插入一次停滞秒数（模拟传输/调度空窗）
NO_V1 = os.environ.get("MOCK_NO_V1", "")  # 置 1 则 /v1/chat/completions 404（chat 端点无前缀，DeepSeek 风格）
NO_USAGE = os.environ.get("MOCK_NO_USAGE", "")  # 置 1 则流末不发 usage 帧（测速端 usage_real=False，走估算兜底）
DIE_AT = int(os.environ.get("MOCK_DIE_AT", "0"))  # 流出 N 个 decode token 后裸断连（不发 usage/[DONE]，直接关流）
SERIALIZE = os.environ.get("MOCK_SERIALIZE", "")  # 置 1 则 prefill 全局串行（模拟服务端并发 prefill 串行化）
MODELS_DELAY = float(os.environ.get("MOCK_MODELS_DELAY", "0"))  # /models 响应前 sleep 秒数（模拟 RTT 基线期慢网关）

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


async def _prefill_sleep(prompt_tokens: int):
    """prefill 耗时等待；MOCK_SERIALIZE=1 时全局串行——模拟真实服务端并发
    prefill 串行化，并发请求的 decode 区间因此先后发生、不再重叠。"""
    if SERIALIZE:
        async with _PREFILL_SEM:
            await asyncio.sleep(prompt_tokens / PP)
    else:
        await asyncio.sleep(prompt_tokens / PP)


async def _chat_impl(req: Request):
    body = await req.json()
    if REJECT_THINKING and "thinking" in body:
        return JSONResponse({"error": {"message": "unknown parameter: thinking"}},
                            status_code=400)
    if REJECT_TEMPERATURE and "temperature" in body:
        return JSONResponse({"error": {"message":
            "invalid temperature: only 0.6 is allowed for this model"}},
            status_code=400)
    text = "".join(m.get("content", "") for m in body.get("messages", []))
    prompt_tokens = max(1, int(len(text) / CPT))
    max_tokens = int(body.get("max_tokens") or 256)

    async def gen():
        if DELAY_HDR:
            pass   # prefill 等待已在端点返回前完成（响应头随首个内容块才发出）
        elif KA > 0:   # prefill 期高频注释心跳：淹没读行超时（回归：估值帧曾因此全程停摆）
            t_wait, t = prompt_tokens / PP, 0.0
            while t < t_wait:
                await asyncio.sleep(KA)
                t += KA
                yield ": keepalive\n\n"
        else:
            await _prefill_sleep(prompt_tokens)          # prefill 耗时
        n_content = 0 if NO_CONTENT else max_tokens
        for _ in range(REASON):
            await asyncio.sleep(1.0 / TG)
            yield "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "思"}}]}) + "\n\n"
        burst = min(BURST, n_content)
        n_out = 0   # 已流出的 decode token 数（MOCK_DIE_AT 断点计数）
        if burst:   # 首批 token 一次性吐出（不占 decode 时间轴上的间隔）
            n_out += burst
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "测" * burst}}]}) + "\n\n"
        for i in range(n_content - burst):
            if DIE_AT and n_out >= DIE_AT:
                # 裸断连：不发 usage/[DONE]，生成器抛错让 uvicorn 直接断 TCP，
                # 测速端读到不完整流 → 请求级 err（模拟服务端/链路中途崩断）
                raise RuntimeError("mock: simulated connection drop")
            await asyncio.sleep(1.0 / TG)                # decode 逐 token
            if STALL and i == (n_content - burst) // 2:
                await asyncio.sleep(STALL)               # 中段一次停滞（空窗诊断）
            n_out += 1
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "测"}}]}) + "\n\n"
        if DIE_AT and n_out >= DIE_AT:   # 断点恰在最后一个 token（usage 帧之前）
            raise RuntimeError("mock: simulated connection drop")
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        if not NO_USAGE:
            final["usage"] = {"prompt_tokens": prompt_tokens,
                              "completion_tokens": REASON + n_content}
        yield "data: " + json.dumps(final) + "\n\n"
        yield "data: [DONE]\n\n"

    if DELAY_HDR:   # prefill 等待放在端点返回前：响应头随首个内容块才发出
        await _prefill_sleep(prompt_tokens)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8901)))
