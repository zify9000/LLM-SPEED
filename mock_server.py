"""假 OpenAI 兼容网关，用于验证测速链路（无需真实模型）。

模拟特性：
- prefill 速度 ≈ MOCK_PP tok/s（TTFT 随 prompt 长度线性增长）
- decode 速度 ≈ MOCK_TG tok/s（流式逐 token 输出）
- 返回 usage（模拟 stream_options.include_usage 行为）

启动：MOCK_PP=1200 MOCK_TG=60 python mock_server.py
DeepSeek 风格（不带 /v1 前缀）：MOCK_NO_V1=1 python mock_server.py
模拟分词偏差（验证输入系数校准）：MOCK_CPT=3.9；模拟严格校验网关：MOCK_REJECT_THINKING=1
模拟 decode 起步突发（首批 token 一次性吐出）：MOCK_BURST=15
"""
import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PP = float(os.environ.get("MOCK_PP", "1200"))    # 模拟 prefill 速度 tok/s
TG = float(os.environ.get("MOCK_TG", "60"))     # 模拟单请求 decode 速度 tok/s
CPT = float(os.environ.get("MOCK_CPT", "1.8"))  # 字符/token，需与测速端估算一致才准
REASON = int(os.environ.get("MOCK_REASON", "0"))   # 正文前先输出多少个 reasoning_content token
NO_CONTENT = os.environ.get("MOCK_NO_CONTENT", "")  # 置 1 则只输出思考、无正文
REJECT_THINKING = os.environ.get("MOCK_REJECT_THINKING", "")  # 置 1 则带 thinking 参数的请求 400
REJECT_TEMPERATURE = os.environ.get("MOCK_REJECT_TEMPERATURE", "")  # 置 1 则带 temperature 参数的请求 400
BURST = int(os.environ.get("MOCK_BURST", "0"))  # 首个内容事件一次性携带的 token 数（模拟服务端缓冲/投机采样突发）

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def mock_marker(request, call_next):
    """所有响应带标记头，测速端据此识别 mock 运行（不落历史记录）。"""
    resp = await call_next(request)
    resp.headers["x-mock-server"] = "1"
    return resp


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "mock-llm-7b"}, {"id": "mock-llm-14b"}]}


async def _chat_impl(req: Request):
    body = await req.json()
    if REJECT_THINKING and "thinking" in body:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": {"message": "unknown parameter: thinking"}},
                            status_code=400)
    if REJECT_TEMPERATURE and "temperature" in body:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": {"message":
            "invalid temperature: only 0.6 is allowed for this model"}},
            status_code=400)
    text = "".join(m.get("content", "") for m in body.get("messages", []))
    prompt_tokens = max(1, int(len(text) / CPT))
    max_tokens = int(body.get("max_tokens") or 256)

    async def gen():
        await asyncio.sleep(prompt_tokens / PP)          # prefill 耗时
        n_content = 0 if NO_CONTENT else max_tokens
        for _ in range(REASON):
            await asyncio.sleep(1.0 / TG)
            yield "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "思"}}]}) + "\n\n"
        burst = min(BURST, n_content)
        if burst:   # 首批 token 一次性吐出（不占 decode 时间轴上的间隔）
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "测" * burst}}]}) + "\n\n"
        for _ in range(n_content - burst):
            await asyncio.sleep(1.0 / TG)                # decode 逐 token
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "测"}}]}) + "\n\n"
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt_tokens,
                           "completion_tokens": REASON + n_content}}
        yield "data: " + json.dumps(final) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_v1(req: Request):
    return await _chat_impl(req)


if os.environ.get("MOCK_NO_V1"):
    # DeepSeek 风格：只提供无前缀端点，/v1/* 一律 404（FastAPI 默认行为）
    app.router.routes = [r for r in app.router.routes
                         if getattr(r, "path", "") not in ("/v1/models", "/v1/chat/completions")]

    @app.get("/models")
    async def models_noprefix():
        return await models()

    @app.post("/chat/completions")
    async def chat_noprefix(req: Request):
        return await _chat_impl(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8901)))
