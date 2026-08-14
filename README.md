# LLM-SPEED · LLM 测速台

对挂在统一网关（LiteLLM 等，OpenAI 兼容协议）后面的多个本地 LLM 服务做自动化测速，
**不关心底层推理框架**（llama.cpp / vLLM / fastllm / ...），只看 API 行为。

一个单页面服务：选择模型与测试矩阵 → 实时展示测速过程 → 折线图汇总 → 生成一张测速卡片 PNG。

## 功能

- **多 Provider**：config.json 定义任意多个网关（云端 API / LiteLLM / llama.cpp server），页面一键切换，key 全部留在服务端 `.env`
- **测试矩阵**：多模型逐一测试 × 场景（创意写作 / 代码生成）× 上下文长度（0K ~ 1M 可选，支持自定义档位如 192K）× 并发（1 / 2）
- **指标**：TTFT、Prefill 速度（tok/s，含扣网络 RTT 的净口径）、Decode 单请求速度（tok/s）、双并发聚合总吞吐（tok/s）
- **实时展示**：流式解码中每 0.4s 推送一次当前速度（SSE）；测速开始自动收起配置面板，专注测试本身
- **折线图**：上下文长度（4K/8K/16K 类别轴）为横轴，prefill / decode 分别成图，按并发分线
- **测速卡片**：合并全部场景——各场景 prefill/decode 速度范围与峰值上下文、单并发 vs 双并发对比、逐场景折线图，一键导出 PNG
- **历史记录**：每次结果自动保存 `results/<run_id>.json`，页面可载入复看

## 快速开始

### 一键启动（推荐）

```bash
./start.sh        # Linux / macOS
start.bat         # Windows
```

自动完成：创建 `.venv` 虚拟环境 → 安装 `requirements.txt` → 缺失时从
`.env.example` 生成 `.env` → 启动服务（缺省 http://127.0.0.1:8501）。
首次启动后先在页面「Provider / 部署配置」卡片里填好各 provider 的 API Key
（或手动编辑 .env），再点「保存配置」即时生效。

### 手动启动

```bash
pip install -r requirements.txt
cp .env.example .env    # 填入各 provider 的 API Key（API_KEY_<名称大写>）

# （可选）先用假网关验证链路，无需真实模型
MOCK_PP=1200 MOCK_TG=60 python mock_server.py     # 模拟 prefill 1200 tok/s、decode 60 tok/s
MOCK_NO_V1=1 python mock_server.py                # DeepSeek 风格：不带 /v1 前缀

python server.py                                   # 默认 http://127.0.0.1:8501
```

> 服务端**缺省只绑定本机回环**（全端点无鉴权：绑 `0.0.0.0` 会让局域网内任何人
> 可触发对云端付费网关的测速并删除历史）。确需局域网访问时 `HOST=0.0.0.0
> python server.py`，并自行评估风险。

### 配置 Provider

两种等价方式：

1. **页面配置（推荐）**：浏览器打开后，「Provider / 部署配置」卡片里直接
   增删 provider、编辑网关地址、本地部署标记与部署环境
   （量化/硬件/框架/核心参数/max_ctx），并可写入或清除 API Key
   （Key 只写入服务端 `.env`，页面不回显）。「保存配置」即时生效，无需重启。
2. **手改文件**：`config.json` 定义 provider 列表（名称 + 网关地址），
   `.env` 提供对应的 key：

```json
{
  "providers": [
    {"name": "deepseek", "label": "DeepSeek 云端", "gateway_url": "https://api.deepseek.com"},
    {"name": "local", "label": "本地网关（LiteLLM）", "gateway_url": "http://127.0.0.1:4000"}
  ]
}
```

```bash
# .env：key 命名 = API_KEY_ + provider 名称大写
API_KEY_DEEPSEEK=sk-xxx
API_KEY_LOCAL=            # 本地网关无鉴权可留空，会回退全局 API_KEY
```

浏览器打开 `http://localhost:8501`：

1. 选择 Provider →「刷新模型列表」
2. 勾选模型 / 场景 / 上下文长度 / 并发，按需填卡片信息（硬件、框架、核心参数，选填）
3. 「开始测速」（配置面板自动收起），顶部 KPI 与实时表格滚动显示速度
4. 结束后折线图自动绘制；顶栏切换模型/场景查看，「生成卡片」→「下载 PNG」

**并行测多个 provider**：开多个浏览器标签页，各选各的 provider 同时开跑即可——
服务端按 run 隔离（事件流、存档互不干扰）。注意同一物理后端（同一台 llama.cpp）
不要并行测，硬件互相挤占会让数据失真；只有跨独立后端（本地 + 云端）并行才有意义

## 测量方法

| 指标 | 定义 |
|---|---|
| Prefill 速度 | 实际 prompt tokens ÷ TTFT（首 token 延迟） |
| Decode 速度 | 生成 tokens ÷ (末 token 时间 − 首 token 时间)，单请求均值 |
| 双并发总吞吐 | 两请求总输出 tokens ÷ (首个 token → 末个 token 的时间窗) |

- **上下文构造**：场景语料池整段乱序拼接撑长度（创意=公版文学文本《红楼梦》，
  见 `corpus/creative/`，缺失回退内置散文池；代码=**真实项目源码**，
  vendored llama.cpp b9934，见 `corpus/code/`），避免模型照抄重复语料导致投机采样命中率
  注水；每次请求注入随机 nonce 编号，**防止 prefix cache 命中导致 prefill 虚高**
- **前置探测校准**：每个模型×场景进矩阵前先发一次短请求（兼服务端预热），用真实
  prompt_tokens 反推字符/token 系数，首个测试点即按实测系数构造，目标长度即实际长度
- **RTT 净口径**：run 开始实测网络往返基线，prefill 净速度 = prompt tokens ÷ (TTFT − RTT)；
  云端短上下文毛口径会被固定开销摊薄（4K 档约低估一倍），表格/图表/卡片优先展示净值
- **复测取中位**：页面可配每点重复 1~5 次，标量取中位——云端单点读数有 ±20% 级调度
  抖动，关键点建议 3 次
- **端点路径自动适配**：默认试 `/v1/chat/completions`，404 时自动回退无前缀的
  `/chat/completions`（DeepSeek 官方 API 不带 `/v1`；LiteLLM/vLLM 带）。模型列表拉取同理
- **token 计数**：优先取流式 `usage`（请求时带 `stream_options.include_usage`，
  vLLM / llama.cpp server 新版 / LiteLLM 透传均支持）；网关不返回时按「字符数 ÷ chars_per_token」估算
  （输入默认 1.8，可在页面调整）
- **测试顺序**：模型 → 场景 → 上下文(升序) → 并发，逐点执行，出错不中断整体
- **思考模式**：DeepSeek V4 思考默认开启，思考流（`reasoning_content`）会占用 max_tokens、
  甚至导致正文为空。页面可选：禁用（默认，纯测速推荐）/ 启用（思考流计入测速）/ 不发参数；
  启用时思考 token 同样被计数，不会出现静默空数据
- **输出截断**：`max_tokens` 达到上限时服务端**立即停止生成**（finish_reason=length），
  不会把答案答完；测速只用已流出的 token 窗口，截断不影响 decode 速度有效性，
  结果表「结束」列会标注 ✂ 截断

## 注意事项

1. **长上下文耗时**：256K 上下文的 prefill 可能要几十秒到几分钟，请把「单请求超时」调大；
   全矩阵（2 场景 × 10 档上下文 × 2 并发）×每模型，需要预留较长时间，可按需裁剪档位
2. **上下文上限**：在 provider 的 `deployments[]` 里配 `max_ctx`（如 llama.cpp 的 `-c` 值），
   连同输出预算超窗的档位自动跳过；未配置时超窗请求会报 400（该点记 ❌，不影响其他点）
3. **网关超时**：测速端超时默认 1200s 足够，但**网关自身**等后端的超时也要够大——
   长上下文 prefill（如 192K 档在低配硬件上超 10 分钟）超过 LiteLLM `request_timeout`
   或 nginx `proxy_read_timeout` 时会收到 504，而模型服务仍在后台继续计算
4. **显存**：长上下文 KV cache 占用大，vLLM 注意 `--gpu-memory-utilization`，llama.cpp 注意 `-c` 与 `-ngl`
5. **decode 输出长度**：`max_tokens` 默认 1024；调太小会让 decode 测量窗口变短、噪声变大
6. prefill 展示口径默认已扣除实测网络 RTT（明细表悬浮可见毛值）；本地部署 RTT < 1ms 可忽略

## 测试

```bash
python -m unittest discover -s tests -v   # 引擎单测 + mock 端到端（需本机 python 环境）
python -m pytest tests/ -q                # 等价跑法（pip install -r requirements-dev.txt）
bash tests/check_frontend_js.sh           # 前端内联 JS 语法检查（需 node）
```

端到端用例在进程内拉起 mock 网关验证全链路口径（prefill 净口径 ≈ MOCK_PP、
双并发总吞吐 ≈ 2× 单请求、cfg 事件脱敏首发、mock 运行不落盘）。

## 文件结构

```
server.py        FastAPI 服务端（provider 解析 + 测速任务 + SSE + 历史 + 配置写入）
bench.py         测速引擎（prompt 构造、流式测量、测试矩阵调度）
corpus/code/     代码场景真实语料（vendored llama.cpp b9934 源码，MIT）
corpus/creative/ 创意场景语料（vendored 公版《红楼梦》，Project Gutenberg License）
static/index.html 单页前端（ECharts 折线图 + html2canvas 卡片导出 + 配置编辑）
mock_server.py   假 OpenAI 网关（自测用）
tests/           回归测试（单元 + mock 端到端 + 配置接口 + JS 语法检查）
start.sh / start.bat  一键启动脚本（Linux/macOS / Windows）
config.json      provider 列表 / 部署映射（页面可编辑）
.env             各 provider 的 API Key（不入库，页面可写入）
results/         历次测速结果 JSON
```
