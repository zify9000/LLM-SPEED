<div align="center">

# ⚡ LLM-SPEED · LLM 测速台

一个 LLM 模型测速台，主打 **数据真实 · 过程可见 · 结论可分享**

**⚡ prefill / decode 分离计量** · **📏 0K~1M 上下文 × 并发** · **🎭 创意写作 · 代码生成 · Agent 连续任务链**

</div>

## 💡 初心

从 vLLM 到 llama.cpp，从 Q4 到 Q8，换卡、调参、比量化，却始终缺一个工具横向回答：
不同上下文长度、不同并发、不同使用场景（创意写作 / 代码生成 / Agent 调用）下，
每个模型的 prefill / decode 到底是多少，与云端 API 的差距是多少。

## ✨ 特点

| 🧮 口径透明 | 📡 逐帧实测 | 🎭 多场景模拟 |
| --- | --- | --- |
| prefill 扣除实测网络往返、decode 按 2 秒滑窗差分并以服务端 usage 回写校准；缓存命中区分服务端真值与链内估算；网络卡顿、网关缓冲冲刷造成的虚高读数一律显式标记 | prefill 等待期逐秒给出估值、decode 逐帧刷新；跑完自动生成折线图和测速卡片 PNG，一键分享 | 创意写作读公版《红楼梦》、代码生成读 llama.cpp 真实源码；Agent 调用跑多轮连续任务链 |

## 🖼️ 测速卡片示例

| DeepSeek 云端 · 0K~256K | Kimi 云端 · 0K~128K |
| --- | --- |
| ![deepseek-v4-pro 测速卡片](docs/images/speed-card-deepseek-v4-pro.png) | ![k3-256k 测速卡片](docs/images/speed-card-k3-256k.png) |

## 🚀 三分钟上手

**1️⃣ 启动**

```bash
./start.sh        # Linux / macOS；Windows 用 start.bat
```

自动建虚拟环境、装依赖、生成 `.env` 与 `config.json`（从模板）；浏览器打开 http://127.0.0.1:8501。

**2️⃣ 配置** — 顶栏「Provider 管理」抽屉里填好网关与 Key（页面写回 `.env`，永不回显）→「刷新模型列表」勾选模型

**3️⃣ 开测** — 「开始测速」：实时表逐帧刷新 → 完成自动生成测速卡片 → 「下载 PNG」

## 🗺️ 数据流

```mermaid
flowchart LR
  A["🖥 单页前端<br/>实时表 · 曲线 · 卡片"] <-->|"SSE 事件流"| B["⚙ FastAPI<br/>测速任务 · 历史存档"]
  B -->|"OpenAI 兼容流式请求"| C["🔗 你的网关<br/>llama.cpp · vLLM · LiteLLM · 云端"]
  B --> D[("💾 results/ JSON 存档")]
  B --> E["🔐 config.json + .env<br/>凭据不出服务端"]
```

## 📁 项目结构

```
start.sh / start.bat  一键启动（建虚拟环境、装依赖、生成配置、起服务）
server.py          FastAPI 服务端（provider 解析 + 测速任务 + SSE + 历史 + 配置写入）
bench.py           测速引擎（prompt 构造、流式计时、矩阵调度）
static/index.html  单页前端（ECharts 图表 + html2canvas 卡片 + 配置编辑）
mock_server.py     假 OpenAI 网关（自测）
corpus/            真实语料（code: llama.cpp 源码 / creative: 公版《红楼梦》/ agent: SWE-agent 真实执行轨迹）
scripts/           语料构建脚本（agent 轨迹提取，仅构建期用）
docs/images/       测速卡片示例图（README 引用）
tests/             回归测试
config.json / .env  provider 配置与凭据（不入库）
results/           历次测速结果 JSON
```

## 📄 许可证

本项目以 [Apache License 2.0](LICENSE) 发布。`corpus/` 内为第三方语料：llama.cpp 源码（MIT）、公版《红楼梦》（Project Gutenberg）、SWE-agent 轨迹（CC-BY-4.0），各循其自带许可证。

