# LLM-SPEED · LLM 测速台

一个私域模型测速台，主打**数据真实、过程可见、结论可分享**。

> 你的模型，你的跑分，你的卡片 —— 速度，你说了才算。

## 三根钉子

- **信不过的数字** —— 厂商宣称的 tok/s，想自己复现？No。
- **测不清的真相** —— prefill / decode 分离、长上下文、并发下的真实速度？No。
- **拿不出的结论** —— 一张写清硬件、参数、场景的测速卡？No。

> 跑分是他们的，显卡是你的——而你连自己到底快了多少，都说不清楚。

## “裸跑分”的觉醒

从 vLLM 到 llama.cpp，从 Q4 到 Q8，你换卡、调参、比量化，却始终缺一个工具
横向回答：不同上下文长度、不同并发、不同使用场景（创意写作 / 代码生成）下，
每个模型的 prefill / decode 到底是多少。

> 朕的 token 呢？！他们只报最好看的那个数字，剩下的让朕自己猜吗？！

## 搭一座测速台

在自己的网关前，用自己的语料，测千百种组合。

- **只认协议，不认框架** —— 挂在 OpenAI 兼容网关后的模型（llama.cpp / vLLM /
  LiteLLM / 云端 API）一视同仁，5 分钟跑完一轮矩阵
- **口径透明** —— prefill 扣除实测网络往返、decode 滑窗差分并以 usage 权威回写、
  真实源码与公版文学文本撑上下文、随机 nonce 破 prefix cache：不注水，也不被水
- **实时可见** —— prefill 等待期估值逐秒收敛、decode 逐帧刷新；跑完自动生成
  折线图 + 测速卡片 PNG，一键分享
- **凭据不出服务端** —— API Key 只留 `.env`，不进 URL、不进前端、不落存档

**数据归你，方法归你，结论归你。**

> 不是要取代谁家的跑分榜，而是以前没得选——现在时代提供了另一种可能：以你为主的测速。

## 三分钟上手

```bash
./start.sh        # Linux / macOS；Windows 用 start.bat
```

自动建虚拟环境、装依赖、生成 `.env`；浏览器打开 http://127.0.0.1:8501：
顶栏「Provider 管理」抽屉里填好网关与 Key（页面写回 `.env`，永不回显）→
「刷新模型列表」勾选模型 → 「开始测速」→ 完成自动生成测速卡片 → 「下载 PNG」。

口径与方法的全文：`design/measurement.md`；每次取舍的来龙去脉：`decision/INDEX.md`。

## 几个会咬人的坑

- 长上下文 prefill 可能跑几分钟：单请求超时默认 1200s，**网关自身**超时
  （nginx `proxy_read_timeout` / LiteLLM `request_timeout`）也要够大
- 超窗档位在「Provider 管理」抽屉配好 `deployments[].max_ctx` 即自动跳过，
  不必用 400 试错
- 同一台物理后端不要多标签页并行测——硬件互挤、数据失真；跨独立后端并行才有意义
- 服务端缺省只绑 `127.0.0.1`；确需局域网访问加 `HOST=0.0.0.0`，但全端点无鉴权，自行评估

## 测试

```bash
python -m unittest discover -s tests -v   # 引擎单测 + mock 端到端 + 配置接口
bash tests/check_frontend_js.sh           # 前端 JS 语法检查（需 node）
```

## 文件结构（节选）

```
server.py          FastAPI 服务端（provider 解析 + 测速任务 + SSE + 历史 + 配置写入）
bench.py           测速引擎（prompt 构造、流式计时、矩阵调度）
static/index.html  单页前端（ECharts 图表 + html2canvas 卡片 + 配置编辑）
mock_server.py     假 OpenAI 网关（自测）
corpus/            真实语料（code: llama.cpp 源码 / creative: 公版《红楼梦》）
tests/             回归测试
config.json / .env  provider 配置与凭据（不入库）
results/           历次测速结果 JSON
```
