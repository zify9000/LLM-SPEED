# Agent 场景语料

来源：[nebius/swe-agent-trajectories](https://huggingface.co/datasets/nebius/swe-agent-trajectories)
（ModelScope 镜像：AI-ModelScope/SWE-agent-trajectories），许可 **CC-BY-4.0**。

内容为 SWE-agent 框架在 SWE-bench dev 与 nebius/SWE-bench-extra 任务上的
真实执行轨迹（系统提示 + 工具定义 + issue 任务 + 思考/命令/观测交替）。
每个文件一个轨迹块（≥2000 字符，按轮次边界聚合，**单块 ≤ 40200 字符**）。

> 块长上界按**实测**写（当前最大 40,183 字符，3 个块超 40,000）：生成脚本
> 的 40000 判定晚于仓内语料，两者已漂移。**不要为了对齐数字重建语料**——
> 重建会改变 chars/token 与 Agent 矩阵的实测形态、让历史存档失去可比性。
> 细节、清单哈希与核对方式见 `PROVENANCE.md`。

重新生成：`scripts/build_agent_corpus.py`（仅需 pyarrow，非运行时依赖）；
自检块长契约：`python scripts/build_agent_corpus.py --verify`（只校验不写盘）。

