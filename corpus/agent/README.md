# Agent 场景语料

来源：[nebius/swe-agent-trajectories](https://huggingface.co/datasets/nebius/swe-agent-trajectories)
（ModelScope 镜像：AI-ModelScope/SWE-agent-trajectories），许可 **CC-BY-4.0**。

内容为 SWE-agent 框架在 SWE-bench dev 与 nebius/SWE-bench-extra 任务上的
真实执行轨迹（系统提示 + 工具定义 + issue 任务 + 思考/命令/观测交替）。
每个文件一个轨迹块（≥2000 字符，按轮次边界聚合，单块 ≤40000 字符）。

重新生成：`scripts/build_agent_corpus.py`（仅需 pyarrow，非运行时依赖）。
