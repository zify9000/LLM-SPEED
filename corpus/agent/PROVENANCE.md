# 语料溯源 — corpus/agent（Agent 调用场景）

## 来源

| 字段 | 值 |
|---|---|
| 上游 | [nebius/swe-agent-trajectories](https://huggingface.co/datasets/nebius/swe-agent-trajectories)（CC-BY-4.0） |
| 镜像 | ModelScope `AI-ModelScope/SWE-agent-trajectories`（同一份数据） |
| 内容 | SWE-agent 在 SWE-bench dev 与 nebius/SWE-bench-extra 上的**真实执行轨迹**（系统提示 + 工具定义 + 任务 + 思考/命令/观测交替） |
| 许可 | CC-BY-4.0（署名：nebius/swe-agent-trajectories；出处见上表） |
| 首次入库 | 2026-08-22（git 首次提交日期） |
| 生成脚本 | `scripts/build_agent_corpus.py`，sha256 `919e11e7c6bc21213ea3229be2a4801bc5c8dc31a34e213f669edbf61167a75f` |

## 产物

| 字段 | 值 |
|---|---|
| 文件数 | 100（`traj-<序号>-<instance_id>.txt`；`README.md` 另计） |
| 总字节 | 3,029,818 |
| 内容清单 sha256 | `4c85ab95e43d58c782546cc7ec345a859283fdd9f540bb93a8890d1367627778` |
| 单块字符数 | min 3,680 ／ 中位 34,434 ／ **max 40,183** |
| 超 40,000 字符的块 | **3 个**（`traj-0015`(40183)、`traj-0097`(40036)、`traj-0011`(40033)） |

清单算法（与 `tests/test_engine_unit.py -k corpus` 内断言一致）：文件按相对路径
排序，逐个累积 `相对路径 + "\0" + 文件内容` 的 sha256（流式，1MB 分块）。

## ⚠ 语料与生成器不同源（已知，勿"顺手重建"）

`README.md` 与脚本 docstring 都声明单块 **≤ 40000 字符**，但实测有 3 个块越界。
2026-09-20 用**当前版脚本**对 3,000 组随机段序列做 fuzz，**不复现越界** ——
说明**仓内语料是脚本早期修订的产物**，两者已漂移。

处理方式（已采用，见下"决策"）：**不改语料、把自述数字对准事实**。理由是改语料
会改变 chars/token 与 Agent 矩阵的实测形态，让历史存档失去可比性；而 3 个块
超出 0.5% 对测量没有实质影响。

## 决策（2026-09-20）

- 保留现有语料原样，**不重新生成**；
- `README.md` 的块长上界改为实测值（≤ 40,200 字符），并注明"生成器修订晚于
  语料，若重建会得到不同产物"；
- 新增 `scripts/build_agent_corpus.py --verify`：只校验不写盘，用来在**未来**
  重建时自检块长契约；
- 新增语料契约测试：文件数 / 清单 sha256 / 字符区间 越界即红。这样"语料被误改
  或误重建"会立刻暴露，而不是静默改变测量尺度。

## 与引擎门限的关系

- 本目录所有 100 个块都 ≥ 2000 字符，满足 `bench.py` `_load_agent_pool()` 的
  采纳门限（`if len(text) >= 2000`）；目录缺失时引擎回退内置合成轨迹块。
- **`README.md` 是目录里唯一的 <2000 字符文件（391 字符）**，它不是语料块，
  不参与加载（引擎只读 `*.txt`）。
- 同名 `instance_id` 会有多条 rollout，文件名用全局序号区分，避免互相覆盖。

## 如何验证

```bash
cd /home/zify/myLog/projectRepo/LLM-SPEED
python -m pytest -q tests/test_engine_unit.py -k corpus -v      # 清单 + 统计区间
python scripts/build_agent_corpus.py --verify                   # 块长契约（不写盘）
```
