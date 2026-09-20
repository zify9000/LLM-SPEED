# 语料溯源 — corpus/code（代码生成场景）

## 来源

| 字段 | 值 |
|---|---|
| 上游 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT） |
| 许可 | MIT，原文见同目录 `LICENSE` |
| 首次入库 | 2026-08-14（git 首次提交日期；上游具体 commit 见下"版本"） |

## ⚠ 版本不可考（待用户确认）

项目文档（`ARCHITECTURE.md`、`design/measurement.md`、ADR-0065）均声称本语料为
**llama.cpp b9934**，但**仓库内没有任何可验证的版本标记**：

- `grep -rn "b9934\|LLAMA_BUILD_NUMBER" corpus/code/` → 无命中；
- vendored 的文件是"挑选后的 15 个源文件"，不是完整仓库快照，因此无法从
  `.git`、`CMakeLists.txt` 或 `include/llama.h` 反查版本号。

**请确认实际版本**（在哪台机器、哪个 commit 抓的），然后：

1. 把版本号（commit SHA + 日期）写进本表；
2. 同步修正文档里的 "b9934" 表述，改为指向本文件；
3. 若无法回溯，就把文档措辞改为"版本不可考（抓取日期 2026-08-14）"——
   **不要保留一个无法验证的版本号**，语料版本直接决定 chars/token 与 MTP
   行为，是测量可比性的一部分。

## 产物

| 字段 | 值 |
|---|---|
| 文件数 | 15 个源文件（`LICENSE` 另计，不参与内容清单） |
| 总字节 | 1,766,874 |
| 内容清单 sha256 | `06b408c7b5d00a2c5d8bb0160f9705d500425ca5802086a06feba162dff8d887` |

清单算法（与 `tests/test_engine_unit.py -k corpus` 内断言完全一致）：
把文件按**相对本目录的路径排序**，逐个累积 `相对路径 + "\0" + 文件内容` 的
sha256（流式，1MB 分块）。

文件清单（`git ls-files corpus/code`，去掉 `LICENSE`）：

```
common/chat.cpp              common/sampling.cpp          common/speculative.cpp
convert_hf_to_gguf_update.py ggml/src/ggml-backend.cpp     ggml/src/ggml-quants.c
ggml/src/ggml.c              include/llama.h              src/llama-batch.cpp
src/llama-context.cpp        src/llama-graph.cpp          src/llama-kv-cache.cpp
src/llama-model.cpp          src/llama-sampler.cpp        src/llama-vocab.cpp
```

## 加工方式

**未加工**：按"真实项目源码"原样 vendored（不裁剪、不改写）。选这 15 个文件的
依据是让代码场景的语料具备真实工程结构（头文件 ↔ 实现互相引用、同模块文件相邻），
见 ADR-0005/0065 与 `design/measurement.md` 的"上下文构造"节。

引擎侧的**乱序策略**（按目录聚成模块块、只乱序块次序、块内文件名排序）是
`bench.py` 的 `_make_module_stream` 实现的，不在语料文件里。

## 如何验证

```bash
cd /home/zify/myLog/projectRepo/LLM-SPEED
python -m pytest -q tests/test_engine_unit.py -k corpus -v   # 断言清单 sha256
```

`bench.py` 的 `_load_code_pool()` 在目录缺失时回退内置合成模块池——所以
**语料消失不会报错，只会静默降级**。上面的测试就是为了堵住这个静默降级
（它同时断言语料确实存在且清单未变）。
