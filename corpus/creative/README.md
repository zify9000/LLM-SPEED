# corpus/creative 语料来源

- `hongloumeng.txt`：清·曹雪芹《紅樓夢》全文，取自
  [Project Gutenberg #24264](https://www.gutenberg.org/ebooks/24264)
  （Project Gutenberg License，公有领域文本，可自由复制/再分发）。
  **PG 的版权声明原文随附在同目录 `LICENSE-PG.txt`**；来源/产物/清单哈希/
  实测关系与一处**待确认的内容缺失**见 `PROVENANCE.md`。
- 加工方式（见 `tests/test_engine_unit.py` 性质与加载逻辑）：
  1. 去除 Project Gutenberg 英文页眉/页脚与残留英文行、U+FFFD 字符；
  2. 移除句内空白（源文本的硬换行）；
  3. 按句切分后聚合成 150~260 字符的语料块（块边界落在句末，完整连贯），
     仅保留中文占比 ≥ 0.8 的块、丢弃 < 40 字符的碎块。
- 用途：创意写作场景的上下文填充语料。原始内置池仅 672 字符，4K 档即整池
  循环 ~9 遍（256K 档 ~585 遍）——高重复文本会让投机采样（MTP 类）draft 命中
  被人为拉满、decode 虚高（ADR-0013 的判据）。83.9 万字符的真实文学文本把
  循环点推到 512K 档之后。
- 目录缺失/为空时引擎回退内置 `_CREATIVE_POOL`（bench.py 内的 6 段散文）。
- **语料是测速的"刻度"**：改语料会让历史存档失去可比性。改动前后请跑
  `python -m pytest -q tests/test_engine_unit.py -k corpus -v`（断言清单
  sha256）并同步更新 `PROVENANCE.md`。
