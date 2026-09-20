# static/vendor — 本地 vendored 前端依赖

页面（`static/index.html`）的图表与卡片导出库**随仓库分发、本地加载**，不再走
CDN。这样做的两个理由：

1. **离线/内网可用**：本工具是本地部署测速台，很多场景没有外网；此前 CDN 不可达
   时曲线与卡片 PNG 直接不可用（只有一条告警）。
2. **消掉一条供应链面**：第三方脚本运行在**能调用本机全部 API**（改配置、删历史、
   触发对云端付费网关的测速）的页面里。本地化后这条面不存在，也就不需要 SRI。

升级库时**必须同时改**：本文件的版本与哈希、`static/index.html` 的引用
（`/static/vendor/<file>`）、以及 `tests/check_frontend_js.sh` 的"无外部资源"门禁
（门禁只保证不外链，不校验哈希——哈希由本文件与 code review 保证）。

## 文件

| 文件 | 版本 | 字节 | sha256 |
|---|---|---|---|
| `echarts.min.js` | 5.5.0 | 1,029,203 | `42f8329d989b6f6539dd2b15bbdf0d82025762ac112fbb60dc57b27d7bcf3946` |
| `html2canvas.min.js` | 1.4.1 | 198,689 | `e87e550794322e574a1fda0c1549a3c70dae5a93d9113417a429016838eab8cb` |

抓取方式（记录于此以便复核"是不是官方原文件"）：

```bash
curl -sS -o static/vendor/echarts.min.js \
  https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js
curl -sS -o static/vendor/html2canvas.min.js \
  https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js
sha256sum static/vendor/*.js   # 与上表比对
```

抓取日期：2026-09-20。两个文件都自带许可头（echarts 为 Apache-2.0 头、
html2canvas 为 MIT 头），未做任何改动（非重打包、非 tree-shaking）。

## 许可

- **Apache ECharts 5.5.0** — Apache License 2.0。
  本项目本身即以 Apache-2.0 发布（见仓库根 `LICENSE`），许可兼容；
  ECharts 的版权与许可头随文件保留。
  上游：https://github.com/apache/echarts
- **html2canvas 1.4.1** — MIT License。版权与许可头随文件保留：

  ```
  Copyright (c) 2022 Niklas von Hertzen <https://hertzen.com>
  Released under the MIT License (https://opensource.org/licenses/MIT)
  ```

## 字体

页面**不加载外部字体**：`--font` 使用系统字体栈
（`Inter → PingFang SC → Microsoft YaHei → Noto Sans CJK SC → system-ui`），
`Inter` 只在本机已安装时生效。此前从 Google Fonts 拉 Inter，对本地工具属无谓的
外发请求，且 Inter 无中文字形、中文仍需系统回退。
