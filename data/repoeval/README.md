# RepoEval inputs

本目录保存论文实际使用的三类 RepoEval 输入。每行是一个 JSON 对象，包含任务元数据、左/右上下文、跨文件检索上下文和人工参考补全。

| 文件 | 任务 | 样本数 |
| --- | --- | ---: |
| `function.jsonl` | Function completion | 455 |
| `line.jsonl` | Line completion | 1600 |
| `api.jsonl` | API completion | 1600 |

这些文件足以复算人工基线、重新评估已有模型输出，以及重新运行不依赖完整仓库快照的生成方法。mini-swe-agent 需要额外准备 RepoEval 的目标仓库快照；快照体积较大且属于第三方项目，因此不直接纳入本仓库。

文件完整性可通过 `SHA256SUMS` 检查。
