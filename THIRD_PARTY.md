# Third-party components

本仓库只保存论文作者编写或修改的实验脚本、论文使用的数据输入和实验产物，不整体复制第三方项目源码。

- RepoEval / RepoCoder：论文的 Function、Line、API 补全任务及目标仓库快照来源。
- Repoformer：检索增强生成基线。仓库中的 `run_repoformer.py` 是为本论文实验适配的运行脚本；模型权重和上游项目需另行获取。
- mini-swe-agent：Agent 运行框架。实验环境使用版本 2.2.8；本仓库只保存任务适配脚本与配置。
- DeepSeek-Coder 7B、Qwen2.5-Coder 7B Instruct、DeepSeek-V4-Flash：模型权重或 API 服务不随仓库分发。

使用上述组件时，请遵守各上游项目、模型权重、数据集和 API 服务各自的许可证与使用条款。
