# 人工与大模型生成代码的多维度对比评估

本仓库是重庆大学软件工程本科毕业论文《人工与大模型生成代码的多维度对比评估》的实验复现代码与数据归档。实验以 RepoEval 为统一任务源，在 Function、Line 和 API 三类仓库级代码补全任务上，对人工参考代码与五类生成方法进行比较，并从参考一致性、可执行性、工程质量、上下文一致性和风险暴露五个维度进行评估。

论文 PDF 未纳入仓库，以避免公开学号等个人信息。

## 实验对象

| 类别 | 方法 |
| --- | --- |
| 人工基线 | RepoEval human reference |
| 本地代码模型 | DeepSeek-Coder 7B、Qwen2.5-Coder 7B Instruct |
| API 模型 | DeepSeek-V4-Flash |
| 检索增强 | Repoformer 7B |
| Agent | mini-swe-agent + DeepSeek-V4-Flash |

RepoEval 共包含 3655 个实验样本：Function 455 个、Line 1600 个、API 1600 个。每个方法均按相同任务集合统计，整体指标按三类任务样本数加权。

## 论文核心结果

| 方法 | EM | ES | Comp | CC | DLR | LLR | IE | CIOR | DCall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Human Baseline | 1.0000 | 1.0000 | 0.9067 | 1.2528 | 0.0042 | 0.0444 | 1.9972 | 0.0000 | 0.0052 |
| DeepSeek-Coder 7B | 0.3395 | 0.5959 | 0.2566 | 1.9174 | 0.0585 | 0.0328 | 4.0245 | 0.0000 | 0.0249 |
| Qwen2.5-Coder 7B Instruct | 0.1633 | 0.3818 | 0.3513 | 2.1789 | 0.0691 | 0.0350 | 3.8037 | 0.0000 | 0.0293 |
| DeepSeek-V4-Flash (API) | 0.1472 | 0.3450 | 0.6416 | 2.7497 | 0.0954 | 0.0415 | 3.4859 | 0.0000 | 0.0339 |
| Repoformer 7B | 0.4211 | 0.6918 | 0.2178 | 7.7702 | 0.1070 | 0.0385 | 3.1108 | 0.8205 | 0.0167 |
| mini-swe-agent (DeepSeek-V4-Flash) | 0.2547 | 0.4303 | 0.4025 | 12.7677 | 0.0528 | 0.0338 | 2.6783 | 0.8246 | 0.0107 |

其中 EM、ES、Comp、CIOR 越大越好；CC、DLR、LLR、IE、DCall 越小越好。完整指标定义见论文附录，逐样本字段保存在各结果目录的 `metrics.jsonl` 中。

## 仓库结构

```text
.
├── configs/                 # Agent 任务模板和模型配置
├── data/repoeval/           # 论文实际使用的 3655 个输入样本
├── requirements/            # 评估、绘图和生成依赖
├── results/
│   ├── final/               # 六种方法的最终输出、逐样本指标和 summary
│   ├── figures/             # 论文图表及可重新生成的图表
│   └── tables/              # 汇总 CSV
├── scripts/verify_results.py
└── src/
    ├── evaluation/          # 多维度评估实现
    ├── generation/          # 本地模型、API、Repoformer、Agent 生成脚本
    └── plotting/            # 从 summary 重建论文图表
```

未纳入的内容包括模型权重、虚拟环境、缓存、日志、探索性 sanity/pilot 运行、重复压缩包、Agent 临时工作区，以及约 1.9 GB 的第三方仓库快照。

## 快速验证

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
python scripts/verify_results.py
```

验证脚本会检查三个数据文件的样本数，重新读取 18 份最终 summary，按 455/1600/1600 加权，并逐项核对论文表 4.4–4.10 的整体数值。

重新绘图：

```bash
pip install -r requirements/plotting.txt
python src/plotting/plot_paper_results.py
```

新图表会写入 `results/figures/regenerated/`。

## 重新评估已有输出

以下示例重新评估 DeepSeek API 的 Function 输出：

```bash
python src/evaluation/evaluate_repoeval.py \
  --groundtruth data/repoeval/function.jsonl \
  --prediction results/final/deepseek_api/function/prediction.jsonl \
  --raw results/final/deepseek_api/function/raw_generation.jsonl \
  --output_dir runs/deepseek_api_function_eval
```

评估器输出 `metrics.jsonl` 和 `summary.json`。它实现 RepoEval 对齐的 EM/ES，并计算 Comp、CC、DLR、LLR、IE、CIOR、DCall 及附录中列出的扩展指标。代码重组与编译检查不是隔离沙箱，只应对可信数据运行。

## 重新生成模型输出

GPU 推理前先安装与本机 CUDA 匹配的 PyTorch，再安装 `requirements/generation.txt`。

### 本地 Hugging Face 模型

```bash
python src/generation/run_hf_prefix_completion.py \
  --data-root . \
  --task-file data/repoeval/function.jsonl \
  --run-id deepseek_coder_function \
  --model-path /path/to/model \
  --limit 455 \
  --max-new-tokens 128 \
  --temperature 0
```

论文中的 DeepSeek-Coder 和 Qwen2.5-Coder 均采用单候选、temperature=0、max_new_tokens=128。模型权重不随仓库分发。

### DeepSeek API

密钥必须通过环境变量传入，禁止写入源码：

```bash
export DEEPSEEK_API_KEY="your-key"
python src/generation/run_deepseek_api.py \
  --data-root . \
  --task-file data/repoeval/api.jsonl \
  --task-type api \
  --run-id deepseek_api_api \
  --model-name deepseek/deepseek-v4-flash \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 4096 \
  --timeout 1200 \
  --num-candidates 1
```

### Repoformer

`run_repoformer.py` 需要 Repoformer 模型权重、vLLM 和处理后的 RepoEval 数据。示例：

```bash
python src/generation/run_repoformer.py \
  --model_path /path/to/repoformer-7b \
  --data_root data/repoeval \
  --output_root runs/repoformer \
  --tasks line api function
```

### mini-swe-agent

Agent 复现还需要 mini-swe-agent 2.2.8，以及按 RepoEval 原始目录结构准备的 `repos_source/function_level/` 和 `repos_source/line_and_api_level/`。三类任务分别使用 `configs/agent/function.yaml`、`line.yaml` 和 `api.yaml`；模型配置为 `configs/models/deepseek_v4_flash_thinking.yaml`。

```bash
export DEEPSEEK_API_KEY="your-key"
python src/generation/run_mini_agent.py \
  --data-root . \
  --task-file data/repoeval/function.jsonl \
  --task-type function \
  --run-id mini_agent_function \
  --limit 455 \
  --mini-cwd /path/to/mini-swe-agent \
  --mini-config-base configs/agent/function.yaml \
  --mini-config-model configs/models/deepseek_v4_flash_thinking.yaml \
  --mini-model-name deepseek/deepseek-v4-flash \
  --timeout 1200 \
  --num-candidates 1
```

论文正式设置为 thinking enabled、reasoning_effort=high、单候选、最长 4096 token。完整目标仓库快照未直接纳入 Git，以控制体积并避免重复分发第三方源码；来源与注意事项见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 数据与结果说明

- `data/repoeval/*.jsonl`：统一实验输入，包含人工参考补全。
- `results/final/**/prediction.jsonl`、`raw_generation.jsonl` 或 `raw_outputs.jsonl`：模型输出（若原运行保存）。
- `results/final/**/metrics.jsonl`：逐样本标准化代码与全部指标；Repoformer 的标准化输出可由该文件恢复。
- `results/final/**/*summary.json`：按方法和任务汇总的最终结果。
- `results/figures/`、`results/tables/`：论文使用的图表和中间表格。

## 安全与许可证

- API 密钥只从环境变量读取；`.env`、密钥文件、模型权重和运行工作区均被 `.gitignore` 排除。
- 本仓库尚未选择开源许可证；在仓库所有者添加许可证之前默认保留全部权利。
- 第三方组件仍受各自许可证约束，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
