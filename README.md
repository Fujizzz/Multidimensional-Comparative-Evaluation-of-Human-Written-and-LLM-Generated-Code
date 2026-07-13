# Multidimensional Comparative Evaluation of Human-Written and LLM-Generated Code

<p align="center">
  <a href="https://github.com/Fujizzz/Multidimensional-Comparative-Evaluation-of-Human-Written-and-LLM-Generated-Code/actions/workflows/verify.yml"><img src="https://github.com/Fujizzz/Multidimensional-Comparative-Evaluation-of-Human-Written-and-LLM-Generated-Code/actions/workflows/verify.yml/badge.svg" alt="Artifact verification"></a>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white" alt="Python 3.10 or 3.11">
  <img src="https://img.shields.io/badge/benchmark-RepoEval-6f42c1" alt="RepoEval benchmark">
  <img src="https://img.shields.io/badge/artifact-reproducible-2ea44f" alt="Reproducible research artifact">
</p>

> A reproducible research artifact for evaluating human-written and LLM-generated code across correctness, executability, engineering quality, contextual coherence, and risk exposure.

## Overview

This repository contains the code, data, experiment outputs, and analysis artifacts for the undergraduate thesis **“Multidimensional Comparative Evaluation of Human-Written and LLM-Generated Code”** at Chongqing University.

The study uses RepoEval as a unified benchmark and compares human-written reference code with five code-generation approaches across repository-level Function, Line, and API completion tasks. Every approach is evaluated on the same 3,655 samples using a consistent multidimensional evaluation pipeline.

| Artifact | Scope |
| --- | ---: |
| Benchmark samples | 3,655 |
| Task types | 3 |
| Evaluated approaches | 6 |
| Primary metrics | 9 |
| Final method/task summaries | 18 |

### What This Repository Provides

- A unified evaluation implementation for nine primary correctness, quality, context, and risk metrics.
- Reproducible inputs, retained model outputs, per-sample metrics, and aggregate summaries.
- Generation runners for local Hugging Face models, an API model, Repoformer, and mini-swe-agent.
- A one-command integrity check that reproduces every weighted result reported below.
- Plotting code and publication-ready figures for independent inspection.

The thesis PDF is intentionally excluded to avoid publishing personal information such as the student ID. Model weights and third-party repository snapshots are also excluded; their requirements are documented below.

## Evaluated Approaches

| Category | Method |
| --- | --- |
| Human baseline | RepoEval human reference |
| Local code models | DeepSeek-Coder 7B and Qwen2.5-Coder 7B Instruct |
| API model | DeepSeek-V4-Flash |
| Retrieval-augmented generation | Repoformer 7B |
| Agent-based generation | mini-swe-agent with DeepSeek-V4-Flash |

RepoEval contains 3,655 samples used in this study: 455 Function completion samples, 1,600 Line completion samples, and 1,600 API completion samples. Every method is evaluated on the same task set, and overall metrics are weighted by the number of samples in each task.

## Main Results

| Method | EM | ES | Comp | CC | DLR | LLR | IE | CIOR | DCall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Human Baseline | 1.0000 | 1.0000 | 0.9067 | 1.2528 | 0.0042 | 0.0444 | 1.9972 | 0.0000 | 0.0052 |
| DeepSeek-Coder 7B | 0.3395 | 0.5959 | 0.2566 | 1.9174 | 0.0585 | 0.0328 | 4.0245 | 0.0000 | 0.0249 |
| Qwen2.5-Coder 7B Instruct | 0.1633 | 0.3818 | 0.3513 | 2.1789 | 0.0691 | 0.0350 | 3.8037 | 0.0000 | 0.0293 |
| DeepSeek-V4-Flash (API) | 0.1472 | 0.3450 | 0.6416 | 2.7497 | 0.0954 | 0.0415 | 3.4859 | 0.0000 | 0.0339 |
| Repoformer 7B | 0.4211 | 0.6918 | 0.2178 | 7.7702 | 0.1070 | 0.0385 | 3.1108 | 0.8205 | 0.0167 |
| mini-swe-agent (DeepSeek-V4-Flash) | 0.2547 | 0.4303 | 0.4025 | 12.7677 | 0.0528 | 0.0338 | 2.6783 | 0.8246 | 0.0107 |

Higher values are better for EM, ES, Comp, and CIOR. Lower values are better for CC, DLR, LLR, IE, and DCall. Per-sample values for the full metric suite are available in the corresponding `metrics.jsonl` files.

## Repository Structure

```text
.
├── configs/                 # Agent task templates and model configuration
├── data/repoeval/           # The 3,655 RepoEval inputs used in the thesis
├── requirements/            # Evaluation, plotting, and generation dependencies
├── results/
│   ├── final/               # Final outputs, per-sample metrics, and summaries
│   ├── figures/             # Thesis figures and regenerated figures
│   └── tables/              # Aggregated CSV tables
├── scripts/verify_results.py
└── src/
    ├── evaluation/          # Multidimensional evaluation implementation
    ├── generation/          # Local, API, Repoformer, and agent runners
    └── plotting/            # Figure generation from summary files
```

Model weights, virtual environments, caches, logs, exploratory sanity or pilot runs, duplicate archives, agent workspaces, and approximately 1.9 GB of third-party repository snapshots are excluded.

## Quick Verification

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
python scripts/verify_results.py
```

The verification script checks the number of samples in all three datasets, loads the 18 final summary files, computes weighted metrics using the 455/1,600/1,600 task sizes, and verifies every overall value reported in the thesis.

To regenerate the main figures:

```bash
pip install -r requirements/plotting.txt
python src/plotting/plot_paper_results.py
```

Generated figures and tables are written to `results/figures/regenerated/`.

## Re-evaluating Existing Outputs

The following command re-evaluates the DeepSeek API Function-completion outputs:

```bash
python src/evaluation/evaluate_repoeval.py \
  --groundtruth data/repoeval/function.jsonl \
  --prediction results/final/deepseek_api/function/prediction.jsonl \
  --raw results/final/deepseek_api/function/raw_generation.jsonl \
  --output_dir runs/deepseek_api_function_eval
```

The evaluator produces `metrics.jsonl` and `summary.json`. It implements RepoEval-aligned EM and ES together with Comp, CC, DLR, LLR, IE, CIOR, DCall, and the extended metrics documented in the thesis appendix.

Code reconstruction and compilation checks are not executed in an isolated security sandbox. Only run the evaluator on trusted data.

## Regenerating Model Outputs

Before GPU inference, install a CUDA-compatible PyTorch build and then install the generation dependencies:

```bash
pip install -r requirements/generation.txt
```

### Local Hugging Face Models

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

The DeepSeek-Coder and Qwen2.5-Coder experiments use one candidate per sample, `temperature=0`, and `max_new_tokens=128`. Model weights are not distributed with this repository.

### DeepSeek API

API credentials must be provided through an environment variable and must never be written into source files:

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

`run_repoformer.py` requires Repoformer model weights, vLLM, and processed RepoEval data:

```bash
python src/generation/run_repoformer.py \
  --model_path /path/to/repoformer-7b \
  --data_root data/repoeval \
  --output_root runs/repoformer \
  --tasks line api function
```

### mini-swe-agent

Agent reproduction additionally requires mini-swe-agent 2.2.8 and RepoEval repository snapshots arranged under `repos_source/function_level/` and `repos_source/line_and_api_level/`.

The three tasks use `configs/agent/function.yaml`, `configs/agent/line.yaml`, and `configs/agent/api.yaml`, respectively. The model configuration is stored in `configs/models/deepseek_v4_flash_thinking.yaml`.

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

The final agent setting uses thinking mode, `reasoning_effort=high`, one candidate per task, and a maximum output length of 4,096 tokens.

Full repository snapshots are not committed to Git in order to control repository size and avoid redistributing third-party source trees. See [THIRD_PARTY.md](THIRD_PARTY.md) for component and licensing information.

## Data and Result Files

- `data/repoeval/*.jsonl`: unified experimental inputs, including the human-written reference completions.
- `results/final/**/prediction.jsonl`, `raw_generation.jsonl`, or `raw_outputs.jsonl`: model outputs where retained by the original run.
- `results/final/**/metrics.jsonl`: normalized code and the complete per-sample metric suite. Repoformer outputs can be recovered from these records.
- `results/final/**/*summary.json`: final per-method and per-task summaries.
- `results/figures/` and `results/tables/`: figures and intermediate tables used in the thesis.

## Security and Licensing

- API credentials are read exclusively from environment variables. `.env` files, credential files, model weights, and runtime workspaces are excluded by `.gitignore`.
- No open-source license has been selected. All rights are reserved until the repository owner adds a license.
- Third-party components remain subject to their respective licenses and usage terms. See [THIRD_PARTY.md](THIRD_PARTY.md).
