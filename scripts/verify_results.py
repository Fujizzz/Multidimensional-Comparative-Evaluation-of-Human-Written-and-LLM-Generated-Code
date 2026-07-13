#!/usr/bin/env python3
"""Verify bundled data and reproduce the thesis' weighted overall metrics."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_COUNTS = {"function": 455, "line": 1600, "api": 1600}
METRICS = {
    "EM": "exact_match",
    "ES": "edit_similarity",
    "Comp": "compile_ok",
    "CC": "cyclomatic_complexity",
    "DLR": "duplicated_line_ratio",
    "LLR": "long_line_ratio",
    "IE": "identifier_entropy",
    "CIOR": "context_identifier_overlap_ratio",
    "DCall": "dangerous_call_count",
}

SUMMARY_PATHS = {
    "Human Baseline": "results/final/human/{task}/quality_eval/human_{task}_summary.json",
    "DeepSeek-Coder 7B": "results/final/deepseek_coder_7b/{task}/custom_quality_eval/deepseek_coder_7b_{task}_summary.json",
    "Qwen2.5-Coder 7B Instruct": "results/final/qwen25_coder_7b_instruct/{task}/custom_quality_eval/qwen_{task}_summary.json",
    "DeepSeek-V4-Flash (API)": "results/final/deepseek_api/{task}/quality_eval/deepseek_v4_flash_{task}_summary.json",
    "Repoformer 7B": "results/final/repoformer/{task}/repoformer_{task}_summary.json",
    "mini-swe-agent (DeepSeek-V4-Flash)": "results/final/mini_agent/{task}/quality_eval/mini_agent_{task}_summary.json",
}

EXPECTED = {
    "Human Baseline": (1.0000, 1.0000, 0.9067, 1.2528, 0.0042, 0.0444, 1.9972, 0.0000, 0.0052),
    "DeepSeek-Coder 7B": (0.3395, 0.5959, 0.2566, 1.9174, 0.0585, 0.0328, 4.0245, 0.0000, 0.0249),
    "Qwen2.5-Coder 7B Instruct": (0.1633, 0.3818, 0.3513, 2.1789, 0.0691, 0.0350, 3.8037, 0.0000, 0.0293),
    "DeepSeek-V4-Flash (API)": (0.1472, 0.3450, 0.6416, 2.7497, 0.0954, 0.0415, 3.4859, 0.0000, 0.0339),
    "Repoformer 7B": (0.4211, 0.6918, 0.2178, 7.7702, 0.1070, 0.0385, 3.1108, 0.8205, 0.0167),
    "mini-swe-agent (DeepSeek-V4-Flash)": (0.2547, 0.4303, 0.4025, 12.7677, 0.0528, 0.0338, 2.6783, 0.8246, 0.0107),
}


def jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def summary_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["overall"]["prediction"]
    return {short: float(values[field]) for short, field in METRICS.items()}


def load_task_metrics() -> dict[str, dict[str, dict[str, float]]]:
    loaded: dict[str, dict[str, dict[str, float]]] = {}
    for method, pattern in SUMMARY_PATHS.items():
        loaded[method] = {}
        for task in TASK_COUNTS:
            path = ROOT / pattern.format(task=task)
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded[method][task] = summary_metrics(path)
    return loaded


def weighted_overall(task_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    total = sum(TASK_COUNTS.values())
    result: dict[str, dict[str, float]] = {}
    for method, by_task in task_metrics.items():
        result[method] = {
            metric: sum(by_task[task][metric] * TASK_COUNTS[task] for task in TASK_COUNTS) / total
            for metric in METRICS
        }
    return result


def main() -> None:
    for task, expected in TASK_COUNTS.items():
        actual = jsonl_count(ROOT / "data" / "repoeval" / f"{task}.jsonl")
        if actual != expected:
            raise AssertionError(f"{task}.jsonl: expected {expected} rows, got {actual}")

    overall = weighted_overall(load_task_metrics())
    headers = list(METRICS)
    print("Method\t" + "\t".join(headers))
    for method, values in overall.items():
        row = [values[name] for name in headers]
        print(method + "\t" + "\t".join(f"{value:.4f}" for value in row))
        expected = EXPECTED[method]
        for name, actual, target in zip(headers, row, expected):
            if abs(actual - target) > 5e-4:
                raise AssertionError(f"{method} {name}: expected {target:.4f}, got {actual:.4f}")

    print("\nOK: data counts and all paper-level weighted metrics match.")


if __name__ == "__main__":
    main()
