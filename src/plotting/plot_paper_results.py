#!/usr/bin/env python3
"""Regenerate the main paper figures from bundled summary JSON files."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_results import load_task_metrics, weighted_overall  # noqa: E402


OUT = ROOT / "results" / "figures" / "regenerated"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Human Baseline": "#1F1F1F",
    "DeepSeek-Coder 7B": "#4C78A8",
    "Qwen2.5-Coder 7B Instruct": "#F58518",
    "DeepSeek-V4-Flash (API)": "#54A24B",
    "Repoformer 7B": "#B279A2",
    "mini-swe-agent (DeepSeek-V4-Flash)": "#E45756",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    task_metrics = load_task_metrics()
    overall = pd.DataFrame.from_dict(weighted_overall(task_metrics), orient="index")
    overall.to_csv(OUT / "overall_metrics.csv", encoding="utf-8-sig")

    scores = pd.DataFrame(index=overall.index)
    scores["Reference Consistency"] = overall[["EM", "ES"]].mean(axis=1)
    scores["Executability"] = pd.concat([overall["Comp"], 1 / (1 + overall["CC"])], axis=1).mean(axis=1)
    scores["Engineering Quality"] = pd.concat([1 - overall["DLR"], 1 - overall["LLR"]], axis=1).mean(axis=1)
    scores["Contextual Coherence"] = pd.concat([1 / (1 + overall["IE"]), overall["CIOR"]], axis=1).mean(axis=1)
    scores["Risk Exposure"] = 1 / (1 + overall["DCall"])
    scores.to_csv(OUT / "five_dimension_scores.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(scores.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(scores.columns)), scores.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(scores.index)), scores.index)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(j, i, f"{scores.iat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Normalized score (higher is better)")
    fig.tight_layout()
    save(fig, "heatmap_five_dimensions")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(overall.index))
    ax.bar(x - 0.18, overall["EM"], 0.36, label="EM")
    ax.bar(x + 0.18, overall["ES"], 0.36, label="ES")
    ax.set_xticks(x, overall.index, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save(fig, "bar_reference_consistency_overall")

    scaled = scores.copy()
    for column in scaled:
        low, high = scaled[column].min(), scaled[column].max()
        scaled[column] = 0.7 if np.isclose(low, high) else 0.35 + 0.65 * (scaled[column] - low) / (high - low)
    angles = np.linspace(0, 2 * np.pi, len(scaled.columns), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    for method, row in scaled.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(closed_angles, values, label=method, color=COLORS[method], linewidth=1.8)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, scaled.columns)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "radar_five_dimensions")

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in overall.index:
        ax.scatter(overall.loc[method, "Comp"], overall.loc[method, "CC"], s=70, color=COLORS[method], label=method)
    ax.set_xlabel("Comp")
    ax.set_ylabel("CC")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "scatter_executability_comp_cc")

    print(f"Generated figures and tables in {OUT}")


if __name__ == "__main__":
    main()
