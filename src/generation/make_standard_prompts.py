import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ALL_REPOS = [
    "alibaba_FederatedScope",
    "awslabs_fortuna",
    "google_vizier",
    "huggingface_diffusers",
    "huggingface_evaluate",
    "nerfstudio-project_nerfstudio",
    "opendilab_ACE",
    "pytorch_rl",
]


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_repo(row: dict) -> str:
    md = row.get("metadata", {}) or {}
    repo = md.get("repo") or md.get("repo_name")
    if repo:
        return repo
    task_id = md.get("task_id", "")
    if "/" in task_id:
        return task_id.split("/")[0]
    return "UNKNOWN_REPO"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-repo-cap", type=int, default=-1)
    parser.add_argument("--use-all-repos", action="store_true")
    args = parser.parse_args()

    task_file = Path(args.task_file)
    output = Path(args.output)

    rows = read_jsonl(task_file)
    print(f"Loaded rows: {len(rows)} from {task_file}")

    by_repo = defaultdict(list)
    for row in rows:
        repo = get_repo(row)
        by_repo[repo].append(row)

    selected = []
    rng = random.Random(args.seed)

    target_repos = list(by_repo.keys()) if args.use_all_repos else ALL_REPOS

    print("\n[Repo counts before sampling]")
    for repo in target_repos:
        print(f"{repo}: {len(by_repo.get(repo, []))}")

    for repo in target_repos:
        repo_rows = by_repo.get(repo, [])
        if not repo_rows:
            continue

        repo_rows = list(repo_rows)
        rng.shuffle(repo_rows)

        if args.per_repo_cap is not None and args.per_repo_cap > 0:
            repo_rows = repo_rows[:args.per_repo_cap]

        for r in repo_rows:
            out_row = {
                "prompt": r.get("prompt", ""),
                "metadata": r.get("metadata", {}),
            }
            selected.append(out_row)

    print(f"\nSelected rows: {len(selected)}")
    print("[Repo counts after sampling]")
    final_counts = defaultdict(int)
    for r in selected:
        final_counts[get_repo(r)] += 1
    for repo in sorted(final_counts):
        print(f"{repo}: {final_counts[repo]}")

    write_jsonl(output, selected)
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()