import argparse
import json
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="mini-swe-agent 输出的 predictions.jsonl")
    parser.add_argument("--prediction_out", required=True, help="给评测脚本的 prediction.jsonl")
    parser.add_argument("--raw_out", required=True, help="给评测脚本的 raw_generation.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))

    pred_rows = []
    raw_rows = []

    for row in rows:
        task_id = row["task_id"]
        pred = row.get("prediction", "") or ""

        pred_rows.append({
            "metadata": {
                "task_id": task_id
            },
            "choices": [
                {"text": pred}
            ]
        })

        raw_rows.append({
            "metadata": {
                "task_id": task_id
            },
            "clean_text": pred,
            "raw_text": pred
        })

    write_jsonl(Path(args.prediction_out), pred_rows)
    write_jsonl(Path(args.raw_out), raw_rows)

    print("converted rows =", len(rows))
    print("prediction_out =", args.prediction_out)
    print("raw_out =", args.raw_out)


if __name__ == "__main__":
    main()