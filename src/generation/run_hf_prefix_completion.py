import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = "".join(parts[1:-1]).strip()
    return t.strip()


def load_model_and_tokenizer(model_path: str, dtype: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    return tok, model


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def gen_one(model, tok, prompt: str, max_new_tokens: int, temperature: float):
    device = get_model_device(model)
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = temperature > 0.0
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=None if not do_sample else temperature,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_tokens = out[0, input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", default="/root/autodl-tmp/LLM/Deepseek-coder")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_file = Path(args.task_file)
    run_dir = data_root / "runs" / args.run_id
    ws_root = data_root / "workspaces" / args.run_id

    ensure_dir(run_dir)
    ensure_dir(ws_root)

    print(f"[LOAD MODEL] {args.model_path}", flush=True)
    tok, model = load_model_and_tokenizer(args.model_path, args.dtype)

    rows = read_jsonl(task_file)
    if args.limit > 0:
        rows = rows[:args.limit]

    pred_out = run_dir / "predictions.jsonl"
    raw_out = run_dir / "raw_outputs.jsonl"
    meta_out = run_dir / "meta.json"

    with pred_out.open("w", encoding="utf-8") as fp, raw_out.open("w", encoding="utf-8") as fr:
        for i, r in enumerate(rows, start=1):
            md = r.get("metadata", {}) or {}
            task_id = md.get("task_id", f"task_{i}")
            repo_id = task_id.split("/")[0] if "/" in task_id else md.get("repo", "")
            function_name = md.get("function_name", "")
            filepath = md.get("filepath", "")

            prompt = r.get("prompt", "")
            if not prompt:
                prompt = r.get("full_left_context", "")

            ws = ws_root / task_id.replace("/", "__")
            ensure_dir(ws)
            write_text(ws / "prompt.txt", prompt)

            print(f"[{i}/{len(rows)}] starting {task_id}", flush=True)

            t0 = time.time()
            try:
                raw = gen_one(
                    model=model,
                    tok=tok,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                latency = time.time() - t0
                comp = clean(raw)

                write_text(ws / "raw_text.txt", raw)
                write_text(ws / "clean_text.txt", comp)

                fp.write(json.dumps({
                    "task_id": task_id,
                    "repo_id": repo_id,
                    "function_name": function_name,
                    "filepath": filepath,
                    "target_file_in_repo": filepath,
                    "prediction": comp,
                    "returncode": 0,
                    "workspace": str(ws),
                    "valid_prediction": bool(comp.strip()),
                    "selected_candidate_index": 0,
                    "selected_candidate_score": 0.0,
                }, ensure_ascii=False) + "\n")

                fr.write(json.dumps({
                    "latency_sec": latency,
                    "raw_text": raw,
                    "clean_text": comp,
                    "metadata": md,
                }, ensure_ascii=False) + "\n")

                print(
                    f"[{i}/{len(rows)}] {task_id} | rc=0 | answer_len={len(comp)}",
                    flush=True,
                )

            except Exception as e:
                latency = time.time() - t0
                err = repr(e)
                write_text(ws / "error.txt", err)

                fp.write(json.dumps({
                    "task_id": task_id,
                    "repo_id": repo_id,
                    "function_name": function_name,
                    "filepath": filepath,
                    "target_file_in_repo": filepath,
                    "prediction": "",
                    "returncode": -1,
                    "workspace": str(ws),
                    "valid_prediction": False,
                    "selected_candidate_index": 0,
                    "selected_candidate_score": -1.0,
                }, ensure_ascii=False) + "\n")

                fr.write(json.dumps({
                    "latency_sec": latency,
                    "raw_text": "",
                    "clean_text": "",
                    "metadata": md,
                    "error": err,
                }, ensure_ascii=False) + "\n")

                print(
                    f"[{i}/{len(rows)}] {task_id} | rc=-1 | error={err}",
                    flush=True,
                )

    with meta_out.open("w", encoding="utf-8") as f:
        json.dump({
            "run_id": args.run_id,
            "task_file": str(task_file),
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "dtype": args.dtype,
            "n_samples": len(rows),
            "protocol": "reproduce_previous_prefix_completion",
        }, f, ensure_ascii=False, indent=2)

    print("DONE", flush=True)
    print(f"PRED: {pred_out}", flush=True)
    print(f"RAW : {raw_out}", flush=True)
    print(f"META: {meta_out}", flush=True)


if __name__ == "__main__":
    main()