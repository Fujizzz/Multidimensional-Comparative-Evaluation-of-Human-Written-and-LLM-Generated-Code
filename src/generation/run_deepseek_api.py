import argparse
import json
import keyword
import os
import re
import textwrap
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import requests


# Never store credentials in source control.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


CHAT_URL_CANDIDATES = [
    "https://api.deepseek.com/chat/completions",
    "https://api.deepseek.com/v1/chat/completions",
]


def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clip_text(text: str, max_chars: int = 1800) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]..."


def normalize_target_path(repo_id: str, filepath: str) -> str:
    prefix = repo_id + "/"
    if filepath.startswith(prefix):
        return filepath[len(prefix):]
    return filepath


def build_masked_target_source(sample: dict) -> str:
    left = sample.get("full_left_context", "") or sample.get("prompt", "")
    right = sample.get("right_context", "")
    return left + right


def extract_target_context_from_masked_source(
    masked_source: str,
    function_name: str,
    window: int = 50,
) -> str:
    lines = masked_source.splitlines()
    hit = None

    needle_def = f"def {function_name}("
    needle_class = f"class {function_name}("
    for i, line in enumerate(lines):
        if needle_def in line or needle_class in line:
            hit = i
            break

    if hit is None:
        start, end = 0, min(len(lines), 120)
    else:
        start = max(0, hit - window)
        end = min(len(lines), hit + window + 1)

    snippet = []
    for idx in range(start, end):
        snippet.append(f"{idx + 1:04d}: {lines[idx]}")
    return "\n".join(snippet)


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip("\n")


def postprocess_prediction(text: str, function_name: str, task_type: str) -> str:
    t = strip_code_fences(text or "")
    if not t.strip():
        return ""

    lines = t.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    while task_type == "function" and lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)

    if task_type == "function" and lines:
        first = lines[0].strip()
        if first.startswith(f"def {function_name}(") and first.endswith(":"):
            lines = lines[1:]
            nonempty = [ln for ln in lines if ln.strip()]
            if nonempty:
                min_indent = min(len(ln) - len(ln.lstrip()) for ln in nonempty)
                lines = [ln[min_indent:] if len(ln) >= min_indent else ln for ln in lines]

    if task_type == "function" and lines:
        first = lines[0].strip()
        if first.startswith("class ") and first.endswith(":"):
            lines = lines[1:]
            nonempty = [ln for ln in lines if ln.strip()]
            if nonempty:
                min_indent = min(len(ln) - len(ln.lstrip()) for ln in nonempty)
                lines = [ln[min_indent:] if len(ln) >= min_indent else ln for ln in lines]

    while lines and not lines[0].strip():
        lines.pop(0)

    out = "\n".join(lines).rstrip()
    return out + ("\n" if out else "")


def extract_identifiers(text: str):
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or "")
    return {
        t for t in toks
        if not keyword.iskeyword(t)
        and t not in {"self", "cls", "True", "False", "None"}
    }


def build_direct_prompt(task_type: str, function_name: str, target_context_text: str, crossfile_context_text: str) -> str:
    target_label = f"Target function:\n{function_name}" if function_name else "Target function: not specified"
    return f"""You are solving a repository-level {task_type} completion task.

{target_label}

You MUST use the provided contexts below as your primary source of truth.

========== TARGET FILE LOCAL CONTEXT (MASKED) ==========
{target_context_text}
========== END TARGET CONTEXT ==========

========== CROSS-FILE CONTEXT (RETRIEVED) ==========
{crossfile_context_text}
========== END CROSS-FILE CONTEXT ==========

Return ONLY the missing completion text that fills the masked hole.

Rules:
- Do NOT write markdown fences
- Do NOT write explanations
- Do NOT write decorators such as @classmethod or @staticmethod unless they are clearly inside the missing hole
- Do NOT write a function header like def ...
- Do NOT write a class header like class ...
- Do NOT rewrite the whole function
- Do NOT repeat surrounding context that already exists outside the hole
- Prefer the shortest correct completion that fits the hole

Output only the completion text.
"""


def wrapped_body_syntax_ok(candidate: str) -> bool:
    import ast
    body = candidate.strip("\n")
    if not body.strip():
        return False
    wrapped = "def __dummy__():\n" + textwrap.indent(body, "    ") + "\n"
    try:
        ast.parse(wrapped)
        return True
    except Exception:
        return False


def score_candidate(prediction: str, task_type: str, function_name: str, target_context_text: str, crossfile_context_text: str):
    text = prediction or ""
    stripped = text.strip()

    info = {
        "score": -100.0,
        "nonempty_lines": 0,
        "syntax_ok_wrapped": False,
        "has_function_header": False,
        "has_decorator": False,
        "has_placeholder": False,
        "identifier_overlap": 0,
    }

    if not stripped:
        return info

    lines = [x for x in text.splitlines() if x.strip()]
    info["nonempty_lines"] = len(lines)

    first = lines[0].strip() if lines else ""
    has_function_header = first.startswith(f"def {function_name}(") or first.startswith("def ")
    has_decorator = first.startswith("@")
    lowered = stripped.lower()
    has_placeholder = (
        "pass" in lowered or "todo" in lowered or "notimplementederror" in lowered
    )
    syntax_ok_wrapped = wrapped_body_syntax_ok(text)

    cand_ids = extract_identifiers(text)
    ctx_ids = extract_identifiers(target_context_text + "\n" + crossfile_context_text)
    overlap = len(cand_ids & ctx_ids)

    score = 0.0
    score += 3.0
    preferred_line_count = 2 <= len(lines) <= 20 if task_type == "function" else 1 <= len(lines) <= 20
    score += 2.0 if preferred_line_count else -1.0
    score += 3.0 if syntax_ok_wrapped else -3.0
    score += 2.0 if not has_function_header else -3.0
    score += 1.0 if not has_decorator else -2.0
    score += 2.0 if not has_placeholder else -3.0
    score += min(overlap, 8) * 0.25

    info.update({
        "score": score,
        "syntax_ok_wrapped": syntax_ok_wrapped,
        "has_function_header": has_function_header,
        "has_decorator": has_decorator,
        "has_placeholder": has_placeholder,
        "identifier_overlap": overlap,
    })
    return info


def preflight_models():
    if (
        not DEEPSEEK_API_KEY
        or DEEPSEEK_API_KEY.strip() == ""
        or DEEPSEEK_API_KEY == "PASTE_YOUR_DEEPSEEK_KEY_HERE"
    ):
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable.")

    req = Request(
        "https://api.deepseek.com/models",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        method="GET",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            status = resp.status
            preview = resp.read(300).decode("utf-8", errors="ignore")
            print(f"[DeepSeek preflight] status={status}", flush=True)
            print(preview, flush=True)
            if status != 200:
                raise RuntimeError(f"DeepSeek /models returned status={status}")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek auth failed: HTTP {e.code} | {body}") from e
    except URLError as e:
        raise RuntimeError(f"DeepSeek connectivity failed: {e}") from e


def call_deepseek_chat(model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int, timeout: int):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a precise code completion model."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_error = None
    for url in CHAT_URL_CANDIDATES:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return text, data, url
            last_error = f"{url} -> HTTP {resp.status_code}: {resp.text[:500]}"
        except Exception as e:
            last_error = f"{url} -> {repr(e)}"

    raise RuntimeError(f"All DeepSeek chat endpoints failed. Last error: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-type", required=True, choices=["function", "line", "api"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model-name", default="deepseek/deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--crossfile-max-chars", type=int, default=1800)
    args = parser.parse_args()

    preflight_models()

    data_root = Path(args.data_root)
    task_file = Path(args.task_file)
    run_dir = data_root / "runs" / args.run_id
    ws_root = data_root / "workspaces" / args.run_id
    ensure_dir(run_dir)
    ensure_dir(ws_root)

    predictions_path = run_dir / "predictions.jsonl"
    samples = read_jsonl(task_file)[: args.limit]

    with predictions_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, start=1):
            metadata = sample["metadata"]
            task_id = metadata["task_id"]
            repo_id = task_id.split("/")[0]
            function_name = metadata.get("function_name", "")
            filepath = metadata["filepath"]
            target_relpath = normalize_target_path(repo_id, filepath)

            print(f"[{idx}/{len(samples)}] starting {task_id}", flush=True)

            sample_ws = ws_root / task_id.replace("/", "__")
            ensure_dir(sample_ws)

            masked_source = build_masked_target_source(sample)
            target_context = extract_target_context_from_masked_source(
                masked_source=masked_source,
                function_name=function_name,
                window=50,
            )
            crossfile_context = clip_text(
                sample.get("crossfile_context", ""),
                max_chars=args.crossfile_max_chars,
            )

            candidate_infos = []

            for cand_idx in range(args.num_candidates):
                prompt = build_direct_prompt(args.task_type, function_name, target_context, crossfile_context)
                prompt_file = sample_ws / f"prompt_{cand_idx}.txt"
                raw_json_file = sample_ws / f"raw_response_{cand_idx}.json"
                error_file = sample_ws / f"error_{cand_idx}.txt"

                write_text(prompt_file, prompt)

                try:
                    raw_text, raw_json, used_url = call_deepseek_chat(
                        model_name=args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name,
                        prompt=prompt,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    prediction = postprocess_prediction(raw_text, function_name, args.task_type)
                    write_text(raw_json_file, json.dumps({"used_url": used_url, "response": raw_json}, ensure_ascii=False, indent=2))
                    returncode = 0
                    error_msg = ""
                except Exception as e:
                    prediction = ""
                    returncode = -1
                    error_msg = repr(e)
                    write_text(error_file, error_msg)

                score_info = score_candidate(
                    prediction=prediction,
                    task_type=args.task_type,
                    function_name=function_name,
                    target_context_text=target_context,
                    crossfile_context_text=crossfile_context,
                )

                candidate_infos.append({
                    "candidate_index": cand_idx,
                    "returncode": returncode,
                    "prediction": prediction,
                    "prompt_file": str(prompt_file),
                    "raw_json_file": str(raw_json_file),
                    "error_file": str(error_file),
                    "score_info": score_info,
                    "error_msg": error_msg,
                })

            best = max(candidate_infos, key=lambda x: (x["score_info"]["score"], len(x["prediction"])))

            out_record = {
                "task_id": task_id,
                "task_type": args.task_type,
                "repo_id": repo_id,
                "function_name": function_name,
                "filepath": filepath,
                "target_file_in_repo": target_relpath,
                "prediction": best["prediction"],
                "returncode": best["returncode"],
                "workspace": str(sample_ws),
                "valid_prediction": bool(best["prediction"].strip()),
                "selected_candidate_index": best["candidate_index"],
                "selected_candidate_score": best["score_info"]["score"],
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            print(
                f"[{idx}/{len(samples)}] {task_id} | rc={best['returncode']} | "
                f"answer_len={len(best['prediction'])} | "
                f"best_cand={best['candidate_index']} | "
                f"score={best['score_info']['score']:.2f}",
                flush=True,
            )

    print(f"\nSaved predictions to: {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
