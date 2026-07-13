
import argparse
import ast
import json
import keyword
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# Never store credentials in source control.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


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


def ensure_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def write_text(path: Path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ensure_text(text), encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def copy_repo(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def normalize_target_path(repo_id: str, filepath: str) -> str:
    prefix = repo_id + "/"
    if filepath.startswith(prefix):
        return filepath[len(prefix):]
    return filepath


def resolve_repo_src(data_root: Path, repo_id: str) -> Path:
    candidates = [
        data_root / "repos_source" / "function_level" / repo_id,
        data_root / "repos_source" / "line_and_api_level" / repo_id,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Repo source not found in either function_level or line_and_api_level: {repo_id}"
    )


def build_masked_target_source(sample: dict) -> str:
    left = sample.get("full_left_context", "") or sample.get("prompt", "")
    right = sample.get("right_context", "")
    return left + right


def overwrite_target_file_with_masked_source(
    repo_dst: Path,
    target_relpath: str,
    masked_source: str,
):
    target_path = repo_dst / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(masked_source, encoding="utf-8")


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


def clip_text(text: str, max_chars: int = 1800) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]..."


def build_agent_task(
    task_type: str,
    function_name: str,
    answer_file: Path,
    target_context_text: str,
    crossfile_context_text: str,
    candidate_index: int,
):
    target_label = f"Target function:\n{function_name}" if function_name else "Target function: not specified"
    return f"""You are solving a repository-level {task_type} completion task.

Candidate number:
{candidate_index + 1}

{target_label}

Your only valid output artifact is:
{answer_file}

You MUST use the provided contexts below as your primary source of truth.

========== TARGET FILE LOCAL CONTEXT (MASKED) ==========
{target_context_text}
========== END TARGET CONTEXT ==========

========== CROSS-FILE CONTEXT (RETRIEVED) ==========
{crossfile_context_text}
========== END CROSS-FILE CONTEXT ==========

Your job:
1. Infer ONLY the missing completion text that fills the masked hole.
2. On your first action, write ONLY the missing completion to:
   {answer_file}
3. Use a shell here-doc to write the file.
4. Do NOT wrap the completion in markdown fences.
5. Do NOT write explanations.
6. Do NOT write decorators such as @classmethod or @staticmethod unless they are clearly inside the missing hole.
7. Do NOT write a function header like "def ..." unless the function header itself is missing.
8. Do NOT write a class header like "class ..." unless the class header itself is missing.
9. Do NOT repeat surrounding context that already exists outside the hole.
10. Do NOT rewrite the whole function.
11. Do NOT modify repository files.
12. On the second action, verify that {answer_file} exists and is not empty.
13. On the third action, finish with exactly:
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Bad output examples:
- Writing the whole function again
- Writing decorators + function header when only the function body is missing
- Writing placeholder code like pass or TODO
- Writing explanatory text

Good output example:
- Only the statements that belong inside the missing region

Important rules:
- Prefer the provided target context and cross-file context.
- Do NOT search the repository unless absolutely necessary.
- The answer file is the only file you are allowed to create or modify.
- Keep command outputs short.
"""


def has_multiple_nonempty_lines(text: str) -> bool:
    lines = [x for x in text.splitlines() if x.strip()]
    return len(lines) > 1


def looks_like_trivial_answer(text: str, task_type: str) -> bool:
    t = text.strip()
    if not t:
        return True

    lines = [x for x in t.splitlines() if x.strip()]
    if task_type == "function" and len(lines) <= 1:
        return True

    lowered = t.lower()
    if lowered.startswith("def ") and len(lines) == 1:
        return True
    if lowered.startswith("class ") and len(lines) == 1:
        return True
    if "pass" in lowered or "todo" in lowered or "notimplementederror" in lowered:
        return True

    return False


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


def wrapped_body_syntax_ok(candidate: str) -> bool:
    body = candidate.strip("\n")
    if not body.strip():
        return False
    wrapped = "def __dummy__():\n" + textwrap.indent(body, "    ") + "\n"
    try:
        ast.parse(wrapped)
        return True
    except Exception:
        return False


def score_candidate(
    prediction: str,
    task_type: str,
    function_name: str,
    target_context_text: str,
    crossfile_context_text: str,
):
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


def prepare_deepseek_auth(model_name: str) -> dict:
    if (
        not DEEPSEEK_API_KEY
        or DEEPSEEK_API_KEY.strip() == ""
        or DEEPSEEK_API_KEY == "PASTE_YOUR_DEEPSEEK_KEY_HERE"
    ):
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable.")

    config_dir = Path("/root/.config/mini-swe-agent")
    ensure_dir(config_dir)

    env_text = (
        f"DEEPSEEK_API_KEY={DEEPSEEK_API_KEY}\n"
        f"MSWEA_MODEL_NAME={model_name}\n"
        "MSWEA_CONFIGURED=1\n"
    )
    write_text(config_dir / ".env", env_text)

    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    env["MSWEA_MODEL_NAME"] = model_name
    env["MSWEA_CONFIGURED"] = "1"

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

    return env


def run_one_candidate(
    cmd,
    mini_cwd: Path,
    env: dict,
    timeout: int,
    stdout_file: Path,
    stderr_file: Path,
):
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(mini_cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        returncode = proc.returncode
        stdout = ensure_text(proc.stdout)
        stderr = ensure_text(proc.stderr)
    except subprocess.TimeoutExpired as e:
        returncode = -999
        stdout = ensure_text(e.stdout)
        stderr = ensure_text(e.stderr) + "\n[TIMEOUT]"

    write_text(stdout_file, stdout)
    write_text(stderr_file, stderr)
    return returncode, stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-type", required=True, choices=["function", "line", "api"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--mini-config-base", required=True)
    parser.add_argument("--mini-config-model", required=True)
    parser.add_argument("--mini-model-name", required=True)
    parser.add_argument("--mini-cwd", default="/root/autodl-tmp/projects/mini-swe-agent")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--mini-bin", default="mini")
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--crossfile-max-chars", type=int, default=1800)
    args = parser.parse_args()

    env = prepare_deepseek_auth(args.mini_model_name)

    data_root = Path(args.data_root)
    task_file = Path(args.task_file)
    mini_cwd = Path(args.mini_cwd)

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

            repo_src = resolve_repo_src(data_root, repo_id)

            sample_ws = ws_root / task_id.replace("/", "__")
            repo_dst = sample_ws / "repo"
            answer_file = sample_ws / "agent_answer.txt"
            metadata_file = sample_ws / "metadata.json"
            prompt_file = sample_ws / "benchmark_prompt.txt"
            crossfile_file = sample_ws / "crossfile_context.txt"
            target_context_file = sample_ws / "target_context.txt"
            task_text_file = sample_ws / "agent_task.txt"
            stdout_file = sample_ws / "mini_stdout.txt"
            stderr_file = sample_ws / "mini_stderr.txt"
            candidate_scores_file = sample_ws / "candidate_scores.json"

            ensure_dir(sample_ws)
            copy_repo(repo_src, repo_dst)

            masked_source = build_masked_target_source(sample)
            overwrite_target_file_with_masked_source(repo_dst, target_relpath, masked_source)

            target_context = extract_target_context_from_masked_source(
                masked_source=masked_source,
                function_name=function_name,
                window=50,
            )
            crossfile_context = clip_text(
                sample.get("crossfile_context", ""),
                max_chars=args.crossfile_max_chars,
            )

            target_file_path = repo_dst / target_relpath
            target_file_before = read_text(target_file_path)

            agent_metadata = {
                "task_id": task_id,
                "repo_id": repo_id,
                "function_name": function_name,
                "filepath_raw": filepath,
                "target_file_in_repo": target_relpath,
                "repository_url": metadata.get("repository", ""),
                "source_url": metadata.get("url", ""),
            }

            write_text(metadata_file, json.dumps(agent_metadata, ensure_ascii=False, indent=2))
            write_text(prompt_file, sample.get("prompt", ""))
            write_text(crossfile_file, sample.get("crossfile_context", ""))
            write_text(target_context_file, target_context)

            candidate_infos = []

            for cand_idx in range(args.num_candidates):
                cand_answer_file = sample_ws / f"agent_answer_{cand_idx}.txt"
                cand_stdout_file = sample_ws / f"mini_stdout_{cand_idx}.txt"
                cand_stderr_file = sample_ws / f"mini_stderr_{cand_idx}.txt"
                cand_task_file = sample_ws / f"agent_task_{cand_idx}.txt"

                task_text = build_agent_task(
                    task_type=args.task_type,
                    function_name=function_name,
                    answer_file=cand_answer_file,
                    target_context_text=target_context,
                    crossfile_context_text=crossfile_context,
                    candidate_index=cand_idx,
                )
                write_text(cand_task_file, task_text)

                cmd = [
                    args.mini_bin,
                    "-c", args.mini_config_base,
                    "-c", args.mini_config_model,
                    "-m", args.mini_model_name,
                    "-y",
                    "-t", task_text,
                    "--exit-immediately",
                ]

                returncode, _, _ = run_one_candidate(
                    cmd=cmd,
                    mini_cwd=mini_cwd,
                    env=env,
                    timeout=args.timeout,
                    stdout_file=cand_stdout_file,
                    stderr_file=cand_stderr_file,
                )

                raw_prediction = read_text(cand_answer_file)
                prediction = postprocess_prediction(raw_prediction, function_name, args.task_type)

                if raw_prediction != prediction:
                    write_text(cand_answer_file, prediction)

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
                    "answer_file": str(cand_answer_file),
                    "stdout_file": str(cand_stdout_file),
                    "stderr_file": str(cand_stderr_file),
                    "task_file": str(cand_task_file),
                    "prediction": prediction,
                    "score_info": score_info,
                })

            write_text(candidate_scores_file, json.dumps(candidate_infos, ensure_ascii=False, indent=2))

            best = max(candidate_infos, key=lambda x: (x["score_info"]["score"], len(x["prediction"])))
            selected_prediction = best["prediction"]
            write_text(answer_file, selected_prediction)

            write_text(stdout_file, read_text(Path(best["stdout_file"])))
            write_text(stderr_file, read_text(Path(best["stderr_file"])))
            write_text(task_text_file, read_text(Path(best["task_file"])))

            target_file_after = read_text(target_file_path)
            target_file_modified = target_file_before != target_file_after

            trivial_prediction = looks_like_trivial_answer(selected_prediction, args.task_type)
            valid_prediction = (
                answer_file.exists()
                and bool(selected_prediction.strip())
                and not trivial_prediction
                and not target_file_modified
            )

            out_record = {
                "task_id": task_id,
                "task_type": args.task_type,
                "repo_id": repo_id,
                "function_name": function_name,
                "filepath": filepath,
                "target_file_in_repo": target_relpath,
                "prediction": selected_prediction,
                "returncode": best["returncode"],
                "workspace": str(sample_ws),
                "has_answer_file": answer_file.exists(),
                "trivial_prediction": trivial_prediction,
                "valid_prediction": valid_prediction,
                "target_file_modified": target_file_modified,
                "stdout_file": str(stdout_file),
                "stderr_file": str(stderr_file),
                "agent_task_file": str(task_text_file),
                "target_context_file": str(target_context_file),
                "selected_candidate_index": best["candidate_index"],
                "selected_candidate_score": best["score_info"]["score"],
                "candidate_scores_file": str(candidate_scores_file),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            print(
                f"[{idx}/{len(samples)}] {task_id} | rc={best['returncode']} | "
                f"answer_exists={answer_file.exists()} | "
                f"trivial={trivial_prediction} | "
                f"valid={valid_prediction} | "
                f"repo_modified={target_file_modified} | "
                f"answer_len={len(selected_prediction)} | "
                f"best_cand={best['candidate_index']} | "
                f"score={best['score_info']['score']:.2f}",
                flush=True,
            )

    print(f"\nSaved predictions to: {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
