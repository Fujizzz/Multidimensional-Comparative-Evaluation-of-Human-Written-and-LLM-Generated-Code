#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RepoEval Fixed Pipeline (LLM / RAG / Agent ready)
-------------------------------------------------
Fixed stages (same as we previously did manually):
  0) Verify standard prompt set (line count + repo distribution)
  1) Inference -> predictions/<RUN_ID>.jsonl + runs/<RUN_ID>/raw_outputs.jsonl + meta.json
  2) Official evaluation (EM/ES) -> runs/<RUN_ID>/official_scores.txt
  3) (Optional) Custom metrics -> runs/<RUN_ID>/custom_metrics.csv (placeholder)

You only change "method backend" for different approaches:
  - method=llm:      plain generation on the given prompt
  - method=rag_offline: same as llm here (prompt already contains retrieval, RG1 baseline)
  - method=rag_online:  TODO: implement retrieve_chunks() + build_rag_prompt()
  - method=agent:       TODO: implement agent_loop()

Backends:
  - hf_local: load local HF model (your DeepSeek path)
  - openai_compat: call OpenAI-compatible endpoint
  - external_jsonl: read already-generated outputs (e.g., Repoformer) from a jsonl and evaluate with our pipeline

Outputs are compatible with RepoCoder official scorer:
  predictions line format:
    {"prompt": ..., "choices":[{"text": <completion>}], "metadata": {...}}
  raw outputs line format:
    {"latency_sec":..., "raw_text":..., "clean_text":..., "metadata":...}

Usage examples:
  # Full run with local DeepSeek (your current baseline)
  python run_repoeval_fixed.py all \
    --prompt_set prompts/standard_api_rg1_ws20_ss2_allrepos_N200_seed42.jsonl \
    --method llm --backend hf_local \
    --model_path /root/autodl-tmp/LLM/Deepseek-coder \
    --max_new_tokens 128 --temperature 0.0

  # Evaluate an existing predictions file only (skip inference)
  python run_repoeval_fixed.py score --run_id <RUN_ID>

  # Use existing method output (e.g., Repoformer) without rewriting your scorer:
  python run_repoeval_fixed.py all \
    --method llm --backend external_jsonl \
    --prompt_set prompts/standard_api_rg1_ws20_ss2_allrepos_N200_seed42.jsonl \
    --external_predictions path/to/other_method_outputs.jsonl

Notes:
- Keep your standard prompt set fixed for fair comparison across models/methods.
- Only Step 1 changes (how completion is generated). Step 2/3 remain identical.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Repo list (line_and_api_level)
# -----------------------------
ALL_REPOS_DEFAULT = [
    "alibaba_FederatedScope",
    "awslabs_fortuna",
    "google_vizier",
    "huggingface_diffusers",
    "huggingface_evaluate",
    "nerfstudio-project_nerfstudio",
    "opendilab_ACE",
    "pytorch_rl",
]

# -----------------------------
# Helpers
# -----------------------------
def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_repo_from_metadata(md: Dict[str, Any]) -> str:
    if md.get("repo"):
        return md["repo"]
    if md.get("repo_name"):
        return md["repo_name"]
    tid = md.get("task_id", "")
    if "/" in tid:
        return tid.split("/")[0]
    return ""


def clean_completion(text: str) -> str:
    """Minimal clean: strip code fences + surrounding whitespace."""
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = "".join(parts[1:-1]).strip()
    return t.strip()


def count_repos_in_promptset(prompt_set_path: str) -> Counter:
    c = Counter()
    for r in stream_jsonl(prompt_set_path):
        md = r.get("metadata", {}) or {}
        repo = get_repo_from_metadata(md)
        c[repo] += 1
    return c


# -----------------------------
# Backends
# -----------------------------
class BackendBase:
    def generate(self, prompt: str) -> str:
        """Return raw_text."""
        raise NotImplementedError


class HFLocalBackend(BackendBase):
    def __init__(
        self,
        model_path: str,
        dtype: str = "float16",
        device: str = "cuda",
        use_fast: bool = False,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.torch = torch
        self.model_path = model_path
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, use_fast=use_fast
        )
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map=("cuda" if device == "cuda" else "cpu"),
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    @self_check
    def generate(self, prompt: str) -> str:
        raise RuntimeError("This method is patched below.")


# Python trick: avoid mypy linting; patch generate after class definition
def _hf_generate(self: HFLocalBackend, prompt: str, max_new_tokens: int, temperature: float) -> str:
    torch = self.torch
    inputs = self.tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(self.device)
    attn = inputs.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(self.device)

    do_sample = temperature > 0.0

    with torch.inference_mode():
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=(temperature if do_sample else None),
            top_p=(0.95 if do_sample else None),
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[1] :]
    return self.tok.decode(new_tokens, skip_special_tokens=True)


# attach a correct generate signature wrapper
def hf_backend_generate(self: HFLocalBackend, prompt: str) -> str:
    # placeholder; real generation is done in PipelineRunner where we pass max_new_tokens/temperature
    raise RuntimeError("HFLocalBackend.generate should not be called directly; use PipelineRunner._gen_raw()")


HFLocalBackend.generate = hf_backend_generate  # type: ignore


class OpenAICompatBackend(BackendBase):
    def __init__(self, base_url: str, model: str, api_key: str, timeout_sec: int = 180):
        import requests

        self.requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": "Output only the completion text."},
                {"role": "user", "content": prompt},
            ],
        }
        r = self.requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


class ExternalJSONLBackend(BackendBase):
    """
    Use existing method outputs (e.g., Repoformer / agent framework) without rewriting our scorer.

    external_predictions must contain a per-sample key that we can map to our promptset sample:
      Prefer: metadata.task_id
      Accept: task_id at top-level
      And predicted text in one of:
        - choices[0].text
        - pred / prediction / output fields
    """

    def __init__(self, external_predictions_path: str):
        self.map: Dict[str, str] = {}
        for r in stream_jsonl(external_predictions_path):
            md = r.get("metadata", {}) or {}
            tid = md.get("task_id") or r.get("task_id")
            if not tid:
                continue

            txt = ""
            if isinstance(r.get("choices"), list) and r["choices"]:
                c0 = r["choices"][0]
                if isinstance(c0, dict) and "text" in c0:
                    txt = c0["text"]
            if not txt:
                for k in ("pred", "prediction", "output", "text"):
                    if isinstance(r.get(k), str) and r.get(k):
                        txt = r[k]
                        break
            if txt:
                self.map[str(tid)] = txt

    def generate(self, prompt: str) -> str:
        raise RuntimeError("ExternalJSONLBackend.generate should not be called directly; use get_by_task_id().")

    def get_by_task_id(self, task_id: str) -> str:
        if task_id not in self.map:
            raise KeyError(f"task_id not found in external predictions: {task_id}")
        return self.map[task_id]


# -----------------------------
# Methods (LLM / RAG / Agent)
# -----------------------------
def build_prompt_for_method(sample: Dict[str, Any], method: str) -> str:
    """
    Where to modify for RAG/Agent:
      - rag_offline: use sample['prompt'] (already contains retrieval snippets from RG1)
      - rag_online: implement retrieve_chunks(sample) and merge into prompt
      - agent: implement agent_loop(sample)
    """
    base_prompt = sample["prompt"]

    if method in ("llm", "rag_offline"):
        return base_prompt

    if method == "rag_online":
        # TODO: implement your own retriever here (BM25/embedding/tree-sitter etc.)
        # chunks = retrieve_chunks(sample, top_k=20)
        # return build_rag_prompt(base_prompt, chunks)
        raise NotImplementedError("rag_online is reserved. Implement retrieve_chunks() + build_rag_prompt().")

    if method == "agent":
        # TODO: implement multi-round tool loop here:
        # return agent_loop(sample)
        raise NotImplementedError("agent is reserved. Implement agent_loop(sample).")

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Runner
# -----------------------------
@dataclass
class RunConfig:
    prompt_set: str
    method: str
    backend: str

    # hf_local
    model_path: str = ""
    dtype: str = "float16"
    device: str = "cuda"
    use_fast: bool = False

    # openai_compat
    base_url: str = ""
    model_name: str = ""
    api_key_env: str = "OPENAI_API_KEY"

    # external_jsonl
    external_predictions: str = ""

    # generation params
    max_new_tokens: int = 128
    temperature: float = 0.0

    # run id
    run_id: str = ""


class PipelineRunner:
    def __init__(self, cfg: RunConfig, repos: List[str]):
        self.cfg = cfg
        self.repos = repos

        # auto run_id (avoid duplicates)
        if not self.cfg.run_id:
            base = Path(self.cfg.model_path).name if self.cfg.model_path else (self.cfg.model_name or self.cfg.backend)
            tag = now_tag()
            self.cfg.run_id = f"{base}__{self.cfg.method}__{Path(self.cfg.prompt_set).stem}__t{self.cfg.temperature}__max{self.cfg.max_new_tokens}__{tag}"

        self.run_dir = ensure_dir(Path("runs") / self.cfg.run_id)
        ensure_dir(Path("predictions"))
        ensure_dir(Path("prompts"))

        self.pred_path = Path("predictions") / f"{self.cfg.run_id}.jsonl"
        self.raw_path = self.run_dir / "raw_outputs.jsonl"
        self.meta_path = self.run_dir / "meta.json"
        self.official_path = self.run_dir / "official_scores.txt"
        self.custom_metrics_path = self.run_dir / "custom_metrics.csv"

        self.backend_obj: Optional[BackendBase] = None
        self.external_backend_obj: Optional[ExternalJSONLBackend] = None

    def preflight(self) -> None:
        # verify prompt set exists
        if not Path(self.cfg.prompt_set).exists():
            raise FileNotFoundError(f"prompt_set not found: {self.cfg.prompt_set}")

        # quick repo distribution print
        c = count_repos_in_promptset(self.cfg.prompt_set)
        total = sum(c.values())
        print("[preflight] prompt_set:", self.cfg.prompt_set)
        print("[preflight] total samples:", total)
        print("[preflight] repo distribution:")
        for k, v in sorted(c.items()):
            if k:
                print("  ", k, v)

    def _init_backend(self) -> None:
        if self.cfg.backend == "hf_local":
            if not self.cfg.model_path:
                raise ValueError("--model_path required for hf_local backend")
            # backend created, but generation done via _gen_raw() to pass parameters
            self.backend_obj = HFLocalBackend(
                model_path=self.cfg.model_path,
                dtype=self.cfg.dtype,
                device=self.cfg.device,
                use_fast=self.cfg.use_fast,
            )

        elif self.cfg.backend == "openai_compat":
            api_key = os.environ.get(self.cfg.api_key_env, "")
            if not (self.cfg.base_url and self.cfg.model_name):
                raise ValueError("--base_url and --model_name required for openai_compat backend")
            self.backend_obj = OpenAICompatBackend(
                base_url=self.cfg.base_url, model=self.cfg.model_name, api_key=api_key
            )

        elif self.cfg.backend == "external_jsonl":
            if not self.cfg.external_predictions:
                raise ValueError("--external_predictions required for external_jsonl backend")
            self.external_backend_obj = ExternalJSONLBackend(self.cfg.external_predictions)

        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

    def _gen_raw(self, prompt: str, task_id: Optional[str]) -> str:
        # hf_local special path (need max_new_tokens/temperature)
        if self.cfg.backend == "hf_local":
            assert isinstance(self.backend_obj, HFLocalBackend)
            return _hf_generate(self.backend_obj, prompt, self.cfg.max_new_tokens, self.cfg.temperature)

        if self.cfg.backend == "openai_compat":
            assert self.backend_obj is not None
            return self.backend_obj.generate(prompt)

        if self.cfg.backend == "external_jsonl":
            assert self.external_backend_obj is not None
            if not task_id:
                raise ValueError("external_jsonl requires metadata.task_id in prompt_set")
            return self.external_backend_obj.get_by_task_id(task_id)

        raise RuntimeError("backend not initialized")

    def infer(self) -> None:
        self.preflight()
        self._init_backend()

        # write meta
        meta = {
            "run_id": self.cfg.run_id,
            "prompt_set": self.cfg.prompt_set,
            "method": self.cfg.method,
            "backend": self.cfg.backend,
            "model_path": self.cfg.model_path,
            "base_url": self.cfg.base_url,
            "model_name": self.cfg.model_name,
            "external_predictions": self.cfg.external_predictions,
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "time": now_tag(),
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # run inference
        n = 0
        t_start = time.time()
        with open(self.pred_path, "w", encoding="utf-8") as fp, open(self.raw_path, "w", encoding="utf-8") as fr:
            for sample in stream_jsonl(self.cfg.prompt_set):
                md = sample.get("metadata", {}) or {}
                task_id = md.get("task_id")  # important for external_jsonl mapping
                prompt = build_prompt_for_method(sample, self.cfg.method)

                t0 = time.time()
                raw = self._gen_raw(prompt, task_id)
                latency = time.time() - t0
                comp = clean_completion(raw)

                fp.write(json.dumps(
                    {"prompt": sample["prompt"], "choices": [{"text": comp}], "metadata": md},
                    ensure_ascii=False
                ) + "\n")

                fr.write(json.dumps(
                    {"latency_sec": latency, "raw_text": raw, "clean_text": comp, "metadata": md},
                    ensure_ascii=False
                ) + "\n")

                n += 1
                if n % 50 == 0:
                    print(f"[infer] {n} done...")

        print(f"[infer] done. n={n} elapsed={time.time()-t_start:.1f}s")
        print("[infer] predictions:", self.pred_path)
        print("[infer] raw outputs :", self.raw_path)
        print("[infer] meta        :", self.meta_path)

    def score_official(self) -> None:
        # RepoCoder official scorer (prints to stdout). We capture stdout into file.
        from compute_score import compute_score_by_repo_with_metadata
        from utils import Tools

        if not self.pred_path.exists():
            raise FileNotFoundError(f"predictions not found: {self.pred_path}")

        lines = Tools.load_jsonl(str(self.pred_path))

        with open(self.official_path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                print("RUN_ID =", self.cfg.run_id)
                print("PRED   =", str(self.pred_path))
                print("PROMPT_SET =", self.cfg.prompt_set)
                print("\n=== EM (pass@1) ===")
                compute_score_by_repo_with_metadata(self.repos, lines, "EM", passk=1)
                print("\n=== ES (pass@1) ===")
                compute_score_by_repo_with_metadata(self.repos, lines, "ES", passk=1)

        print("[score] saved:", self.official_path)

    def custom_metrics_placeholder(self) -> None:
        """
        Interface you extend later:
        - Read runs/<RUN_ID>/raw_outputs.jsonl
        - Write runs/<RUN_ID>/custom_metrics.csv
        """
        ensure_dir(self.custom_metrics_path.parent)
        if not self.raw_path.exists():
            print("[custom] raw_outputs not found, skip:", self.raw_path)
            return

        # Placeholder CSV with minimal fields. Replace with your own metrics later.
        with open(self.custom_metrics_path, "w", encoding="utf-8") as f:
            f.write("task_id,repo,latency_sec,raw_len,clean_len\n")
            for r in stream_jsonl(str(self.raw_path)):
                md = r.get("metadata", {}) or {}
                task_id = md.get("task_id", "")
                repo = get_repo_from_metadata(md)
                latency = r.get("latency_sec", 0.0)
                raw_len = len(r.get("raw_text", "") or "")
                clean_len = len(r.get("clean_text", "") or "")
                f.write(f"{task_id},{repo},{latency},{raw_len},{clean_len}\n")

        print("[custom] placeholder metrics saved:", self.custom_metrics_path)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "score", "custom", "all"], help="Which stage to run.")
    ap.add_argument("--prompt_set", required=True, help="Standard prompt set jsonl (fixed benchmark input).")
    ap.add_argument("--method", choices=["llm", "rag_offline", "rag_online", "agent"], default="llm")
    ap.add_argument("--backend", choices=["hf_local", "openai_compat", "external_jsonl"], default="hf_local")

    ap.add_argument("--repos", nargs="*", default=ALL_REPOS_DEFAULT, help="Repo list for official scoring.")
    ap.add_argument("--run_id", default="", help="Optional run_id. If empty, auto-generated with timestamp.")

    # hf_local
    ap.add_argument("--model_path", default="", help="Local HF model path (e.g., /root/autodl-tmp/LLM/Deepseek-coder)")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--use_fast", action="store_true", help="Use fast tokenizer if available (default False).")

    # openai_compat
    ap.add_argument("--base_url", default="")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")

    # external_jsonl
    ap.add_argument("--external_predictions", default="", help="External method output jsonl to map by task_id.")

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)

    return ap.parse_args()


def main():
    args = parse_args()

    cfg = RunConfig(
        prompt_set=args.prompt_set,
        method=args.method,
        backend=args.backend,
        model_path=args.model_path,
        dtype=args.dtype,
        device=args.device,
        use_fast=args.use_fast,
        base_url=args.base_url,
        model_name=args.model_name,
        api_key_env=args.api_key_env,
        external_predictions=args.external_predictions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        run_id=args.run_id,
    )

    runner = PipelineRunner(cfg, repos=args.repos)

    if args.stage == "infer":
        runner.infer()

    elif args.stage == "score":
        # assumes predictions/<run_id>.jsonl exists, so run_id must be given or inferred
        if not cfg.run_id:
            raise ValueError("score stage requires --run_id (to locate predictions/<run_id>.jsonl).")
        runner.score_official()

    elif args.stage == "custom":
        if not cfg.run_id:
            raise ValueError("custom stage requires --run_id (to locate runs/<run_id>/raw_outputs.jsonl).")
        runner.custom_metrics_placeholder()

    elif args.stage == "all":
        runner.infer()
        runner.score_official()
        runner.custom_metrics_placeholder()

    else:
        raise ValueError("unknown stage")


if __name__ == "__main__":
    main()