#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

CFC_CANDIDATES = ["<cfc_info>", "<cc>"]
EOF_CANDIDATES = ["<end_rc>", "<eof>"]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def resolve_repoformer_tokens(tokenizer) -> Tuple[str, str, int, int]:
    def _find(candidates: List[str]) -> Tuple[str, int]:
        vocab = tokenizer.get_vocab()
        for tok in candidates:
            if tok in vocab:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != tokenizer.unk_token_id:
                    return tok, tid
        for tok in candidates:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                return tok, tid
        raise ValueError(
            f"Failed to find any token from candidates={candidates}. "
            f"Available special tokens: {tokenizer.all_special_tokens}"
        )

    cfc_token, cfc_id = _find(CFC_CANDIDATES)
    eof_token, eof_id = _find(EOF_CANDIDATES)
    return cfc_token, eof_token, cfc_id, eof_id


def load_tokenizer(tokenizer_name: str):
    local_only = os.path.exists(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cfc_token, eof_token, cfc_id, eof_id = resolve_repoformer_tokens(tokenizer)
    print(f"[TOKENS] cfc_token={cfc_token} (id={cfc_id}), eof_token={eof_token} (id={eof_id})")
    return tokenizer, cfc_token, eof_token, cfc_id, eof_id


def extract_crossfile_text(entry: Dict[str, Any]) -> Optional[str]:
    cfc = entry.get("crossfile_context")
    if cfc is None:
        return None
    if isinstance(cfc, str):
        return cfc
    if isinstance(cfc, dict):
        return cfc.get("text")
    return None


def _truncate_by_tokens(tokenizer, text: str, max_tokens: int, from_right: bool = False) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if from_right:
        ids = ids[-max_tokens:]
    else:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids)


def prepare_prompt(
    tokenizer,
    left_cxt: str,
    right_cxt: Optional[str],
    crossfile_cxt: Optional[str],
    mode: str,
    max_seq_length: int,
    gen_length: int,
    right_context_length: int,
    cfc_seq_length: int,
    cfc_token: str,
    eof_token: str,
) -> str:
    if mode == "codelm_leftright_context":
        left_budget = max_seq_length - gen_length - right_context_length
        left_cxt_truncated = _truncate_by_tokens(tokenizer, left_cxt, left_budget, from_right=True)
        right_cxt_truncated = _truncate_by_tokens(tokenizer, right_cxt or "", right_context_length, from_right=False)
        return f"{FIM_PREFIX}{left_cxt_truncated}{FIM_SUFFIX}{right_cxt_truncated}{eof_token}{FIM_MIDDLE}"

    if mode == "codelm_right_cfc_left":
        if crossfile_cxt is None:
            raise ValueError("crossfile_cxt is required for codelm_right_cfc_left")
        left_budget = max_seq_length - gen_length - right_context_length - cfc_seq_length
        left_cxt_truncated = _truncate_by_tokens(tokenizer, left_cxt, left_budget, from_right=True)
        right_cxt_truncated = _truncate_by_tokens(tokenizer, right_cxt or "", right_context_length, from_right=False)
        crossfile_cxt_truncated = _truncate_by_tokens(tokenizer, "\n\n" + crossfile_cxt, cfc_seq_length, from_right=False)
        return (
            f"{FIM_PREFIX}{left_cxt_truncated}{FIM_SUFFIX}{right_cxt_truncated}"
            f"{eof_token}{cfc_token}{crossfile_cxt_truncated}{FIM_MIDDLE}"
        )

    raise NotImplementedError(f"Unsupported mode: {mode}")



def build_examples(
    prompt_file: Path,
    tokenizer,
    gen_length: int,
    max_seq_length: int,
    right_context_length: int,
    cfc_seq_length: int,
    cfc_token: str,
    eof_token: str,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            left_cxt = entry["prompt"]
            right_cxt = entry.get("right_context", "")
            crossfile_cxt = extract_crossfile_text(entry)
            entry["llm_prompt_lrcontext"] = prepare_prompt(
                tokenizer=tokenizer,
                left_cxt=left_cxt,
                right_cxt=right_cxt,
                crossfile_cxt=crossfile_cxt,
                mode="codelm_leftright_context",
                max_seq_length=max_seq_length,
                gen_length=gen_length,
                right_context_length=right_context_length,
                cfc_seq_length=cfc_seq_length,
                cfc_token=cfc_token,
                eof_token=eof_token,
            )
            entry["llm_prompt_right_cfc_left"] = None
            if crossfile_cxt:
                entry["llm_prompt_right_cfc_left"] = prepare_prompt(
                    tokenizer=tokenizer,
                    left_cxt=left_cxt,
                    right_cxt=right_cxt,
                    crossfile_cxt=crossfile_cxt,
                    mode="codelm_right_cfc_left",
                    max_seq_length=max_seq_length,
                    gen_length=gen_length,
                    right_context_length=right_context_length,
                    cfc_seq_length=cfc_seq_length,
                    cfc_token=cfc_token,
                    eof_token=eof_token,
                )
            examples.append(entry)
    return examples



def compute_selective_probs(
    model_path: str,
    tokenizer,
    examples: List[Dict[str, Any]],
    target_token_id: int,
    target_token_name: str = "decision_target",
    batch_size: int = 4,
    dtype: str = "float16",
) -> List[float]:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    local_only = os.path.exists(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=local_only,
    )
    model.eval()

    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != vocab_size:
        print(f"[WARN] tokenizer size ({len(tokenizer)}) != model vocab size ({vocab_size}); resizing embeddings.")
        model.resize_token_embeddings(len(tokenizer))
        vocab_size = model.get_input_embeddings().weight.shape[0]

    if target_token_id is None or target_token_id < 0 or target_token_id >= vocab_size:
        raise ValueError(
            f"Resolved target token id is invalid: name={target_token_name}, id={target_token_id}, vocab_size={vocab_size}. "
            f"Tokenizer/model mismatch."
        )

    prompts = [e["llm_prompt_lrcontext"].replace(FIM_MIDDLE, "") for e in examples]
    probs: List[float] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Selective decision"):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            max_id = int(enc["input_ids"].max().item())
            min_id = int(enc["input_ids"].min().item())
            if min_id < 0 or max_id >= vocab_size:
                raise ValueError(
                    f"Encoded input ids out of vocab range: min={min_id}, max={max_id}, vocab_size={vocab_size}."
                )

            emb_device = model.get_input_embeddings().weight.device
            enc = {k: v.to(emb_device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits
            last_positions = enc["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            next_token_logits = logits[batch_indices, last_positions, :].float()
            batch_probs = torch.softmax(next_token_logits, dim=-1)[:, target_token_id]
            probs.extend(batch_probs.detach().cpu().tolist())

    try:
        model.cpu()
    except Exception:
        pass
    del model
    cleanup_cuda()
    return probs



def save_decisions(task_out: Path, examples: List[Dict[str, Any]], selective_probs: List[float]) -> None:
    with open(task_out / "decision_probs.jsonl", "w", encoding="utf-8") as f:
        for entry, prob in zip(examples, selective_probs):
            task_id = entry.get("metadata", {}).get("task_id", entry.get("task_id"))
            f.write(json.dumps({"task_id": task_id, "selective_prob": prob}, ensure_ascii=False) + "\n")



def load_decisions(task_out: Path) -> List[float]:
    path = task_out / "decision_probs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Decision file not found: {path}")
    probs: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            probs.append(float(json.loads(line)["selective_prob"]))
    return probs



def generate_task(
    model_path: str,
    tokenizer_name: str,
    examples: List[Dict[str, Any]],
    output_dir: Path,
    gen_length: int,
    threshold: float,
    selective_probs: List[float],
    decision_mode: str = "paper",
    gpu_memory_utilization: float = 0.45,
    vllm_dtype: str = "auto",
    max_model_len: Optional[int] = None,
    decision_target_token: Optional[str] = None,
    decision_target_token_id: Optional[int] = None,
):
    if len(examples) != len(selective_probs):
        raise ValueError(f"Length mismatch: len(examples)={len(examples)} vs len(selective_probs)={len(selective_probs)}")

    llm_kwargs = dict(
        model=model_path,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        dtype=vllm_dtype,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if max_model_len is not None and max_model_len > 0:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=gen_length)

    preds = []
    raws = []
    rag_count = 0

    iterator = zip(examples, selective_probs)
    for entry, selective_prob in tqdm(list(iterator), total=len(examples), desc=f"Generate {output_dir.name}"):
        use_crossfile = False
        if entry.get("llm_prompt_right_cfc_left") is not None:
            if decision_mode == "paper":
                use_crossfile = selective_prob > threshold
            elif decision_mode == "public_script":
                use_crossfile = not (selective_prob > threshold)
            else:
                raise ValueError(f"Unsupported decision_mode: {decision_mode}")

        if use_crossfile and entry.get("llm_prompt_right_cfc_left") is not None:
            used_prompt_type = "right_cfc_left"
            used_prompt = entry["llm_prompt_right_cfc_left"]
            rag_count += 1
        else:
            used_prompt_type = "lrcontext"
            used_prompt = entry["llm_prompt_lrcontext"]

        pred = llm.generate(used_prompt, sampling_params, use_tqdm=False)
        pred_text = pred[0].outputs[0].text
        task_id = entry.get("metadata", {}).get("task_id", entry.get("task_id"))

        trace = {
            "selective_prob": selective_prob,
            "selective_threshold": threshold,
            "selected_prompt_type": used_prompt_type,
            "use_crossfile": use_crossfile,
            "decision_mode": decision_mode,
        }
        preds.append({"task_id": task_id, "pred": pred_text, **trace})
        raws.append({
            "task_id": task_id,
            "metadata": entry.get("metadata", {}),
            "prompt": entry.get("prompt", ""),
            "right_context": entry.get("right_context", ""),
            "crossfile_context": entry.get("crossfile_context", None),
            "llm_prompt_lrcontext": entry.get("llm_prompt_lrcontext", None),
            "llm_prompt_right_cfc_left": entry.get("llm_prompt_right_cfc_left", None),
            "generated_code": pred_text,
            **trace,
        })

    with open(output_dir / "prediction.jsonl", "w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "raw_generation.jsonl", "w", encoding="utf-8") as f:
        for row in raws:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "count": len(examples),
        "threshold": threshold,
        "rag_count": rag_count,
        "rag_ratio": rag_count / max(len(examples), 1),
        "avg_selective_prob": float(sum(selective_probs) / max(len(selective_probs), 1)),
        "decision_mode": decision_mode,
        "gpu_memory_utilization": gpu_memory_utilization,
        "vllm_dtype": vllm_dtype,
        "max_model_len": max_model_len,
        "decision_target_token": decision_target_token,
        "decision_target_token_id": decision_target_token_id,
    }
    with open(output_dir / "selection_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[DONE] {output_dir.name}: rag_ratio={stats['rag_ratio']:.4f}, avg_selective_prob={stats['avg_selective_prob']:.6f}")

    del llm
    cleanup_cuda()



def run_decision_stage(args) -> None:
    tokenizer_name = args.tokenizer_name or args.model_path
    tokenizer, cfc_token, eof_token, cfc_token_id, eof_token_id = load_tokenizer(tokenizer_name)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        gen_length = 256 if task == "function_completion" else 50
        threshold = args.function_threshold if task == "function_completion" else args.default_threshold
        prompt_file = Path(args.data_root) / f"python_{task}_{args.ranker}_rg1.jsonl"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        examples = build_examples(
            prompt_file=prompt_file,
            tokenizer=tokenizer,
            gen_length=gen_length,
            max_seq_length=args.max_seq_length,
            right_context_length=args.right_context_length,
            cfc_seq_length=args.cfc_seq_length,
            cfc_token=cfc_token,
            eof_token=eof_token,
        )

        if args.decision_mode == "paper":
            decision_target_token = cfc_token
            decision_target_token_id = cfc_token_id
        elif args.decision_mode == "public_script":
            decision_target_token = eof_token
            decision_target_token_id = eof_token_id
        else:
            raise ValueError(f"Unsupported decision_mode: {args.decision_mode}")

        print(f"[INFO] Building selective decisions for {task} from {prompt_file} | target={decision_target_token} (id={decision_target_token_id})")
        selective_probs = compute_selective_probs(
            model_path=args.model_path,
            tokenizer=tokenizer,
            examples=examples,
            target_token_id=decision_target_token_id,
            target_token_name=decision_target_token,
            batch_size=args.decision_batch_size,
            dtype=args.decision_dtype,
        )

        task_out = output_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        meta = {
            "task": task,
            "model_path": args.model_path,
            "tokenizer_name": tokenizer_name,
            "prompt_file": str(prompt_file),
            "ranker": args.ranker,
            "max_seq_length": args.max_seq_length,
            "right_context_length": args.right_context_length,
            "cfc_seq_length": args.cfc_seq_length,
            "gen_length": gen_length,
            "threshold": threshold,
            "decision_batch_size": args.decision_batch_size,
            "decision_dtype": args.decision_dtype,
            "decision_mode": args.decision_mode,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "vllm_dtype": args.vllm_dtype,
            "max_model_len": args.max_model_len,
            "resolved_cfc_token": cfc_token,
            "resolved_eof_token": eof_token,
            "resolved_cfc_token_id": cfc_token_id,
            "resolved_eof_token_id": eof_token_id,
            "decision_target_token": decision_target_token,
            "decision_target_token_id": decision_target_token_id,
        }
        with open(task_out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        save_decisions(task_out, examples, selective_probs)
        cleanup_cuda()

        if args.stage == "all":
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage", "generate",
                "--model_path", args.model_path,
                "--data_root", args.data_root,
                "--output_root", args.output_root,
                "--tasks", task,
                "--ranker", args.ranker,
                "--max_seq_length", str(args.max_seq_length),
                "--right_context_length", str(args.right_context_length),
                "--cfc_seq_length", str(args.cfc_seq_length),
                "--default_threshold", str(args.default_threshold),
                "--function_threshold", str(args.function_threshold),
                "--decision_mode", args.decision_mode,
                "--gpu_memory_utilization", str(args.gpu_memory_utilization),
                "--vllm_dtype", args.vllm_dtype,
            ]
            if args.tokenizer_name is not None:
                cmd.extend(["--tokenizer_name", args.tokenizer_name])
            if args.max_model_len is not None:
                cmd.extend(["--max_model_len", str(args.max_model_len)])
            print(f"[INFO] Launching fresh generation process for task={task}")
            subprocess.run(cmd, check=True, env=os.environ.copy())



def run_generate_stage(args) -> None:
    tokenizer_name = args.tokenizer_name or args.model_path
    tokenizer, cfc_token, eof_token, cfc_token_id, eof_token_id = load_tokenizer(tokenizer_name)
    output_root = Path(args.output_root)

    for task in args.tasks:
        gen_length = 256 if task == "function_completion" else 50
        threshold = args.function_threshold if task == "function_completion" else args.default_threshold
        prompt_file = Path(args.data_root) / f"python_{task}_{args.ranker}_rg1.jsonl"
        task_out = output_root / task
        task_out.mkdir(parents=True, exist_ok=True)

        examples = build_examples(
            prompt_file=prompt_file,
            tokenizer=tokenizer,
            gen_length=gen_length,
            max_seq_length=args.max_seq_length,
            right_context_length=args.right_context_length,
            cfc_seq_length=args.cfc_seq_length,
            cfc_token=cfc_token,
            eof_token=eof_token,
        )
        selective_probs = load_decisions(task_out)
        computed_max_model_len = args.max_model_len or min(8192, args.max_seq_length + gen_length + 16)

        generate_task(
            model_path=args.model_path,
            tokenizer_name=tokenizer_name,
            examples=examples,
            output_dir=task_out,
            gen_length=gen_length,
            threshold=threshold,
            selective_probs=selective_probs,
            decision_mode=args.decision_mode,
            gpu_memory_utilization=args.gpu_memory_utilization,
            vllm_dtype=args.vllm_dtype,
            max_model_len=computed_max_model_len,
            decision_target_token=(eof_token if args.decision_mode == "public_script" else cfc_token),
            decision_target_token_id=(eof_token_id if args.decision_mode == "public_script" else cfc_token_id),
        )



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "decision", "generate"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tasks", nargs="+", default=["line_completion", "api_completion", "function_completion"])
    parser.add_argument("--ranker", default="sparse", choices=["sparse", "unixcoder"])
    parser.add_argument("--tokenizer_name", default=None, help="Defaults to model_path. Use a local path for offline runs.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--right_context_length", type=int, default=512)
    parser.add_argument("--cfc_seq_length", type=int, default=512)
    parser.add_argument("--default_threshold", type=float, default=0.2)
    parser.add_argument("--function_threshold", type=float, default=0.15)
    parser.add_argument("--decision_batch_size", type=int, default=4)
    parser.add_argument("--decision_dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--decision_mode", default="paper", choices=["paper", "public_script"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--vllm_dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--max_model_len", type=int, default=None, help="For vLLM. Defaults to max_seq_length + gen_length + 16.")
    return parser.parse_args()



def main():
    args = parse_args()
    if args.stage in {"all", "decision"}:
        run_decision_stage(args)
    else:
        run_generate_stage(args)


if __name__ == "__main__":
    main()
