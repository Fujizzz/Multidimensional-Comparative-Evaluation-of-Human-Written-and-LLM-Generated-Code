#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import builtins
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ============================================================
# RepoEval quality evaluation suite
# - Official-aligned EM / ES for correctness
# - Extended quality metrics for patch/completion quality
# - Single-file, local-only, with pure-Python fallbacks
# ============================================================

CODE_KEYS = [
    "pred", "prediction", "completion", "generated_code", "generated_text",
    "code", "output", "text", "response", "content", "raw_generation",
    "raw_output",
]

# Coarse-grained security / risk patterns
DANGEROUS_PATTERNS = {
    "dynamic_exec": [
        r"\beval\s*\(", r"\bexec\s*\(", r"\bcompile\s*\(",
        r"\b__import__\s*\(", r"\bglobals\s*\(", r"\blocals\s*\(",
    ],
    "filesystem_write": [
        r"\bopen\s*\([^\)]*,\s*[\"'](?:w|a|x|wb|ab|xb)",
        r"\bos\.remove\s*\(", r"\bos\.unlink\s*\(", r"\bos\.rmdir\s*\(",
        r"\bshutil\.rmtree\s*\(", r"\bshutil\.move\s*\(",
        r"\bos\.rename\s*\(", r"\bos\.replace\s*\(",
    ],
    "process": [
        r"\bos\.system\s*\(", r"\bos\.popen\s*\(", r"\bsubprocess\.Popen\s*\(",
        r"\bsubprocess\.run\s*\(", r"\bsubprocess\.call\s*\(",
        r"\bsubprocess\.check_output\s*\(", r"\bsubprocess\.check_call\s*\(",
    ],
    "deserialization": [
        r"\bpickle\.loads\s*\(", r"\bmarshal\.loads\s*\(", r"\byaml\.load\s*\(",
    ],
    "network": [
        r"\brequests\.", r"\burllib\.", r"\bhttp\.", r"\bftplib\.", r"\bsocket\.",
    ],
    "native_bridge": [r"\bctypes\.", r"\bcffi\."]
}

SENSITIVE_IMPORTS = {
    "subprocess", "socket", "pickle", "marshal", "ctypes", "requests", "urllib",
    "ftplib", "http", "yaml", "shutil", "os",
}

PY_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.IfExp,
    ast.With,
    ast.AsyncWith,
    ast.ExceptHandler,
    ast.Assert,
    ast.Match,
)

PY_BUILTINS = set(dir(builtins))
COMMON_SHORT_IDS = {"i", "j", "k", "x", "y", "z", "n", "m", "p", "q", "r", "s", "t"}
STOPWORDS = {
    "self", "cls", "true", "false", "none", "and", "or", "not", "in", "is", "for",
    "if", "else", "elif", "try", "except", "while", "return", "lambda", "with", "as",
}

try:
    from radon.complexity import cc_visit  # type: ignore
    from radon.metrics import h_visit, mi_visit  # type: ignore
    HAS_RADON = True
except Exception:
    HAS_RADON = False


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL {path} line {lineno}: {e}")
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_task_id(obj: Dict[str, Any]) -> Optional[str]:
    md = obj.get("metadata") or {}
    for key in ["task_id", "id", "problem_id", "name"]:
        if obj.get(key) is not None:
            return str(obj[key])
    for key in ["task_id", "id", "problem_id", "name"]:
        if md.get(key) is not None:
            return str(md[key])
    return None


def get_first(obj: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split('.'):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def extract_candidates(obj: Dict[str, Any], preferred: Optional[str] = None) -> List[str]:
    if preferred:
        val = _get_by_path(obj, preferred)
        if val is not None:
            if isinstance(val, list):
                return [str(x) for x in val]
            return [str(val)]

    raw_candidates: List[str] = []
    for key in ["clean_text", "raw_text"]:
        val = obj.get(key)
        if isinstance(val, str):
            raw_candidates.append(val)
    if raw_candidates:
        return raw_candidates

    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        out = []
        for ch in choices:
            if isinstance(ch, dict):
                for key in ["text", "content", "code", "completion", "generated_text"]:
                    if ch.get(key) is not None:
                        out.append(str(ch[key]))
                        break
            elif isinstance(ch, str):
                out.append(ch)
        if out:
            return out

    val = get_first(obj, CODE_KEYS, "")
    if isinstance(val, list):
        return [str(x) for x in val]
    if val is None:
        return [""]
    return [str(val)]


def strip_code_fence(text: str) -> Tuple[str, str]:
    text = text or ""
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)?\s*(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    if blocks:
        best = max(blocks, key=lambda x: len(x.strip()))
        return best.strip(), "fenced"
    return text.strip(), "as_is"


def infer_language(task: Dict[str, Any], code: str = "") -> str:
    md = task.get("metadata") or {}
    lang_candidates = [
        task.get("language"), task.get("lang"), md.get("language"),
        md.get("lang"), md.get("programming_language"),
    ]
    for x in lang_candidates:
        if isinstance(x, str) and x.strip():
            return x.strip().lower()

    ext = get_first(task, ["ext", "file_ext"], None)
    if ext is None:
        ext = md.get("ext")
    if isinstance(ext, str):
        ext = ext.lower().lstrip(".")
        ext_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "java": "java", "cs": "csharp", "cpp": "cpp", "c": "c",
        }
        if ext in ext_map:
            return ext_map[ext]

    filepath = get_first(task, ["filepath", "file_path", "path"], "") or md.get("filepath") or md.get("path") or ""
    if isinstance(filepath, str):
        suffix = Path(filepath).suffix.lower()
        suffix_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".cs": "csharp", ".cpp": "cpp", ".cc": "cpp",
            ".cxx": "cpp", ".c": "c",
        }
        if suffix in suffix_map:
            return suffix_map[suffix]

    low = (code or "").lower()
    if "def " in low or "import " in low or "from " in low:
        return "python"
    if "function " in low or "console.log" in low or "=>" in low:
        return "javascript"
    return "unknown"


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def mean_or_none(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


# ============================================================
# Official RepoCoder / RepoEval EM & ES
# ============================================================

def official_normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def levenshtein_distance(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def official_compute_em(target: str, predictions: List[str], passk: int = 1) -> int:
    target_lines = official_normalize_lines(target)
    target_len = len(target_lines)
    for prediction in predictions[:passk]:
        pred_lines = official_normalize_lines(prediction)[:target_len]
        if len(target_lines) != len(pred_lines):
            continue
        if target_lines == pred_lines:
            return 1
    return 0


def official_compute_es(target: str, predictions: List[str], passk: int = 1) -> float:
    target_lines = official_normalize_lines(target)
    target_str = "\n".join(target_lines)
    target_len = len(target_lines)
    scores = []
    for prediction in predictions[:passk]:
        pred_lines = official_normalize_lines(prediction)[:target_len]
        pred_str = "\n".join(pred_lines)
        denom = max(len(target_str), len(pred_str))
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(1.0 - levenshtein_distance(target_str, pred_str) / denom)
    return max(scores) if scores else 0.0


# ============================================================
# Context reconstruction and lexical helpers
# ============================================================

def stringify_crossfile_context(cfc: Any) -> str:
    if cfc is None:
        return ""
    if isinstance(cfc, str):
        return cfc
    if isinstance(cfc, list):
        parts = []
        for item in cfc:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = get_first(item, ["text", "content", "code"], "")
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts)
    return str(cfc)


def reconstruct_full_code(task: Dict[str, Any], completion: str) -> str:
    prompt = str(task.get("prompt") or "")
    right_context = str(task.get("right_context") or "")
    return prompt + (completion or "") + right_context


def extract_identifiers(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text or "")


def normalize_identifier_list(ids: List[str]) -> List[str]:
    out = []
    for x in ids:
        low = x.lower()
        if low in STOPWORDS:
            continue
        out.append(low)
    return out


def shannon_entropy(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


# ============================================================
# Quality metrics
# ============================================================

def compute_size_metrics(code: str) -> Dict[str, Any]:
    lines = (code or "").splitlines()
    stripped = [x.rstrip("\n") for x in lines]
    non_empty = [x for x in stripped if x.strip()]
    lengths = [len(x) for x in stripped]
    tokens = re.findall(r"\S+", code or "")
    return {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
        "char_count": len(code or ""),
        "token_count": len(tokens),
        "avg_line_length": mean_or_none(lengths),
        "max_line_length": max(lengths) if lengths else 0,
        "blank_line_ratio": safe_div(len(lines) - len(non_empty), len(lines)),
    }


def compute_format_metrics(code: str) -> Dict[str, Any]:
    lines = (code or "").splitlines()
    if not lines:
        return {
            "long_line_ratio": 0.0,
            "trailing_whitespace_ratio": 0.0,
            "tab_indent_ratio": 0.0,
            "mixed_indent_flag": 0,
        }
    long_lines = sum(1 for x in lines if len(x) > 88)
    trailing_ws = sum(1 for x in lines if x.rstrip(" \t") != x)
    tab_indent = sum(1 for x in lines if re.match(r"^\t+", x))
    space_indent = sum(1 for x in lines if re.match(r"^ +", x))
    return {
        "long_line_ratio": safe_div(long_lines, len(lines)),
        "trailing_whitespace_ratio": safe_div(trailing_ws, len(lines)),
        "tab_indent_ratio": safe_div(tab_indent, len(lines)),
        "mixed_indent_flag": int(tab_indent > 0 and space_indent > 0),
    }


def comment_stats(code: str, language: str) -> Dict[str, Any]:
    lines = (code or "").splitlines()
    total = len(lines)
    comment = 0
    inline_comment = 0
    block_markers = []
    if language == "python":
        line_markers = ["#"]
        block_markers = [('"""', '"""'), ("'''", "'''")]
    else:
        line_markers = ["//", "#"]
        block_markers = [("/*", "*/")]

    in_block = False
    current_end = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if in_block:
            comment += 1
            if current_end and current_end in s:
                in_block = False
                current_end = None
            continue
        matched_block = False
        for start, end in block_markers:
            if s.startswith(start):
                comment += 1
                matched_block = True
                if s.count(start) >= 2 or end in s[len(start):]:
                    in_block = False
                    current_end = None
                else:
                    in_block = True
                    current_end = end
                break
        if matched_block:
            continue
        if any(s.startswith(m) for m in line_markers):
            comment += 1
            continue
        if language == "python":
            if "#" in s and not s.startswith("#"):
                inline_comment += 1
        else:
            if "//" in s and not s.startswith("//"):
                inline_comment += 1
    return {
        "comment_line_count": comment,
        "inline_comment_count": inline_comment,
        "comment_ratio": safe_div(comment, total),
    }


def dangerous_usage_stats(code: str) -> Dict[str, Any]:
    category_hits: Dict[str, int] = {}
    flattened = []
    for category, patterns in DANGEROUS_PATTERNS.items():
        cnt = 0
        for pat in patterns:
            n = len(re.findall(pat, code or ""))
            cnt += n
            if n:
                flattened.append({"category": category, "pattern": pat, "count": n})
        category_hits[category] = cnt
    return {
        "dangerous_call_count": sum(category_hits.values()),
        "dangerous_dynamic_exec_count": category_hits["dynamic_exec"],
        "dangerous_filesystem_count": category_hits["filesystem_write"],
        "dangerous_process_count": category_hits["process"],
        "dangerous_deserialization_count": category_hits["deserialization"],
        "dangerous_network_count": category_hits["network"],
        "dangerous_native_bridge_count": category_hits["native_bridge"],
        "dangerous_patterns": flattened,
    }


def duplicated_line_ratio(code: str, language: str) -> float:
    lines = []
    for line in (code or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if language == "python" and s.startswith("#"):
            continue
        if language != "python" and (s.startswith("//") or s.startswith("#")):
            continue
        lines.append(s)
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(v for v in counts.values() if v > 1)
    return safe_div(repeated, len(lines))


def identifier_metrics(code: str, context_text: str) -> Dict[str, Any]:
    patch_ids = normalize_identifier_list(extract_identifiers(code))
    context_ids = set(normalize_identifier_list(extract_identifiers(context_text)))
    if not patch_ids:
        return {
            "identifier_count": 0,
            "unique_identifier_count": 0,
            "avg_identifier_length": 0.0,
            "short_identifier_ratio": 0.0,
            "snake_case_ratio": 0.0,
            "context_identifier_overlap_ratio": 0.0,
            "identifier_entropy": 0.0,
        }

    unique_ids = list(dict.fromkeys(patch_ids))
    short_ratio = safe_div(sum(1 for x in patch_ids if len(x) <= 2 and x not in COMMON_SHORT_IDS), len(patch_ids))
    snake_ratio = safe_div(sum(1 for x in patch_ids if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", x) is not None), len(patch_ids))
    overlap_ratio = safe_div(sum(1 for x in unique_ids if x in context_ids), len(unique_ids))
    return {
        "identifier_count": len(patch_ids),
        "unique_identifier_count": len(set(patch_ids)),
        "avg_identifier_length": mean_or_none([len(x) for x in patch_ids]) or 0.0,
        "short_identifier_ratio": short_ratio,
        "snake_case_ratio": snake_ratio,
        "context_identifier_overlap_ratio": overlap_ratio,
        "identifier_entropy": shannon_entropy(patch_ids),
    }


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.cc_score = 1
        self.max_nesting_depth = 0
        self._current_depth = 0
        self.function_count = 0
        self.class_count = 0
        self.lambda_count = 0
        self.function_docstring_count = 0
        self.class_docstring_count = 0
        self.branch_count = 0
        self.loop_count = 0
        self.try_count = 0
        self.return_count = 0
        self.import_count = 0
        self.call_count = 0
        self.listcomp_count = 0
        self.param_counts: List[int] = []
        self.function_lengths: List[int] = []
        self.defined_names: Set[str] = set()
        self.loaded_names: List[str] = []
        self.sensitive_import_count = 0

    def _enter_nested(self) -> None:
        self._current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self._current_depth)

    def _leave_nested(self) -> None:
        self._current_depth -= 1

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, PY_DECISION_NODES):
            self.cc_score += 1
            self.branch_count += int(isinstance(node, (ast.If, ast.IfExp, ast.Match, ast.ExceptHandler)))
            self.loop_count += int(isinstance(node, (ast.For, ast.AsyncFor, ast.While)))
            self.try_count += int(isinstance(node, ast.Try))
            self._enter_nested()
            super().generic_visit(node)
            self._leave_nested()
            return
        if isinstance(node, ast.BoolOp):
            self.cc_score += max(1, len(node.values) - 1)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self.cc_score += len(getattr(node, "generators", []))
            self.listcomp_count += 1
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_count += 1
        self.defined_names.add(node.name)
        if ast.get_docstring(node):
            self.function_docstring_count += 1
        argc = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg is not None:
            argc += 1
        if node.args.kwarg is not None:
            argc += 1
        self.param_counts.append(argc)
        for arg in node.args.args + node.args.kwonlyargs:
            self.defined_names.add(arg.arg)
        if node.args.vararg is not None:
            self.defined_names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self.defined_names.add(node.args.kwarg.arg)
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None and end is not None:
            self.function_lengths.append(max(0, end - start + 1))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_count += 1
        self.defined_names.add(node.name)
        if ast.get_docstring(node):
            self.class_docstring_count += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.lambda_count += 1
        self.cc_score += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_count += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.import_count += 1
        for alias in node.names:
            base = alias.name.split('.')[0]
            self.defined_names.add(alias.asname or base)
            self.sensitive_import_count += int(base in SENSITIVE_IMPORTS)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_count += 1
        base = (node.module or "").split('.')[0]
        self.sensitive_import_count += int(base in SENSITIVE_IMPORTS)
        for alias in node.names:
            self.defined_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            for name in extract_target_names(t):
                self.defined_names.add(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in extract_target_names(node.target):
            self.defined_names.add(name)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in extract_target_names(node.target):
            self.defined_names.add(name)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for name in extract_target_names(node.target):
            self.defined_names.add(name)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for name in extract_target_names(item.optional_vars):
                    self.defined_names.add(name)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)  # type: ignore[arg-type]

    def visit_Call(self, node: ast.Call) -> None:
        self.call_count += 1
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.append(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        self.generic_visit(node)


def extract_target_names(node: ast.AST) -> List[str]:
    out: List[str] = []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            out.extend(extract_target_names(elt))
    return out


def python_ast_stats(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code or "")
    except Exception as e:
        return {
            "syntax_ok": 0,
            "compile_ok": 0,
            "syntax_error": str(e),
            "cyclomatic_complexity": None,
            "complexity_source": None,
            "maintainability_index": None,
            "halstead_volume": None,
            "function_count": None,
            "class_count": None,
            "lambda_count": None,
            "branch_count": None,
            "loop_count": None,
            "try_count": None,
            "return_count": None,
            "import_count": None,
            "call_count": None,
            "listcomp_count": None,
            "max_nesting_depth": None,
            "avg_function_length": None,
            "max_function_length": None,
            "avg_parameter_count": None,
            "function_docstring_coverage": None,
            "class_docstring_coverage": None,
            "module_docstring": None,
            "sensitive_import_count": None,
            "unknown_identifier_ratio": None,
        }

    try:
        compile(code or "", "<repoeval>", "exec")
        compile_ok = 1
        syntax_err = None
    except Exception as e:
        compile_ok = 0
        syntax_err = str(e)

    analyzer = PythonAnalyzer()
    analyzer.visit(tree)
    module_doc = ast.get_docstring(tree)

    if HAS_RADON:
        try:
            cc_blocks = cc_visit(code or "")
            cc_score = max([getattr(x, "complexity", 1) for x in cc_blocks] + [analyzer.cc_score])
        except Exception:
            cc_score = analyzer.cc_score
        try:
            mi = float(mi_visit(code or "", multi=True))
        except Exception:
            mi = None
        try:
            hv = h_visit(code or "")
            halstead_volume = float(getattr(hv.total, "volume", None)) if getattr(hv, "total", None) is not None else None
        except Exception:
            halstead_volume = None
        complexity_source = "radon_ast"
    else:
        cc_score = analyzer.cc_score
        mi = None
        halstead_volume = None
        complexity_source = "python_ast"

    loaded = [x for x in analyzer.loaded_names if x not in STOPWORDS]
    unknown = [x for x in loaded if x not in analyzer.defined_names and x not in PY_BUILTINS]

    return {
        "syntax_ok": 1,
        "compile_ok": compile_ok,
        "syntax_error": syntax_err,
        "cyclomatic_complexity": cc_score,
        "complexity_source": complexity_source,
        "maintainability_index": mi,
        "halstead_volume": halstead_volume,
        "function_count": analyzer.function_count,
        "class_count": analyzer.class_count,
        "lambda_count": analyzer.lambda_count,
        "branch_count": analyzer.branch_count,
        "loop_count": analyzer.loop_count,
        "try_count": analyzer.try_count,
        "return_count": analyzer.return_count,
        "import_count": analyzer.import_count,
        "call_count": analyzer.call_count,
        "listcomp_count": analyzer.listcomp_count,
        "max_nesting_depth": analyzer.max_nesting_depth,
        "avg_function_length": mean_or_none(analyzer.function_lengths),
        "max_function_length": max(analyzer.function_lengths) if analyzer.function_lengths else 0,
        "avg_parameter_count": mean_or_none(analyzer.param_counts),
        "function_docstring_coverage": safe_div(analyzer.function_docstring_count, analyzer.function_count),
        "class_docstring_coverage": safe_div(analyzer.class_docstring_count, analyzer.class_count),
        "module_docstring": int(module_doc is not None),
        "sensitive_import_count": analyzer.sensitive_import_count,
        "unknown_identifier_ratio": safe_div(len(unknown), len(loaded)),
    }


def approx_complexity_text(code: str, language: str) -> Dict[str, Any]:
    text = code or ""
    if language == "python":
        keywords = [" if ", " for ", " while ", " elif ", " except", " and ", " or ", " case ", " lambda "]
    else:
        keywords = [" if ", " for ", " while ", " case ", " catch", "&&", "||", "?", "=>"]
    score = 1
    lower = f" {text.lower()} "
    for kw in keywords:
        score += lower.count(kw)
    return {
        "cyclomatic_complexity": score,
        "complexity_source": "heuristic_text",
        "maintainability_index": None,
        "halstead_volume": None,
        "function_count": None,
        "class_count": None,
        "lambda_count": None,
        "branch_count": None,
        "loop_count": None,
        "try_count": None,
        "return_count": None,
        "import_count": None,
        "call_count": None,
        "listcomp_count": None,
        "max_nesting_depth": None,
        "avg_function_length": None,
        "max_function_length": None,
        "avg_parameter_count": None,
        "function_docstring_coverage": None,
        "class_docstring_coverage": None,
        "module_docstring": None,
        "sensitive_import_count": None,
        "unknown_identifier_ratio": None,
    }


def generic_syntax_stats(snippet_code: str, full_code: str, language: str) -> Dict[str, Any]:
    text = full_code or ""
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    balanced = 1
    for l, r in pairs:
        if text.count(l) != text.count(r):
            balanced = 0
            break
    cc = approx_complexity_text(snippet_code, language)
    return {
        "syntax_ok": balanced,
        "compile_ok": balanced,
        "syntax_error": None if balanced else "delimiter_unbalanced",
        "snippet_syntax_ok": None,
        "snippet_compile_ok": None,
        **cc,
    }


def syntax_and_structure_stats(snippet_code: str, full_code: str, language: str) -> Dict[str, Any]:
    if language == "python":
        stats = python_ast_stats(full_code)
        if not stats["syntax_ok"]:
            snippet_stats = python_ast_stats(snippet_code)
            stats["snippet_syntax_ok"] = snippet_stats["syntax_ok"]
            stats["snippet_compile_ok"] = snippet_stats["compile_ok"]
        else:
            stats["snippet_syntax_ok"] = None
            stats["snippet_compile_ok"] = None
        if stats.get("cyclomatic_complexity") is None:
            stats.update(approx_complexity_text(snippet_code, language))
        return stats
    return generic_syntax_stats(snippet_code, full_code, language)


def semantic_coherence_metrics(code: str, prompt: str, right_context: str, crossfile_context: str) -> Dict[str, Any]:
    context = "\n".join([prompt or "", right_context or "", crossfile_context or ""])
    return identifier_metrics(code, context)


def compute_quality_metrics(task: Dict[str, Any], primary_code: str) -> Dict[str, Any]:
    prompt = str(task.get("prompt") or "")
    right_context = str(task.get("right_context") or "")
    crossfile_context = stringify_crossfile_context(task.get("crossfile_context"))
    full_code = reconstruct_full_code(task, primary_code)
    language = infer_language(task, code=full_code)

    size_metrics = compute_size_metrics(primary_code)
    format_metrics = compute_format_metrics(primary_code)
    comment_metrics = comment_stats(primary_code, language)
    danger_metrics = dangerous_usage_stats(primary_code)
    structure_metrics = syntax_and_structure_stats(primary_code, full_code, language)
    semantic_metrics = semantic_coherence_metrics(primary_code, prompt, right_context, crossfile_context)

    return {
        "language": language,
        **size_metrics,
        **format_metrics,
        **comment_metrics,
        **danger_metrics,
        **structure_metrics,
        **semantic_metrics,
        "duplicated_line_ratio": duplicated_line_ratio(primary_code, language),
        "prompt_length_chars": len(prompt),
        "right_context_length_chars": len(right_context),
        "crossfile_context_length_chars": len(crossfile_context),
    }


def compute_sample_metrics(
    task: Dict[str, Any],
    candidates_raw: List[str],
    gold: str,
    passk: int = 1,
    extraction_mode: str = "extract",
) -> Dict[str, Any]:
    raw_candidates = [c if isinstance(c, str) else str(c) for c in (candidates_raw or [""])]
    if extraction_mode == "extract":
        processed_candidates = [strip_code_fence(c)[0] for c in raw_candidates]
        extraction_note = strip_code_fence(raw_candidates[0])[1] if raw_candidates else "none"
    else:
        processed_candidates = [(c or "").strip() for c in raw_candidates]
        extraction_note = "none"

    primary_code = processed_candidates[0] if processed_candidates else ""
    quality = compute_quality_metrics(task, primary_code)

    official_em = official_compute_em(gold, raw_candidates, passk=passk)
    official_es = official_compute_es(gold, raw_candidates, passk=passk)

    return {
        "normalized_code": primary_code,
        "candidate_count": len(raw_candidates),
        "scored_passk": min(passk, len(raw_candidates)),
        "extraction_note": extraction_note,
        "exact_match": official_em,
        "edit_similarity": official_es,
        "whitespace_normalized_exact_match": int(normalize_whitespace(primary_code) == normalize_whitespace(gold)),
        **quality,
    }


def load_groundtruth(path: str) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tid = get_task_id(row)
        if tid is None:
            continue
        out[tid] = row
    if not out:
        raise ValueError(f"No valid task_id found in groundtruth file: {path}")
    return out


def load_generations(path: Optional[str], preferred_field: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    rows = read_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tid = get_task_id(row)
        if tid is None:
            continue
        row = dict(row)
        row["__candidates__"] = extract_candidates(row, preferred=preferred_field)
        out[tid] = row
    return out


def get_gold_completion(task: Dict[str, Any]) -> str:
    md = task.get("metadata") or {}
    val = get_first(task, ["completion", "groundtruth", "ground_truth", "target", "answer"], None)
    if val is None:
        val = get_first(md, ["groundtruth", "ground_truth", "completion", "target", "answer"], "")
    return str(val if val is not None else "")


def numeric_summary(rows: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    selected = [r[variant] for r in rows if r.get(variant) is not None]
    if not selected:
        return {"count": 0}

    numeric_fields: Set[str] = set()
    for item in selected:
        for k, v in item.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_fields.add(k)

    out: Dict[str, Any] = {"count": len(selected)}
    for field in sorted(numeric_fields):
        vals = [float(item[field]) for item in selected if isinstance(item.get(field), (int, float)) and not isinstance(item.get(field), bool)]
        if vals:
            out[field] = statistics.mean(vals)

    out["complexity_source_counts"] = dict(Counter(
        item.get("complexity_source") or "unknown" for item in selected
    ))
    out["extraction_note_counts"] = dict(Counter(
        item.get("extraction_note") or "unknown" for item in selected
    ))
    return out


def summarize_delta(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    paired = [r for r in rows if r.get("prediction") is not None and r.get("raw") is not None]
    if not paired:
        return {"paired_count": 0}

    metrics_to_compare = [
        "exact_match", "edit_similarity", "syntax_ok", "compile_ok", "comment_ratio",
        "cyclomatic_complexity", "maintainability_index", "dangerous_call_count",
        "context_identifier_overlap_ratio", "unknown_identifier_ratio",
        "duplicated_line_ratio", "long_line_ratio",
    ]
    out: Dict[str, Any] = {"paired_count": len(paired)}
    for metric in metrics_to_compare:
        diffs = []
        for r in paired:
            a = r["prediction"].get(metric)
            b = r["raw"].get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diffs.append(float(a) - float(b))
        out[f"avg_{metric}_delta"] = statistics.mean(diffs) if diffs else None

    out["prediction_better_count_exact_match"] = sum(
        1 for r in paired if r["prediction"].get("exact_match", -1) > r["raw"].get("exact_match", -1)
    )
    out["prediction_better_count_edit_similarity"] = sum(
        1 for r in paired if r["prediction"].get("edit_similarity", -1) > r["raw"].get("edit_similarity", -1)
    )
    out["prediction_lower_risk_count"] = sum(
        1 for r in paired if r["prediction"].get("dangerous_call_count", 1e9) < r["raw"].get("dangerous_call_count", 1e9)
    )
    return out


def summarize_groups(rows: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row.get(group_key) or "unknown"
        buckets[str(key)].append(row)
    out: Dict[str, Any] = {}
    for k, items in buckets.items():
        out[k] = {
            "prediction": numeric_summary(items, "prediction"),
            "raw": numeric_summary(items, "raw"),
            "delta": summarize_delta(items),
        }
    return out


def evaluate_repoeval(
    groundtruth_file: str,
    prediction_file: str,
    raw_file: Optional[str],
    output_dir: str,
    prediction_field: Optional[str] = None,
    raw_field: Optional[str] = None,
    extraction_mode: str = "extract",
    passk: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt = load_groundtruth(groundtruth_file)
    preds = load_generations(prediction_file, preferred_field=prediction_field)
    raws = load_generations(raw_file, preferred_field=raw_field) if raw_file else {}

    rows: List[Dict[str, Any]] = []
    missing_pred: List[str] = []
    missing_raw: List[str] = []

    for tid, task in gt.items():
        gold = get_gold_completion(task)
        md = task.get("metadata") or {}
        repo_name = tid.split('/')[0] if '/' in tid else None
        task_type = None
        fpath = os.path.basename(groundtruth_file).lower()
        if "api_level" in fpath:
            task_type = "api_level"
        elif "line_level" in fpath:
            task_type = "line_level"
        elif "function_level" in fpath:
            task_type = "function_level"

        row: Dict[str, Any] = {
            "task_id": tid,
            "repo_name": repo_name,
            "task_type": task_type,
            "file_path": get_first(task, ["filepath", "file_path", "path"], None) or md.get("filepath") or md.get("path"),
            "groundtruth": {
                "completion": gold,
                "language": infer_language(task, reconstruct_full_code(task, gold)),
            },
            "prediction": None,
            "raw": None,
        }

        pred_obj = preds.get(tid)
        if pred_obj is None:
            missing_pred.append(tid)
        else:
            row["prediction"] = compute_sample_metrics(
                task,
                pred_obj.get("__candidates__", [""]),
                gold,
                passk=passk,
                extraction_mode=extraction_mode,
            )
            row["prediction"]["source_record_keys"] = sorted(k for k in pred_obj.keys() if not k.startswith("__"))

        raw_obj = raws.get(tid)
        if raw_file and raw_obj is None:
            missing_raw.append(tid)
        elif raw_obj is not None:
            row["raw"] = compute_sample_metrics(
                task,
                raw_obj.get("__candidates__", [""]),
                gold,
                passk=passk,
                extraction_mode=extraction_mode,
            )
            row["raw"]["source_record_keys"] = sorted(k for k in raw_obj.keys() if not k.startswith("__"))

        if row.get("prediction") is not None:
            row["language_group"] = row["prediction"].get("language")
        elif row.get("raw") is not None:
            row["language_group"] = row["raw"].get("language")
        else:
            row["language_group"] = row["groundtruth"].get("language")

        rows.append(row)

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    write_jsonl(metrics_path, rows)

    summary = {
        "files": {
            "groundtruth": groundtruth_file,
            "prediction": prediction_file,
            "raw": raw_file,
        },
        "settings": {
            "passk": passk,
            "extraction_mode": extraction_mode,
            "official_em_es_alignment": True,
            "radon_available": HAS_RADON,
        },
        "counts": {
            "groundtruth_tasks": len(gt),
            "prediction_rows": len(preds),
            "raw_rows": len(raws),
            "aligned_rows": len(rows),
            "missing_prediction_count": len(missing_pred),
            "missing_raw_count": len(missing_raw),
        },
        "missing_prediction_task_ids": missing_pred[:200],
        "missing_raw_task_ids": missing_raw[:200],
        "overall": {
            "prediction": numeric_summary(rows, "prediction"),
            "raw": numeric_summary(rows, "raw"),
            "delta": summarize_delta(rows),
        },
        "by_repo": summarize_groups(rows, "repo_name"),
        "by_task_type": summarize_groups(rows, "task_type"),
        "by_language": summarize_groups(rows, "language_group"),
        "distributions": {
            "prediction_language_counts": dict(Counter(
                r["prediction"].get("language") for r in rows if r.get("prediction") is not None
            )),
            "raw_language_counts": dict(Counter(
                r["raw"].get("language") for r in rows if r.get("raw") is not None
            )),
            "task_type_counts": dict(Counter((r.get("task_type") or "unknown") for r in rows)),
        },
    }

    dump_json(summary_path, summary)
    return rows, summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RepoEval evaluation with official EM/ES + extended quality metrics")
    p.add_argument("--groundtruth", "--prompt_file", dest="groundtruth", required=True,
                   help="RepoEval groundtruth/prompt JSONL file")
    p.add_argument("--prediction", required=True, help="prediction.jsonl file")
    p.add_argument("--raw", default=None, help="raw_outputs.jsonl / raw_generation.jsonl")
    p.add_argument("--output_dir", required=True, help="Directory to save metrics.jsonl and summary.json")
    p.add_argument("--prediction_field", default=None,
                   help="Optional explicit code field/path in prediction file, e.g. choices.0.text")
    p.add_argument("--raw_field", default=None,
                   help="Optional explicit code field/path in raw file, e.g. clean_text")
    p.add_argument("--extraction_mode", choices=["extract", "none"], default="extract",
                   help="Whether to strip markdown code fences before quality analysis")
    p.add_argument("--passk", type=int, default=1,
                   help="Official pass@k-style top-k scoring for EM/ES (default: 1)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _, summary = evaluate_repoeval(
        groundtruth_file=args.groundtruth,
        prediction_file=args.prediction,
        raw_file=args.raw,
        output_dir=args.output_dir,
        prediction_field=args.prediction_field,
        raw_field=args.raw_field,
        extraction_mode=args.extraction_mode,
        passk=args.passk,
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Saved metrics to {os.path.join(args.output_dir, 'metrics.jsonl')}")
    print(f"Saved summary to {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
