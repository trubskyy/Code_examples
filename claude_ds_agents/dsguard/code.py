"""Code health: complexity metrics, duplicated functions, pandas/sklearn pitfalls, ruff."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict

from .core import CheckResult, Context, PySource, call_name, dotted_name, kwarg, python_sources

COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
GROW_IN_LOOP = {"pd.concat", "pandas.concat", "np.append", "np.concatenate", "np.vstack", "np.hstack",
                "numpy.append", "numpy.concatenate", "numpy.vstack", "numpy.hstack"}
SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "ShuffleSplit", "StratifiedShuffleSplit", "RepeatedKFold"}


def _walk_scope(node: ast.AST):
    """Walk a function body without descending into nested functions or classes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, SCOPES):
            yield from _walk_scope(child)


def complexity(fn: ast.AST) -> int:
    score = 1
    for node in _walk_scope(fn):
        if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.Assert)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            score += 1 + len(node.ifs)
        elif isinstance(node, ast.match_case):
            score += 1
    return score


def nesting(node: ast.AST, depth: int = 0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            continue
        deepest = max(deepest, nesting(child, depth + 1 if isinstance(child, COMPOUND) else depth))
    return deepest


class _Normalize(ast.NodeTransformer):
    """Rename local identifiers so copy-pasted functions with renamed variables still match."""

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node):
        return ast.copy_location(ast.arg(arg="_"), node)


def fingerprint(fn: ast.AST) -> str:
    body = ast.Module(body=[_Normalize().visit(ast.parse(ast.unparse(s))) for s in fn.body], type_ignores=[])
    return hashlib.sha1(ast.dump(body).encode()).hexdigest()


def _metrics(src: PySource, cfg: dict, result: CheckResult, fingerprints: dict) -> None:
    for node in ast.walk(src.tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        args = node.args
        n_args = len(args.posonlyargs + args.args + args.kwonlyargs)
        if args.args and args.args[0].arg in ("self", "cls"):
            n_args -= 1
        cc, depth = complexity(node), nesting(node)
        problems = []
        if lines > cfg["max_function_lines"]:
            problems.append(f"{lines} lines")
        if cc > cfg["max_complexity"]:
            problems.append(f"complexity {cc}")
        if depth > cfg["max_nesting"]:
            problems.append(f"nesting {depth}")
        if n_args > cfg["max_args"]:
            problems.append(f"{n_args} args")
        if problems:
            result.add("function-size", "info", src.path, f"{node.name}(): {', '.join(problems)}", node.lineno, src.cell)
        if lines >= cfg["min_duplicate_lines"]:
            fingerprints[fingerprint(node)].append((src, node))


def _pitfalls(src: PySource, result: CheckResult) -> None:
    uses_pandas = bool(re.search(r"^\s*(import|from)\s+(pandas|polars|modin)\b", src.source, re.M))
    loop_calls: set[int] = set()
    for loop in ast.walk(src.tree):
        if isinstance(loop, LOOPS):
            for stmt in loop.body:
                loop_calls.update(id(n) for n in ast.walk(stmt) if isinstance(n, ast.Call))

    def add(rule, severity, node, message):
        result.add(rule, severity, src.path, message, node.lineno, src.cell)

    for node in ast.walk(src.tree):
        if isinstance(node, ast.Call):
            name, full = call_name(node.func), dotted_name(node.func)
            if name in ("iterrows", "itertuples"):
                # iterrows builds a Series per row (slow, loses dtypes); itertuples is fine on small tables.
                add("row-iteration", "warn" if name == "iterrows" else "info", node,
                    f".{name}() loops in Python; vectorize or use groupby/merge if the frame is large")
            elif name == "apply" and (ax := kwarg(node, "axis")) and getattr(ax.value, "value", None) in (1, "columns"):
                add("row-apply", "info", node, ".apply(axis=1) is a Python loop per row; vectorize if hot")
            elif name == "merge" and kwarg(node, "validate") is None:
                add("merge-unvalidated", "info", node, "merge() without validate= lets duplicate keys silently fan out rows")
            elif name == "train_test_split" and kwarg(node, "random_state") is None:
                add("unseeded-split", "info", node, "train_test_split without random_state is not reproducible")
            elif name in SPLITTERS and (sh := kwarg(node, "shuffle")) and getattr(sh.value, "value", None) is True \
                    and kwarg(node, "random_state") is None:
                add("unseeded-split", "info", node, f"{name}(shuffle=True) without random_state is not reproducible")
            if full in GROW_IN_LOOP and id(node) in loop_calls:
                add("grow-in-loop", "warn", node, f"{full}() inside a loop is quadratic; collect a list and combine once")
            elif (full.startswith(("pd.read_", "pandas.read_"))) and id(node) in loop_calls:
                add("read-in-loop", "info", node, f"{full}() inside a loop; consider reading once or in bulk")
            if (ip := kwarg(node, "inplace")) and getattr(ip.value, "value", None) is True:
                add("inplace", "info", node, "inplace=True blocks method chaining and is slated for deprecation")
        elif isinstance(node, ast.Assign) and uses_pandas:
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript):
                    # df[mask]["col"] = x is the classic bug; d["a"]["b"] = x may just be a dict.
                    is_mask = not isinstance(target.value.slice, ast.Constant)
                    add("chained-assignment", "warn" if is_mask else "info", node,
                        "chained indexing assignment may write to a copy and be lost; use .loc[row, col]")
        elif isinstance(node, ast.ExceptHandler):
            if all(isinstance(s, (ast.Pass, ast.Continue)) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                   for s in node.body):
                add("swallowed-exception", "warn", node, "exception is silently swallowed")


def _ruff(ctx: Context, result: CheckResult) -> None:
    cfg = ctx.config["code"]
    exe = shutil.which("ruff")
    files = [str(ctx.root / f) for f in ctx.files if f.endswith((".py", ".ipynb"))]
    if not cfg["ruff"] or not files:
        return
    if not exe:
        result.notes.append("ruff not installed: lint rules skipped")
        return
    cmd = [exe, "check", "--output-format", "json", "--exit-zero", "--no-cache",
           "--extend-select", ",".join(cfg["ruff_extend_select"]), *files]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=ctx.root)
    try:
        issues = json.loads(proc.stdout)
    except ValueError:
        result.notes.append(f"ruff failed: {proc.stderr.strip()[:300]}")
        return
    for issue in issues:
        code = issue.get("code") or "syntax"
        severity = "warn" if re.match(r"(F\d|E9|B\d|PLE|syntax)", code) else "info"
        rel = issue["filename"].removeprefix(str(ctx.root) + "/")
        result.add(f"ruff:{code}", severity, rel, issue.get("message", ""),
                   issue.get("location", {}).get("row"), issue.get("cell"))


def run(ctx: Context) -> CheckResult:
    result = CheckResult("code")
    cfg = ctx.config["code"]
    fingerprints: dict[str, list] = defaultdict(list)
    for src in python_sources(ctx):
        if src.tree is None:
            if src.source.strip():
                result.add("syntax-error", "warn", src.path, "does not parse; metrics skipped", cell=src.cell)
            continue
        _metrics(src, cfg, result, fingerprints)
        _pitfalls(src, result)
    for group in fingerprints.values():
        if len(group) > 1:
            places = ", ".join(f"{s.path}:{n.lineno} {n.name}()" for s, n in group)
            first_src, first = group[0]
            result.add("duplicate-function", "info", first_src.path,
                       f"{len(group)} functions share the same body: {places}", first.lineno, first_src.cell)
    _ruff(ctx, result)
    return result
