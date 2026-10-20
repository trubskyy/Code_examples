"""Python code-security patterns common in DS code; adds bandit findings if installed."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess

from .core import CheckResult, Context, call_name, dotted_name, kwarg, python_sources

UNSAFE_LOADERS = {"pickle.load", "pickle.loads", "joblib.load", "pd.read_pickle", "pandas.read_pickle",
                  "cloudpickle.load", "dill.load"}


def _is_true(node: ast.keyword | None) -> bool:
    return node is not None and isinstance(node.value, ast.Constant) and node.value.value is True


def scan_tree(tree: ast.Module) -> list[tuple[str, str, int, str]]:
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name, full = call_name(node.func), dotted_name(node.func)
        if full in UNSAFE_LOADERS:
            hits.append(("unsafe-deserialization", "info", node.lineno,
                         f"{full}() runs arbitrary code from the file; only load artifacts you produced"))
        elif full == "torch.load" and not _is_true(kwarg(node, "weights_only")):
            hits.append(("unsafe-deserialization", "info", node.lineno,
                         "torch.load without weights_only=True can run arbitrary code"))
        elif full == "yaml.load" and kwarg(node, "Loader") is None and len(node.args) < 2:
            hits.append(("yaml-unsafe-load", "warn", node.lineno, "yaml.load without Loader; use yaml.safe_load"))
        elif name in {"eval", "exec"} and isinstance(node.func, ast.Name):
            hits.append(("eval-exec", "warn", node.lineno, f"{name}() on dynamic input is code injection"))
        elif full == "os.system" or (full.startswith("subprocess.") and _is_true(kwarg(node, "shell"))):
            hits.append(("shell-injection", "warn", node.lineno, "shell command string; pass an argument list"))
        verify = kwarg(node, "verify")
        if verify is not None and isinstance(verify.value, ast.Constant) and verify.value.value is False:
            hits.append(("tls-verify-disabled", "warn", node.lineno, "verify=False disables TLS certificate checks"))
        if _is_true(kwarg(node, "trust_remote_code")):
            hits.append(("trust-remote-code", "warn", node.lineno,
                         "trust_remote_code=True executes code from the model repo; pin a revision you reviewed"))
    return hits


def _bandit(ctx: Context, result: CheckResult) -> None:
    exe = shutil.which("bandit")
    py_files = [str(ctx.root / f) for f in ctx.files if f.endswith(".py")]
    if not exe:
        result.notes.append("bandit not installed: builtin patterns only")
        return
    if not py_files:
        return
    proc = subprocess.run([exe, "-f", "json", "-q", *py_files], capture_output=True, text=True, check=False)
    try:
        issues = json.loads(proc.stdout).get("results", [])
    except ValueError:
        result.notes.append(f"bandit failed: {proc.stderr.strip()[:300]}")
        return
    for issue in issues:
        if issue.get("issue_severity") == "LOW":
            continue
        rel = issue["filename"].removeprefix(str(ctx.root) + "/")
        severity = "warn" if issue.get("issue_severity") == "HIGH" else "info"
        result.add(f"bandit:{issue.get('test_id')}", severity, rel, issue.get("issue_text", ""), issue.get("line_number"))


def run(ctx: Context) -> CheckResult:
    result = CheckResult("pysec")
    for src in python_sources(ctx):
        if src.tree:
            for rule, severity, line, message in scan_tree(src.tree):
                result.add(rule, severity, src.path, message, line, src.cell)
    _bandit(ctx, result)
    return result
