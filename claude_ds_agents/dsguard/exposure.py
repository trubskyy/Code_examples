"""Data exposure: committed data/model files, notebook outputs, PII, open network binds."""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

from .core import CheckResult, Context, call_name, cell_text, const_str, kwarg, match_any, notebook_cells, python_sources

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PII_TEXT_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".txt", ".log", ".xml")


def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def pii_hits(text: str, allow_domains: list[str]) -> list[tuple[str, int]]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in EMAIL.finditer(line):
            domain = m.group(1).lower()
            if not any(domain == d or domain.endswith("." + d) for d in allow_domains):
                hits.append(("email address", lineno))
                break
        if SSN.search(line):
            hits.append(("SSN-like number", lineno))
        for m in CARD.finditer(line):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and luhn(digits) and len(set(digits)) > 2:
                hits.append(("payment card number", lineno))
                break
    return hits


def _notebook(ctx: Context, rel: str, text: str, result: CheckResult, allow_domains: list[str]) -> None:
    with_output = 0
    for index, cell in enumerate(notebook_cells(text)):
        outputs = cell.get("outputs") or []
        if not outputs:
            continue
        with_output += 1
        rendered = []
        for out in outputs:
            rendered.append(cell_text(out.get("text")))
            data = out.get("data") or {}
            rendered.append(cell_text(data.get("text/plain")))
            rendered.append(cell_text(data.get("text/html")))
        for kind, line in pii_hits("\n".join(rendered), allow_domains):
            result.add("pii-in-notebook-output", "warn", rel, f"{kind} in cell output", line, index)
    if with_output:
        result.add("notebook-outputs", "warn" if ctx.tracked and rel in ctx.tracked else "info", rel,
                   f"{with_output} cell(s) with saved outputs; strip them (nbstripout) so data is not committed")


def _network(src, result: CheckResult) -> None:
    for node in ast.walk(src.tree):
        if not isinstance(node, ast.Call):
            continue
        host = kwarg(node, "host") or kwarg(node, "server_name")
        if host and const_str(host.value) == "0.0.0.0":
            result.add("binds-all-interfaces", "warn", src.path,
                       "server binds 0.0.0.0; anyone on the network can reach it", node.lineno, src.cell)
        share = kwarg(node, "share")
        if call_name(node.func) == "launch" and share and getattr(share.value, "value", None) is True:
            result.add("public-share-link", "block", src.path,
                       "launch(share=True) publishes a public URL to this app", node.lineno, src.cell)
        debug = kwarg(node, "debug")
        if call_name(node.func) == "run" and debug and getattr(debug.value, "value", None) is True:
            result.add("debug-server", "warn", src.path,
                       "run(debug=True) exposes an interactive debugger", node.lineno, src.cell)
        origins = kwarg(node, "allow_origins")
        if origins and isinstance(origins.value, ast.List) and any(
            const_str(e) == "*" for e in origins.value.elts
        ):
            result.add("cors-wildcard", "warn", src.path, "CORS allows any origin", node.lineno, src.cell)


def run(ctx: Context) -> CheckResult:
    result = CheckResult("exposure")
    cfg = ctx.config["exposure"]
    allow_domains = cfg["pii_email_allow_domains"]
    for rel in ctx.files:
        suffix = PurePosixPath(rel).suffix.lower()
        tracked = ctx.tracked is not None and rel in ctx.tracked
        allowed = match_any(rel, cfg["allowed_data_paths"])
        mb = ctx.size(rel) / 1e6
        if suffix in cfg["data_extensions"] and not allowed:
            result.add("data-file", "warn" if tracked else "info", rel,
                       f"data file ({mb:.1f} MB) {'committed' if tracked else 'not gitignored'}; keep data out of the repo")
        elif suffix in cfg["model_extensions"] and not allowed:
            result.add("model-artifact", "warn" if tracked else "info", rel,
                       f"model/array artifact ({mb:.1f} MB) {'committed' if tracked else 'not gitignored'}; use a model registry")
        if suffix == ".ipynb":
            text = ctx.read(rel)
            if text:
                _notebook(ctx, rel, text, result, allow_domains)
        elif suffix in PII_TEXT_EXTS:
            text = ctx.read(rel)
            if text:
                hits = pii_hits(text, allow_domains)
                kinds = sorted({kind for kind, _ in hits})
                if hits:
                    result.add("pii-in-file", "warn", rel,
                               f"{len(hits)} line(s) with {', '.join(kinds)}", hits[0][1])
    for src in python_sources(ctx):
        if src.tree:
            _network(src, result)
    return result
