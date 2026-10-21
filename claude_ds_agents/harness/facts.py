#!/usr/bin/env python3
"""Facts ledger for stakeholder summaries: every number in a summary must trace to a report.

  python facts.py collect REPORT.json [REPORT.json ...] [--compare OLD.json NEW.json] \
      [--extra business_facts.json] --out .dsguard/summaries/facts.json
  python facts.py verify summary.md --facts .dsguard/summaries/facts.json
  python facts.py render summary.md --facts .dsguard/summaries/facts.json --out summary.final.md

collect  turns harness/dsguard reports (diagnostics, leakage, drift, llm eval, dsguard run,
         or any JSON) into numbered facts F1..Fn: label, value, source file and JSON path.
         --compare adds before/after/delta/relative-change facts for shared numeric fields.
verify   fails (exit 1) unless every sentence with a number cites [F#], every cited fact
         exists, and every number in the sentence matches one of its cited facts at the
         precision written (0.0712 may be written 0.07, 7.1% or 7%).
render   strips the [F#] tags for the reader and appends a sources appendix.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])([-+−]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|pp|k|K|M|bn|x|×)?(?![\w])")
CITATION = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(19|20)\d{2}\b|\b\d{1,2}:\d{2}\b|\bQ[1-4]\b|\bH[12]\b|\bv\d+(\.\d+)*\b")
SCALE = {"k": 1e3, "K": 1e3, "M": 1e6, "bn": 1e9}
MAX_GENERIC = 400


def flatten(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            key = v.get("feature") or v.get("id") or v.get("rule") if isinstance(v, dict) else None
            yield from flatten(v, f"{path}[{key if key is not None else i}]")
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        yield path, value


def numbers_in(text: str) -> list[float]:
    return [float((sign or "") + whole.replace(",", "") + (f".{frac}" if frac else ""))
            for sign, whole, frac, _ in NUMBER.findall(str(text))]


def detect(report: dict) -> str:
    if "task" in report and "metrics" in report and "flags" in report:
        return "diagnostics"
    if "rows_train" in report and "findings" in report:
        return "leakage"
    if "prediction_drift" in report and "features" in report:
        return "drift"
    if "summary" in report and "results" in report and "target" in report:
        return "llm_eval"
    if "results" in report and "blocked" in report:
        return "dsguard"
    return "generic"


def collect_one(path: str, report: dict) -> list[dict]:
    kind = detect(report)
    facts: list[dict] = []

    def add(label, value=None, json_path="", text=None):
        facts.append({"label": label, "value": value, "text": text, "source": path, "path": json_path,
                      "kind": kind, "values": numbers_in(text) if text else []})

    if kind == "diagnostics":
        for key, value in report["metrics"].items():
            add(f"model {key}", value, f"metrics.{key}")
        add("evaluation rows", report["rows"], "rows")
        for i, flag in enumerate(report["flags"]):
            add(f"diagnostic flag ({flag['severity']}, {flag['rule']})", None, f"flags[{i}]", flag["message"])
        for row in report.get("segments", [])[:10]:
            add(f"segment {row['segment']}={row['level']} metric", row["metric"], f"segments[{row['level']}]")
            add(f"segment {row['segment']}={row['level']} rows", row["n"], f"segments[{row['level']}].n")
        for row in report.get("permutation_importance", [])[:10]:
            add(f"importance of {row['feature']}", row["importance"], f"permutation_importance[{row['feature']}]")
    elif kind == "leakage":
        add("train rows", report["rows_train"], "rows_train")
        add("test rows", report["rows_test"], "rows_test")
        for sev in ("high", "medium", "low"):
            add(f"{sev} leakage findings", sum(f["severity"] == sev for f in report["findings"]), "findings")
        for i, f in enumerate(report["findings"]):
            add(f"leakage finding ({f['severity']}, {f['rule']})", None, f"findings[{i}]", f["message"])
    elif kind == "drift":
        add("drift status", None, "status", report["status"])
        add("current rows", report["rows_current"], "rows_current")
        for i, reason in enumerate(report["reasons"]):
            add("drift reason", None, f"reasons[{i}]", reason)
        if report.get("prediction_drift"):
            add("prediction PSI", report["prediction_drift"]["psi"], "prediction_drift.psi")
        perf = report.get("performance") or {}
        for key in ("reference", "current", "change", "relative_change", "label_coverage"):
            if isinstance(perf.get(key), (int, float)):
                add(f"{perf.get('metric', 'performance')} {key}", perf[key], f"performance.{key}")
        majors = [f for f in report["features"] if f["status"] == "major"]
        add("features with major drift", len(majors), "features")
        for f in report["features"][:10]:
            add(f"PSI of {f['feature']}", f["psi"], f"features[{f['feature']}].psi")
    elif kind == "llm_eval":
        for key, value in report["summary"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                add(f"eval {key}", value, f"summary.{key}")
            elif isinstance(value, dict):
                for sub, v in value.items():
                    add(f"eval {key} {sub}", v, f"summary.{key}.{sub}")
        base = report.get("baseline")
        if base:
            add("eval regressions (pass to fail)", len(base["regressions"]), "baseline.regressions")
            add("eval fixes (fail to pass)", len(base["fixes"]), "baseline.fixes")
            for key, value in base["delta"].items():
                add(f"eval change in {key}", value, f"baseline.delta.{key}")
    elif kind == "dsguard":
        for result in report["results"]:
            for sev in ("block", "warn", "info"):
                add(f"{result['name']} {sev} findings", sum(f["severity"] == sev for f in result["findings"]),
                    f"results[{result['name']}]")
    else:
        for json_path, value in list(flatten(report))[:MAX_GENERIC]:
            add(json_path, value, json_path)
    return facts


def compare_facts(old_path: str, new_path: str) -> list[dict]:
    old = dict(flatten(json.loads(Path(old_path).read_text())))
    new = dict(flatten(json.loads(Path(new_path).read_text())))
    facts = []
    for key in [k for k in new if k in old][:MAX_GENERIC]:
        a, b = old[key], new[key]
        if a == b:
            continue
        base = {"source": f"{old_path} -> {new_path}", "path": key, "kind": "comparison", "text": None, "values": []}
        facts += [base | {"label": f"{key} before", "value": a}, base | {"label": f"{key} after", "value": b},
                  base | {"label": f"{key} change", "value": b - a}]
        if a:
            facts.append(base | {"label": f"{key} relative change", "value": (b - a) / abs(a)})
    return facts


def cmd_collect(args) -> int:
    facts = []
    for path in args.reports:
        facts += collect_one(path, json.loads(Path(path).read_text()))
    if args.compare:
        facts += compare_facts(*args.compare)
    if args.extra:
        for item in json.loads(Path(args.extra).read_text()):
            facts.append({"label": item["label"], "value": item.get("value"), "text": item.get("text"),
                          "source": item.get("source", args.extra), "path": "", "kind": "extra",
                          "values": numbers_in(item.get("text") or "")})
    for i, fact in enumerate(facts, 1):
        fact["id"] = f"F{i}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"facts": facts}, indent=2, default=str))
    lines = [f"[{f['id']}] {f['label']}: {f['value'] if f['value'] is not None else f['text']}" for f in facts]
    out.with_suffix(".md").write_text("# Facts ledger\n\n" + "\n".join(f"- {line}" for line in lines) + "\n")
    print(f"{len(facts)} facts -> {out} (readable: {out.with_suffix('.md')})")
    return 0


def matches(written: float, decimals: int, unit: str | None, fact_values: list[float]) -> bool:
    """True if a written number is the displayed form of any fact value (allowing %, pp, k/M, sign words)."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    scale = SCALE.get(unit or "", 1.0)
    for v in fact_values:
        candidates = [v, abs(v)]
        if unit in ("%", "pp"):
            candidates += [v * 100, abs(v) * 100]
        for c in candidates:
            if abs(c / scale - written) <= tol or abs(c / scale - abs(written)) <= tol:
                return True
    return False


def sentences(text: str):
    """(line_number, sentence) pairs; headings, table rows and list items count as one sentence each."""
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("<!--", "```")):
            continue
        if stripped.startswith(("|", "#", "-", "*")) or re.match(r"\d+\.\s", stripped):
            yield lineno, stripped
            continue
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", stripped):
            yield lineno, part


def verify(text: str, facts: dict[str, dict]) -> list[str]:
    problems = []
    for lineno, sentence in sentences(text):
        cited = [c.strip() for group in CITATION.findall(sentence) for c in group.split(",")]
        for c in cited:
            if c not in facts:
                problems.append(f"line {lineno}: cites {c}, which is not in the ledger")
        body = CITATION.sub(" ", sentence)
        body = DATE.sub(" ", body)
        body = re.sub(r"^\s*(\d+\.|[-*])\s+", " ", body)                 # list markers
        body = re.sub(r"\bF\d+\b|\[\d+\]", " ", body)
        found = list(NUMBER.finditer(body))
        if not found:
            continue
        if not cited:
            problems.append(f"line {lineno}: numbers without a citation: {sentence[:100]!r}")
            continue
        values = []
        for c in cited:
            fact = facts.get(c)
            if fact:
                values += ([fact["value"]] if isinstance(fact.get("value"), (int, float)) else []) + fact.get("values", [])
        for m in found:
            sign, whole, frac, unit = m.groups()
            written = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
            if unit in ("x", "×"):
                continue  # ratios like 2x are derived; cite the facts but do not require an exact match
            if not matches(written, len(frac or ""), unit, values):
                problems.append(f"line {lineno}: {m.group(0).strip()!r} does not match cited {', '.join(cited)}")
    return problems


def load_facts(path: str) -> dict[str, dict]:
    return {f["id"]: f for f in json.loads(Path(path).read_text())["facts"]}


def cmd_verify(args) -> int:
    problems = verify(Path(args.summary).read_text(), load_facts(args.facts))
    if problems:
        print(f"UNVERIFIED: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERIFIED: every number traces to a cited fact")
    return 0


def cmd_render(args) -> int:
    text = Path(args.summary).read_text()
    if not args.skip_verify:
        problems = verify(text, load_facts(args.facts))
        if problems:
            print("refusing to render: summary does not verify (run verify for details)")
            return 1
    facts = load_facts(args.facts)
    used = sorted({c.strip() for g in CITATION.findall(text) for c in g.split(",")}, key=lambda x: int(x[1:]))
    clean = re.sub(r"\s*" + CITATION.pattern, "", text)
    sources = sorted({facts[c]["source"] for c in used if c in facts})
    clean = clean.rstrip() + "\n\n---\n_Sources: " + "; ".join(Path(s).as_posix() for s in sources) + "_\n"
    Path(args.out).write_text(clean)
    print(f"rendered {args.out} ({len(used)} facts cited)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("reports", nargs="*")
    c.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    c.add_argument("--extra", help="JSON list of {label, value|text, source} business facts you supply")
    c.add_argument("--out", default=".dsguard/summaries/facts.json")
    c.set_defaults(func=cmd_collect)
    v = sub.add_parser("verify")
    v.add_argument("summary")
    v.add_argument("--facts", default=".dsguard/summaries/facts.json")
    v.set_defaults(func=cmd_verify)
    r = sub.add_parser("render")
    r.add_argument("summary")
    r.add_argument("--facts", default=".dsguard/summaries/facts.json")
    r.add_argument("--out", required=True)
    r.add_argument("--skip-verify", action="store_true")
    r.set_defaults(func=cmd_render)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
