#!/usr/bin/env python3
"""Golden-set evaluation for LLM components, with regression diffs against a baseline.

The harness never calls a provider itself. It imports YOUR function (which must use
the LLM gateway) and scores what it returns, so gateway policy still applies.

  python llm_eval.py run --cases evals/golden.jsonl --target myproj.extract:run_case \
      [--judge myproj.judges:faithfulness] [--baseline .dsguard/evals/last.json] \
      [--repeat 3] [--concurrency 4] [--price-in 3 --price-out 15]
  python llm_eval.py compare old.json new.json

Case file (JSONL), one object per line:
  {"id": "c1", "input": {...} | "text", "expected": "..." | {...},
   "checks": [{"type": "fields"}], "tags": ["edge"], "human_score": 1}

Check types: exact, normalized, contains, regex, json_valid, fields (dict field accuracy,
optional "keys"), numeric (with "tol"). With no checks, "expected" picks fields (dict)
or normalized (anything else). The target returns the output, or
{"output": ..., "usage": {"input_tokens": n, "output_tokens": m}, "model": "..."}.
A judge returns a 0-1 float or {"score": float, "reason": str}.
Stdlib only, so it runs in any project environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

STATE = Path(".dsguard")


def load_callable(spec: str):
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected module:function, got {spec!r}")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), attr)


def load_cases(path: str) -> list[dict]:
    cases = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            case = json.loads(line)
            case.setdefault("id", f"line{lineno}")
            cases.append(case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("case ids must be unique")
    return cases


def normalize(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower()).strip(" .")


def as_json(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(value).strip())
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def run_check(check: dict, output, expected) -> tuple[bool, float, str]:
    """Return (passed, score in [0, 1], detail)."""
    kind = check["type"]
    want = check.get("value", expected)
    if kind == "exact":
        ok = output == want
        return ok, float(ok), "" if ok else "differs"
    if kind == "normalized":
        ok = normalize(output) == normalize(want)
        return ok, float(ok), "" if ok else f"got {normalize(output)[:80]!r}"
    if kind == "contains":
        needles = want if isinstance(want, list) else [want]
        hits = [n for n in needles if normalize(n) in normalize(output)]
        return len(hits) == len(needles), len(hits) / len(needles), f"missing {[n for n in needles if n not in hits]}"
    if kind == "regex":
        ok = re.search(want, str(output)) is not None
        return ok, float(ok), "" if ok else "no match"
    if kind == "json_valid":
        ok = as_json(output) is not None
        return ok, float(ok), "" if ok else "not JSON"
    if kind == "fields":
        got, exp = as_json(output), want
        if not isinstance(got, dict) or not isinstance(exp, dict):
            return False, 0.0, "output or expected is not an object"
        keys = check.get("keys") or list(exp)
        wrong = [k for k in keys if normalize(got.get(k)) != normalize(exp.get(k))]
        score = 1 - len(wrong) / len(keys) if keys else 1.0
        return not wrong, score, f"wrong fields {wrong}" if wrong else ""
    if kind == "numeric":
        try:
            got = float(re.findall(r"-?\d+(?:\.\d+)?", str(output))[0]) if not isinstance(output, (int, float)) else output
        except IndexError:
            return False, 0.0, "no number in output"
        ok = math.isclose(got, float(want), rel_tol=check.get("tol", 0.0), abs_tol=check.get("abs_tol", 0.0))
        return ok, float(ok), "" if ok else f"got {got}"
    raise SystemExit(f"unknown check type {kind!r}")


def default_checks(case: dict) -> list[dict]:
    if case.get("checks"):
        return case["checks"]
    if "expected" not in case:
        return []
    return [{"type": "fields"}] if isinstance(case["expected"], dict) else [{"type": "normalized"}]


def cache_key(target: str, case: dict, version: str, attempt: int) -> str:
    raw = json.dumps([target, case.get("input"), version, attempt], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def call_target(fn, spec: str, case: dict, cache_dir: Path | None, version: str, attempt: int) -> dict:
    key = cache_key(spec, case, version, attempt)
    cached = cache_dir / f"{key}.json" if cache_dir else None
    if cached and cached.exists():
        return json.loads(cached.read_text()) | {"cached": True}
    start = time.perf_counter()
    try:
        raw = fn(case.get("input"))
        error = None
    except Exception as exc:  # a failing case is a result, not a crash
        raw, error = None, f"{type(exc).__name__}: {exc}"
    record = {"latency_s": time.perf_counter() - start, "error": error, "cached": False}
    if isinstance(raw, dict) and "output" in raw:
        record.update(output=raw["output"], usage=raw.get("usage") or {}, model=raw.get("model"))
    else:
        record.update(output=raw, usage={}, model=None)
    if cached and error is None:
        cached.write_text(json.dumps(record, default=str))
    return record


def kappa(a: list[int], b: list[int]) -> float:
    if not a:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    labels = set(a) | set(b)
    expected = sum((a.count(k) / len(a)) * (b.count(k) / len(b)) for k in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(results: list[dict], price_in: float, price_out: float) -> dict:
    scored = [r for r in results if r["checks"]]
    tags: dict[str, list[bool]] = {}
    for r in scored:
        for tag in r.get("tags", []):
            tags.setdefault(tag, []).append(r["passed"])
    per_check: dict[str, list[bool]] = {}
    for r in scored:
        for c in r["checks"]:
            per_check.setdefault(c["type"], []).append(c["passed"])
    latencies = [a["latency_s"] for r in results for a in r["attempts"] if not a.get("cached")]
    tok_in = sum(a["usage"].get("input_tokens", 0) for r in results for a in r["attempts"])
    tok_out = sum(a["usage"].get("output_tokens", 0) for r in results for a in r["attempts"])
    summary = {
        "cases": len(results),
        "scored_cases": len(scored),
        "pass_rate": sum(r["passed"] for r in scored) / len(scored) if scored else float("nan"),
        "mean_score": statistics.mean(r["score"] for r in scored) if scored else float("nan"),
        "error_rate": sum(any(a["error"] for a in r["attempts"]) for r in results) / len(results) if results else 0,
        "by_tag": {t: sum(v) / len(v) for t, v in sorted(tags.items())},
        "by_check": {t: sum(v) / len(v) for t, v in sorted(per_check.items())},
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "est_cost": (tok_in * price_in + tok_out * price_out) / 1e6 if (price_in or price_out) else None,
        "models": sorted({a["model"] for r in results for a in r["attempts"] if a.get("model")}),
    }
    repeated = [r for r in results if len(r["attempts"]) > 1]
    if repeated:
        summary["consistency"] = sum(r["consistent"] for r in repeated) / len(repeated)
    judged = [r for r in results if r.get("judge_score") is not None]
    if judged:
        summary["judge_mean"] = statistics.mean(r["judge_score"] for r in judged)
        paired = [r for r in judged if r.get("human_score") is not None]
        if len(paired) >= 5:
            a = [int(r["judge_score"] >= 0.5) for r in paired]
            b = [int(float(r["human_score"]) >= 0.5) for r in paired]
            summary["judge_human_agreement"] = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
            summary["judge_human_kappa"] = kappa(a, b)
            summary["judge_human_pairs"] = len(paired)
    return summary


def compare(old: dict, new: dict) -> dict:
    before = {r["id"]: r for r in old["results"]}
    after = {r["id"]: r for r in new["results"]}
    common = sorted(set(before) & set(after))
    regressions = [i for i in common if before[i]["passed"] and not after[i]["passed"]]
    fixes = [i for i in common if not before[i]["passed"] and after[i]["passed"]]
    delta = {}
    for key in ("pass_rate", "mean_score", "error_rate", "latency_p95_s", "est_cost", "judge_mean", "consistency"):
        a, b = old["summary"].get(key), new["summary"].get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[key] = b - a
    return {"regressions": regressions, "fixes": fixes, "only_in_old": sorted(set(before) - set(after)),
            "only_in_new": sorted(set(after) - set(before)), "delta": delta}


def markdown(run: dict, diff: dict | None) -> str:
    s = run["summary"]
    lines = [f"# LLM eval: {run['name']}\n", f"target `{run['target']}`, {s['cases']} cases, version `{run['version']}`\n",
             "\n| metric | value |\n|---|---|"]
    for key, value in s.items():
        if not isinstance(value, dict):
            lines.append(f"| {key} | {value if not isinstance(value, float) else round(value, 4)} |")
    for title, key in (("By tag", "by_tag"), ("By check", "by_check")):
        if s[key]:
            lines += [f"\n## {title}\n", "| name | pass rate |", "|---|---|"]
            lines += [f"| {k} | {v:.3f} |" for k, v in s[key].items()]
    failures = [r for r in run["results"] if r["checks"] and not r["passed"]]
    lines += [f"\n## Failures ({len(failures)})\n"]
    for r in failures[:30]:
        detail = "; ".join(c["detail"] for c in r["checks"] if not c["passed"] and c["detail"])
        err = next((a["error"] for a in r["attempts"] if a["error"]), None)
        lines.append(f"- `{r['id']}` {err or detail}")
    if diff:
        lines += ["\n## Versus baseline\n", f"- regressions (pass → fail): {diff['regressions'] or 'none'}",
                  f"- fixes (fail → pass): {diff['fixes'] or 'none'}"]
        lines += [f"- Δ {k}: {v:+.4g}" for k, v in diff["delta"].items()]
    return "\n".join(lines) + "\n"


def cmd_run(args) -> int:
    cases = load_cases(args.cases)
    fn = load_callable(args.target)
    judge = load_callable(args.judge) if args.judge else None
    cache_dir = None if args.no_cache else STATE / "llm_cache"
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not (STATE / ".gitignore").exists():
            (STATE / ".gitignore").write_text("*\n")

    def evaluate(case: dict) -> dict:
        attempts = [call_target(fn, args.target, case, cache_dir, args.version, k) for k in range(args.repeat)]
        first = attempts[0]
        checks = []
        if first["error"] is None:
            for check in default_checks(case):
                passed, score, detail = run_check(check, first["output"], case.get("expected"))
                checks.append({"type": check["type"], "passed": passed, "score": score, "detail": detail})
        else:
            checks = [{"type": c["type"], "passed": False, "score": 0.0, "detail": "error"} for c in default_checks(case)]
        outputs = [json.dumps(a["output"], sort_keys=True, default=str) for a in attempts]
        record = {
            "id": case["id"], "tags": case.get("tags", []), "attempts": attempts, "checks": checks,
            "passed": bool(checks) and all(c["passed"] for c in checks),
            "score": statistics.mean(c["score"] for c in checks) if checks else None,
            "consistent": len(set(outputs)) == 1, "human_score": case.get("human_score"),
            "output": first["output"],
        }
        if judge and first["error"] is None:
            verdict = judge(case, first["output"])
            record["judge_score"] = float(verdict["score"] if isinstance(verdict, dict) else verdict)
            record["judge_reason"] = verdict.get("reason") if isinstance(verdict, dict) else None
        return record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(evaluate, cases))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = {"name": args.name or stamp, "target": args.target, "cases_file": args.cases, "version": args.version,
           "created": stamp, "summary": summarize(results, args.price_in, args.price_out), "results": results}
    diff = compare(json.loads(Path(args.baseline).read_text()), run) if args.baseline else None
    if diff:
        run["baseline"] = {"path": args.baseline, **diff}
    out = STATE / "evals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run['name']}.json"
    path.write_text(json.dumps(run, indent=2, default=str))
    (out / f"{run['name']}.md").write_text(markdown(run, diff))
    (out / "last.json").write_text(json.dumps(run, indent=2, default=str))
    s = run["summary"]
    print(f"pass_rate {s['pass_rate']:.3f}  mean_score {s['mean_score']:.3f}  errors {s['error_rate']:.1%}  "
          f"p95 {s['latency_p95_s']:.2f}s  report {out / (run['name'] + '.md')}")
    if diff:
        print(f"vs baseline: {len(diff['regressions'])} regressions, {len(diff['fixes'])} fixes")
    return 1 if diff and diff["regressions"] else 0


def cmd_compare(args) -> int:
    old, new = (json.loads(Path(p).read_text()) for p in (args.old, args.new))
    diff = compare(old, new)
    print(json.dumps(diff, indent=2))
    return 1 if diff["regressions"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--cases", required=True)
    run.add_argument("--target", required=True, help="module:function taking the case input")
    run.add_argument("--judge", help="module:function(case, output) -> score")
    run.add_argument("--baseline", help="previous run JSON; exit 1 on any pass->fail regression")
    run.add_argument("--name")
    run.add_argument("--version", default="", help="prompt/model version label; part of the cache key")
    run.add_argument("--repeat", type=int, default=1, help="calls per case, to measure consistency")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--price-in", type=float, default=0.0, help="cost per million input tokens")
    run.add_argument("--price-out", type=float, default=0.0, help="cost per million output tokens")
    run.set_defaults(func=cmd_run)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("old")
    cmp_.add_argument("new")
    cmp_.set_defaults(func=cmd_compare)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
