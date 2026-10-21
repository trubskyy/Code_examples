"""dsguard command line: run checks, verify docs-only edits, apply/restore cleanup plans."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import cleanup, code, docs, exposure, gateway, leakage, licenses, onboard, pysec, secrets, sql, vulns
from .core import SEVERITY_ORDER, CheckResult, build_context, ensure_state_dir, find_root

CHECKS = {
    "gateway": gateway.run, "secrets": secrets.run, "exposure": exposure.run, "pysec": pysec.run,
    "licenses": licenses.run, "vulns": vulns.run, "sql": sql.run, "code": code.run, "leakage": leakage.run,
    "docs": docs.run,
    "cleanup": cleanup.run,
}
GROUPS = {
    "security": ["gateway", "secrets", "exposure", "pysec", "licenses", "vulns"],
    "quality": ["code", "leakage", "sql", "docs"],
    "commit": ["gateway", "secrets"],
    "all": list(CHECKS),
}


def expand(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names or ["all"]:
        for check in GROUPS.get(name, [name]):
            if check not in CHECKS:
                raise SystemExit(f"dsguard: unknown check {check!r}; choose from {', '.join([*CHECKS, *GROUPS])}")
            if check not in out:
                out.append(check)
    return out


def render_text(results: list[CheckResult], limit: int) -> str:
    counts = {s: sum(f.severity == s for r in results for f in r.findings) for s in SEVERITY_ORDER}
    verdict = "BLOCKED" if counts["block"] else ("WARNINGS" if counts["warn"] else "OK")
    lines = [f"dsguard: {verdict}  block={counts['block']} warn={counts['warn']} info={counts['info']}"]
    for result in results:
        data = result.to_dict()
        header = f"\n== {result.name}: {result.status}"
        if result.findings:
            header += f" ({len(result.findings)})"
        lines.append(header)
        lines.extend(f"   note: {n}" for n in result.notes)
        for i, f in enumerate(data["findings"]):
            if i == limit:
                lines.append(f"   … {len(result.findings) - limit} more (use --format json)")
                break
            loc = f["path"] + (f"[cell {f['cell']}]" if f["cell"] is not None else "") + (f":{f['line']}" if f["line"] else "")
            lines.append(f"   {f['severity'].upper():5}  {loc}  [{f['rule']}] {f['message']}")
    return "\n".join(lines)


def _save(root: Path, payload: dict) -> Path:
    reports = ensure_state_dir(root) / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    text = json.dumps(payload, indent=2)
    path.write_text(text)
    (reports / "latest.json").write_text(text)
    return path


def cmd_run(args) -> int:
    root = find_root(Path(args.root))
    ctx = build_context(root, args.changed_since, args.staged, args.paths)
    results = []
    for name in expand(args.checks):
        try:
            results.append(CHECKS[name](ctx))
        except Exception as exc:  # one broken check must not hide the others
            failed = CheckResult(name, status="error")
            failed.notes.append(f"{type(exc).__name__}: {exc}")
            results.append(failed)
    blocked = any(f.severity == "block" for r in results for f in r.findings)
    payload = {
        "root": str(root), "scope": args.changed_since or ("staged" if args.staged else "all"),
        "files_in_scope": len(ctx.files), "blocked": blocked, "results": [r.to_dict() for r in results],
    }
    saved = None if args.no_save else _save(root, payload)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(results, args.limit))
        if saved:
            print(f"\nfull report: {saved}")
    return 1 if blocked else 0


def cmd_verify_docs(args) -> int:
    root = find_root(Path(args.root))
    result = docs.verify_docs_only(build_context(root), args.ref)
    print(render_text([result], 200))
    return 1 if result.findings else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run checks or groups: " + ", ".join([*GROUPS, *CHECKS]))
    run.add_argument("checks", nargs="*")
    run.add_argument("--root", default=".")
    scope = run.add_mutually_exclusive_group()
    scope.add_argument("--changed-since", metavar="REF", help="only files changed since REF (plus uncommitted)")
    scope.add_argument("--staged", action="store_true", help="only staged files (pre-commit)")
    run.add_argument("--paths", nargs="*", help="limit to these files/directories")
    run.add_argument("--format", choices=["text", "json"], default="text")
    run.add_argument("--limit", type=int, default=40, help="max findings per check in text output")
    run.add_argument("--no-save", action="store_true", help="do not write .dsguard/reports")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify-docs-only", help="fail if edits since REF changed code, not just docs")
    verify.add_argument("--ref", default="HEAD")
    verify.add_argument("--root", default=".")
    verify.set_defaults(func=cmd_verify_docs)

    gw = sub.add_parser("suggest-gateway", help="draft [llm_gateway] config from existing gateway client code")
    gw.add_argument("path", help="gateway package directory or .py file")
    gw.add_argument("--root", default=".")
    gw.set_defaults(func=lambda a: print(onboard.suggest(Path(a.path), find_root(Path(a.root)))) or 0)

    apply = sub.add_parser("apply-cleanup", help="execute an approved cleanup plan (reversible)")
    apply.add_argument("plan")
    apply.add_argument("--root", default=".")
    apply.set_defaults(func=lambda a: cleanup.apply_plan(find_root(Path(a.root)), Path(a.plan)))

    restore = sub.add_parser("restore-cleanup", help="undo an apply-cleanup by its timestamp")
    restore.add_argument("stamp")
    restore.add_argument("--root", default=".")
    restore.set_defaults(func=lambda a: cleanup.restore(find_root(Path(a.root)), a.stamp))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
