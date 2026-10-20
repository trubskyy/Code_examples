"""Known-vulnerable dependencies via pip-audit (skipped, not passed, when it is missing)."""

from __future__ import annotations

import json
import shutil
import subprocess

from .core import CheckResult, Context


def run(ctx: Context) -> CheckResult:
    result = CheckResult("vulns")
    exe = shutil.which("pip-audit")
    if not exe:
        return result.skip("pip-audit not installed (pip install pip-audit); vulnerabilities NOT checked")
    requirements = [f for f in ctx.all_files if f.rsplit("/", 1)[-1].startswith("requirements") and f.endswith(".txt")]
    if not requirements:
        return result.skip("no requirements*.txt found; run pip-audit inside the project environment")
    for req in requirements:
        proc = subprocess.run([exe, "-r", str(ctx.root / req), "-f", "json", "--progress-spinner", "off"],
                              capture_output=True, text=True, check=False)
        try:
            data = json.loads(proc.stdout)
        except ValueError:
            result.notes.append(f"pip-audit failed on {req}: {proc.stderr.strip()[:300]}")
            continue
        deps = data.get("dependencies", []) if isinstance(data, dict) else data
        for dep in deps:
            for vuln in dep.get("vulns", []):
                fix = ", ".join(vuln.get("fix_versions", [])) or "no fix released"
                result.add("vulnerable-dependency", "block" if vuln.get("fix_versions") else "warn", req,
                           f"{dep['name']}=={dep.get('version')} {vuln.get('id')} (fixed in: {fix})")
    return result
