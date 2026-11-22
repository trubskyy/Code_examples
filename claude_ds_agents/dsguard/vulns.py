"""Known-vulnerable dependencies via pip-audit (skipped, not passed, when it is missing).

pip-audit normally has pip resolve a requirements file, which builds packages and so runs the
project's code. It gets only the exact `name==version` pins here, and never uses pip.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .core import CheckResult, Context

PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([0-9][A-Za-z0-9.+!_-]*)(?=\s*(?:[;#\\]|--|$)|\s+#)")


def pins(text: str) -> tuple[list[str], int]:
    """(exact pins, number of other requirement lines)."""
    found, other = {}, 0
    for line in map(str.strip, text.splitlines()):
        if not line or line.startswith(("#", "--hash")):
            continue
        m = PIN.match(line)
        if m:
            found.setdefault(m.group(1).lower(), f"{m.group(1)}=={m.group(2)}")
        else:
            other += 1
    return list(found.values()), other


def run(ctx: Context) -> CheckResult:
    result = CheckResult("vulns")
    exe = shutil.which("pip-audit")
    if not exe:
        return result.skip("pip-audit not installed (pip install pip-audit); vulnerabilities NOT checked")
    requirements = [f for f in ctx.all_files if f.rsplit("/", 1)[-1].startswith("requirements") and f.endswith(".txt")]
    if not requirements:
        return result.skip("no requirements*.txt found; run pip-audit inside the project environment")
    audited = False
    for req in requirements:
        pinned, other = pins(ctx.read(req) or "")
        if other:
            result.notes.append(f"{req}: {other} line(s) without an exact pin were NOT checked")
        if not pinned:
            continue
        audited = True
        with tempfile.TemporaryDirectory() as tmp:
            safe = Path(tmp) / "requirements.txt"
            safe.write_text("\n".join(pinned) + "\n")
            proc = subprocess.run([exe, "-r", str(safe), "--no-deps", "--disable-pip", "-f", "json",
                                   "--progress-spinner", "off"], capture_output=True, text=True, check=False, cwd=tmp)
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
    if not audited:
        return result.skip("no exactly pinned requirements; vulnerabilities NOT checked")
    return result
