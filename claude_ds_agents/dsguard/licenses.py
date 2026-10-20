"""License policy for installed Python packages, plus Hugging Face models/datasets to verify."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from .core import CheckResult, Context, call_name, const_str, kwarg, python_sources

DUMP = r"""
import importlib.metadata as m, json
out = []
for d in m.distributions():
    md = d.metadata
    out.append({"name": md.get("Name"), "version": d.version,
                "expression": md.get("License-Expression"), "license": md.get("License"),
                "classifiers": [c for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]})
print(json.dumps(out))
"""
HF_LOADERS = {"from_pretrained", "pipeline", "load_dataset", "SentenceTransformer", "CrossEncoder",
              "snapshot_download", "hf_hub_download"}
# Families from least to most restrictive. Bucket "always-free" means use, modify and ship
# freely with attribution; everything else is "potential-issues" with the obligation spelled out.
# Patterns are tried most-restrictive first so "LGPL" is not read as "GPL" and so on.
FAMILIES = [
    # (family, bucket, rank, patterns, what it means)
    ("permissive", "always-free", 0,
     [r"\bMIT", r"\bBSD", r"Apache", r"\bISC", r"\bPSF", r"Python Software Foundation", r"Unlicense", r"\bZlib",
      r"\bCC0", r"\bHPND", r"Public Domain", r"\b0BSD", r"BSL-1\.0", r"Boost Software", r"\bUPL",
      r"Historical Permission", r"\bWTFPL", r"\bCC-BY(?!-)", r"\bCC-BY-4"],
     "free to use, modify and ship; keep the copyright/license notice"),
    ("weak-copyleft", "potential-issues", 1,
     [r"\bLGPL", r"Lesser General Public", r"Library General Public", r"\bMPL", r"Mozilla", r"\bEPL", r"Eclipse",
      r"\bCDDL", r"\bOSL", r"Artistic"],
     "fine to use unmodified as a library; if you modify the library itself and distribute it, "
     "those changes must be released under the same license"),
    ("unknown", "potential-issues", 2, [],
     "no recognizable license metadata; check the project's LICENSE file before relying on it"),
    ("strong-copyleft", "potential-issues", 3,
     [r"\bGPL", r"(?<!Lesser )(?<!Library )General Public License", r"CC-BY-SA"],
     "internal use is usually fine; if you distribute software containing it (installers, shipped "
     "containers, client code) the whole program may have to be released under the GPL"),
    ("network-copyleft", "potential-issues", 4,
     [r"\bAGPL", r"Affero"],
     "like GPL, and serving it over a network (an API, a web app) counts as distribution"),
    ("restricted", "potential-issues", 5,
     [r"\bSSPL", r"Server Side Public", r"Commons Clause", r"\bBUSL", r"Business Source", r"CC-BY-NC",
      r"NonCommercial", r"Non-Commercial", r"Elastic License", r"Research", r"Proprietary"],
     "not open source: limits commercial use or offering it as a service; needs legal review"),
]
FAMILY = {f[0]: f for f in FAMILIES}
MATCH_ORDER = sorted((f for f in FAMILIES if f[3]), key=lambda f: -f[2])


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def find_python(root: Path, cfg: dict) -> tuple[str, bool]:
    if cfg["python"]:
        return cfg["python"], True
    for candidate in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe"):
        if (root / candidate).exists():
            return str(root / candidate), True
    return sys.executable, False


def license_options(dist: dict) -> list[str]:
    if dist.get("expression"):
        return [p.strip(" ()") for p in re.split(r"\s+OR\s+", dist["expression"])]
    classifiers = [c.split("::")[-1].strip() for c in dist.get("classifiers", [])]
    classifiers = [c for c in classifiers if c != "OSI Approved"]
    if classifiers:
        return classifiers
    text = (dist.get("license") or "").strip()
    if text and len(text) <= 120 and "\n" not in text:
        return [text]
    return [text[:200]] if text else []


def classify_one(text: str) -> str:
    for family, _, _, patterns, _ in MATCH_ORDER:
        if any(re.search(p, text) for p in patterns):
            return family
    return "unknown"


def classify(dist: dict) -> tuple[str, str]:
    """License family (least restrictive across dual-license options) and a short label."""
    options = license_options(dist)
    if not options:
        return "unknown", "no license metadata"
    best = min(options, key=lambda o: FAMILY[classify_one(o)][2])
    return classify_one(best), " OR ".join(o[:60] for o in options)


def direct_dependencies(root: Path) -> set[str]:
    names: set[str] = set()
    for req in root.glob("**/requirements*.txt"):
        if ".venv" in req.parts or "venv" in req.parts:
            continue
        for line in req.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
            if m and not line.lstrip().startswith(("-", "#")):
                names.add(normalize(m.group(1)))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text())
        except tomllib.TOMLDecodeError:
            data = {}
        project = data.get("project", {})
        specs = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            specs.extend(extra)
        specs.extend(data.get("tool", {}).get("poetry", {}).get("dependencies", {}).keys())
        for spec in specs:
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
            if m and m.group(1).lower() != "python":
                names.add(normalize(m.group(1)))
    return names


def _hf_ids(ctx: Context, result: CheckResult) -> None:
    seen = set()
    for src in python_sources(ctx):
        if not src.tree:
            continue
        for node in ast.walk(src.tree):
            if not isinstance(node, ast.Call) or call_name(node.func) not in HF_LOADERS:
                continue
            candidates = [const_str(a) for a in node.args[:2]]
            for key in ("model", "repo_id", "path", "pretrained_model_name_or_path"):
                kw = kwarg(node, key)
                candidates.append(const_str(kw.value) if kw else None)
            for value in candidates:
                if value and "/" in value and not value.startswith((".", "/", "~")) and value not in seen:
                    seen.add(value)
                    result.add("hub-asset-license", "info", src.path,
                               f"verify the license/terms of hub asset '{value}' (model cards can bar commercial use)",
                               node.lineno, src.cell)


def run(ctx: Context) -> CheckResult:
    result = CheckResult("licenses")
    cfg = ctx.config["licenses"]
    python, project_env = find_python(ctx.root, cfg)
    if not project_env:
        result.notes.append(f"no project venv found; scanned {python}. Set licenses.python to the project interpreter")
    proc = subprocess.run([python, "-c", DUMP], capture_output=True, text=True, check=False)
    try:
        dists = json.loads(proc.stdout)
    except ValueError:
        result.status = "error"
        result.notes.append(f"could not list packages with {python}: {proc.stderr.strip()[:300]}")
        return result
    direct = direct_dependencies(ctx.root)
    exceptions = {normalize(k): v for k, v in cfg["exceptions"].items()}
    block = set(cfg["block_families"])
    seen, counts = set(), {"always-free": 0, "potential-issues": 0}
    for dist in sorted(dists, key=lambda d: normalize(d["name"] or "")):
        name = normalize(dist["name"] or "?")
        if (name, dist["version"]) in seen:
            continue
        seen.add((name, dist["version"]))
        family, label = classify(dist)
        _, bucket, rank, _, meaning = FAMILY[family]
        if name in exceptions:
            bucket = "always-free"
        counts[bucket] += 1
        if bucket == "always-free":
            continue
        is_direct = name in direct
        if family in block:
            severity = "block"
        elif rank >= 3 or (is_direct and rank == 2):
            severity = "warn"
        else:
            severity = "info"
        result.add(f"license-{family}", severity, f"{dist['name']}=={dist['version']}",
                   f"{'direct' if is_direct else 'transitive'} dependency, {label}: {meaning}")
    result.notes.append(f"{counts['always-free']} packages always free, {counts['potential-issues']} with potential issues"
                        + ("" if block else "; nothing blocks until licenses.block_families is set"))
    _hf_ids(ctx, result)
    return result
