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
RANK = {"allow": 0, "review": 1, "unknown": 2, "deny": 3}


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


def classify_one(text: str, cfg: dict) -> str:
    if any(re.search(p, text) for p in cfg["deny"]):
        return "deny"
    if any(re.search(p, text) for p in cfg["review"]):
        return "review"
    if any(re.search(p, text) for p in cfg["allow"]):
        return "allow"
    return "unknown"


def classify(dist: dict, cfg: dict) -> tuple[str, str]:
    """Best category across dual-licence options, and a short label."""
    options = license_options(dist)
    if not options:
        return "unknown", "no license metadata"
    best = min(options, key=lambda o: RANK[classify_one(o, cfg)])
    label = " OR ".join(o[:60] for o in options)
    return classify_one(best, cfg), label


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
    seen = set()
    for dist in sorted(dists, key=lambda d: normalize(d["name"] or "")):
        name = normalize(dist["name"] or "?")
        if name in exceptions or (name, dist["version"]) in seen:
            continue
        seen.add((name, dist["version"]))
        category, label = classify(dist, cfg)
        is_direct = name in direct
        where = "direct" if is_direct else "transitive"
        package = f"{dist['name']}=={dist['version']}"
        if category == "deny":
            result.add("license-denied", "block", package, f"{where} dependency licensed {label}")
        elif category in ("review", "unknown"):
            result.add(f"license-{category}", "warn" if is_direct else "info", package,
                       f"{where} dependency licensed {label}; confirm it is acceptable")
    _hf_ids(ctx, result)
    return result
