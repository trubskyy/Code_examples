"""Cleanup candidates (never deletes), plus a reversible apply/restore for an approved plan."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

from .core import CheckResult, Context, ensure_state_dir, match_any, python_sources

JUNK_DIRS = {
    "__pycache__": "bytecode cache", ".ipynb_checkpoints": "notebook checkpoints",
    ".pytest_cache": "test cache", ".mypy_cache": "type-check cache", ".ruff_cache": "lint cache",
    ".venv": "virtualenv", "venv": "virtualenv", "node_modules": "node packages",
    "wandb": "W&B run logs", "lightning_logs": "Lightning logs", "catboost_info": "CatBoost logs",
    "mlruns": "MLflow local runs", ".hypothesis": "Hypothesis cache",
}
JUNK_FILES = [
    ("*.pyc", "bytecode"), ("*.pyo", "bytecode"), (".DS_Store", "macOS metadata"), ("Thumbs.db", "Windows metadata"),
    ("*.log", "log file"), ("*.tmp", "temp file"), ("*~", "editor backup"), ("*.swp", "editor swap"),
    ("*.orig", "merge leftover"), ("*.rej", "patch leftover"), ("*.bak", "backup copy"), ("nohup.out", "nohup output"),
]
SUSPECT_NAME = re.compile(
    r"(?i)(^untitled\d*\.ipynb$|[ _-]copy\d*\.|\(\d+\)\.|_old\.|_bak\.|_backup|^tmp_|^scratch|_final\d*\.|_v\d+\.(py|ipynb|sql)$)"
)
NOT_ORPHANS = {"__init__.py", "__main__.py", "setup.py", "conftest.py", "manage.py", "noxfile.py", "wsgi.py", "asgi.py"}
KEEP_EMPTY = {"__init__.py", ".gitkeep", ".keep", "py.typed"}


def _junk_dir(rel: str) -> tuple[str, str] | None:
    parts = PurePosixPath(rel).parts
    for i, part in enumerate(parts[:-1]):
        if part in JUNK_DIRS or part.endswith(".egg-info"):
            return "/".join(parts[: i + 1]), JUNK_DIRS.get(part, "packaging metadata")
    return None


def _module_names(rel: str) -> set[str]:
    parts = list(PurePosixPath(rel).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return {".".join(parts[i:]) for i in range(len(parts))} if parts else set()


def _imported_names(ctx: Context) -> set[str]:
    names: set[str] = set()
    for src in python_sources(ctx, ctx.all_files):
        if not src.tree:
            continue
        package = list(PurePosixPath(src.path).parent.parts)
        for node in ast.walk(src.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    names.update(".".join(parts[: i + 1]) for i in range(len(parts)))
            elif isinstance(node, ast.ImportFrom):
                base = (package[: len(package) - node.level + 1] if node.level else [])
                module = ".".join(base + ([node.module] if node.module else []))
                names.add(module)
                names.update(f"{module}.{a.name}".lstrip(".") for a in node.names)
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) == "import_module":
                if node.args and isinstance(node.args[0], ast.Constant):
                    names.add(str(node.args[0].value))
    return names


def _orphans(ctx: Context, result: CheckResult, keep: list[str]) -> None:
    imported = _imported_names(ctx)
    other_text = " ".join(
        ctx.read(f) or "" for f in ctx.all_files
        if not f.endswith(".py") and ctx.size(f) < 500_000 and not f.endswith((".csv", ".parquet", ".jsonl"))
    )
    entrypoints = ctx.config["cleanup"]["entrypoints"]
    for rel in ctx.files:
        name = PurePosixPath(rel).name
        if not rel.endswith(".py") or name in NOT_ORPHANS or name.startswith("test_") or name.endswith("_test.py"):
            continue
        if match_any(rel, entrypoints + keep):
            continue
        text = ctx.read(rel) or ""
        if re.search(r"__name__\s*==\s*['\"]__main__['\"]", text):
            continue
        if _module_names(rel) & imported:
            continue
        stem = PurePosixPath(rel).stem
        if re.search(rf"\b{re.escape(stem)}\b", other_text):
            continue
        result.add("possibly-unused-module", "info", rel,
                   "not imported anywhere, no __main__ guard, not referenced in configs/notebooks; review before removing")


def run(ctx: Context) -> CheckResult:
    result = CheckResult("cleanup")
    cfg = ctx.config["cleanup"]
    keep = cfg["keep"]
    tracked = ctx.tracked or set()
    seen_dirs: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    for rel in ctx.files:
        if match_any(rel, keep):
            continue
        is_tracked = rel in tracked
        junk = _junk_dir(rel)
        if junk:
            directory, why = junk
            if directory not in seen_dirs:
                seen_dirs.add(directory)
                result.add("junk-dir", "warn" if is_tracked else "info", directory + "/",
                           f"{why}: {'committed, untrack and gitignore' if is_tracked else 'not gitignored'}")
            continue
        name = PurePosixPath(rel).name
        why = next((w for pattern, w in JUNK_FILES if PurePosixPath(name).match(pattern)), None)
        if why:
            result.add("junk-file", "warn" if is_tracked else "info", rel,
                       f"{why}: {'committed, untrack and gitignore' if is_tracked else 'not gitignored'}")
            continue
        if SUSPECT_NAME.search(name):
            result.add("suspect-name", "info", rel, "name suggests a scratch/backup/duplicate copy; review")
        size = ctx.size(rel)
        if size == 0 and name not in KEEP_EMPTY:
            result.add("empty-file", "info", rel, "empty file")
        if size > cfg["large_file_mb"] * 1e6:
            result.add("large-file", "warn" if is_tracked else "info", rel,
                       f"{size / 1e6:.1f} MB; move to object storage, DVC or Git LFS")
        if 1024 < size < 50e6:
            hashes[hashlib.sha1((ctx.root / rel).read_bytes()).hexdigest()].append(rel)
    for copies in hashes.values():
        if len(copies) > 1:
            result.add("identical-files", "info", copies[0], f"identical content: {', '.join(copies)}")
    _orphans(ctx, result, keep)
    return result


def _inside(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"dsguard: refusing path outside the project: {rel}")
    return path


def apply_plan(root: Path, plan_file: Path) -> int:
    """Execute an approved plan: move 'delete' items to .dsguard/trash, 'untrack' via git rm --cached.

    Plan format: {"items": [{"path": "...", "action": "delete" | "untrack"}], "gitignore": ["pattern", ...]}
    """
    root = root.resolve()
    plan = json.loads(plan_file.read_text())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trash = ensure_state_dir(root) / "trash" / stamp
    manifest = []
    for item in plan.get("items", []):
        rel, action = item["path"].rstrip("/"), item["action"]
        path = _inside(root, rel)
        if not path.exists():
            print(f"skip (missing): {rel}")
            continue
        if action == "delete":
            dest = trash / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            manifest.append(rel)
            print(f"moved to trash: {rel}")
        elif action == "untrack":
            subprocess.run(["git", "-C", str(root), "rm", "-r", "-q", "--cached", "--", rel], check=True)
            print(f"untracked (file kept on disk): {rel}")
        else:
            raise SystemExit(f"dsguard: unknown action {action!r} for {rel}")
    patterns = plan.get("gitignore", [])
    if patterns:
        gitignore = root / ".gitignore"
        existing = gitignore.read_text().splitlines() if gitignore.exists() else []
        new = [p for p in patterns if p not in existing]
        if new:
            with gitignore.open("a") as fh:
                fh.write(("\n" if existing and existing[-1] else "") + "\n".join(new) + "\n")
            print(f"added to .gitignore: {', '.join(new)}")
    if manifest:
        (trash / "manifest.json").write_text(json.dumps({"items": manifest}, indent=2))
        print(f"\nrestore with: dsguard restore-cleanup {stamp}")
    return 0


def restore(root: Path, stamp: str) -> int:
    root = root.resolve()
    trash = root / ".dsguard" / "trash" / stamp
    manifest = json.loads((trash / "manifest.json").read_text())
    for rel in manifest["items"]:
        dest = _inside(root, rel)
        if dest.exists():
            print(f"skip (exists): {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trash / rel), str(dest))
        print(f"restored: {rel}")
    return 0
