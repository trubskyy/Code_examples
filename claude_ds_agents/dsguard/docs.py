"""Documentation coverage, stale docstring parameters, and the docs-only edit verifier."""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

from .core import CheckResult, Context, cell_text, git, notebook_cells, parse, python_sources, strip_magics

NUMPY_PARAMS = re.compile(r"(?ms)^\s*(?:Parameters|Other Parameters)\s*\n\s*-{3,}\s*\n(.*?)(?=^\s*\w[\w ]*\n\s*-{3,}|\Z)")
GOOGLE_PARAMS = re.compile(r"(?ms)^(?:Args|Arguments|Parameters):\s*\n(.*?)(?=^\S[\w ]*:\s*$|\Z)")
SPHINX_PARAM = re.compile(r":param\s+(?:[\w\[\], .]+\s+)?\*{0,2}(\w+):")
DOC_EXTS = (".md", ".rst")


def documented_params(doc: str) -> set[str] | None:
    """Parameter names a docstring documents, or None if it has no parameter section."""
    names: set[str] = set()
    found = False
    m = NUMPY_PARAMS.search(doc)
    if m:
        found = True
        # get_docstring() dedents, so entries start at column 0 and descriptions are indented.
        for line in m.group(1).splitlines():
            if line.strip() and not line[0].isspace():
                names.update(n.strip(" *") for n in line.split(":")[0].split(","))
    m = GOOGLE_PARAMS.search(doc)
    if m:
        found = True
        indent = None
        for line in m.group(1).splitlines():
            entry = re.match(r"^(\s+)\*{0,2}(\w+)\s*(?:\(.*?\))?\s*:", line)
            if entry and (indent is None or len(entry.group(1)) == indent):
                indent = len(entry.group(1))
                names.add(entry.group(2))
    sphinx = SPHINX_PARAM.findall(doc)
    if sphinx:
        found = True
        names.update(sphinx)
    return names if found else None


def signature_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names - {"self", "cls"}


def _public(name: str) -> bool:
    return not name.startswith("_")


def run(ctx: Context) -> CheckResult:
    result = CheckResult("docs")
    cfg = ctx.config["docs"]
    total = documented = 0
    for src in python_sources(ctx):
        if src.tree is None or src.cell is not None:
            continue
        name = PurePosixPath(src.path).name
        is_test = name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
        if cfg["require_module_docstring"] and not is_test and src.source.strip() and ast.get_docstring(src.tree) is None:
            result.add("missing-module-docstring", "info", src.path, "module has no docstring")
        scopes = [(src.tree, True)]
        while scopes:
            scope, public_scope = scopes.pop()
            for node in scope.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                public = public_scope and _public(node.name) and not is_test
                doc = ast.get_docstring(node)
                if public:
                    total += 1
                    documented += doc is not None
                    if doc is None:
                        kind = "class" if isinstance(node, ast.ClassDef) else "function"
                        result.add("missing-docstring", "info", src.path, f"public {kind} {node.name} has no docstring", node.lineno)
                if isinstance(node, ast.ClassDef):
                    scopes.append((node, public))
                elif doc:
                    params = documented_params(doc)
                    if params is not None:
                        actual = signature_params(node)
                        stale = sorted(params - actual)
                        missing = sorted(actual - params)
                        if stale:
                            result.add("stale-docstring-param", "warn", src.path,
                                       f"{node.name}() documents {', '.join(stale)} which it does not take", node.lineno)
                        if missing and public:
                            result.add("undocumented-param", "info", src.path,
                                       f"{node.name}() does not document {', '.join(missing)}", node.lineno)
    if total:
        coverage = documented / total
        if coverage < cfg["min_public_coverage"]:
            result.add("docstring-coverage", "warn", ".", f"public docstring coverage {coverage:.0%} "
                       f"({documented}/{total}) is below {cfg['min_public_coverage']:.0%}")
        result.notes.append(f"public docstring coverage {coverage:.0%} ({documented}/{total})")
    if not any(f.lower().startswith("readme") for f in ctx.all_files):
        result.add("missing-readme", "warn", ".", "project has no README at its root")
    for rel in ctx.files:
        if rel.endswith(".ipynb"):
            cells = notebook_cells(ctx.read(rel) or "")
            if cells and not any(c.get("cell_type") == "markdown" for c in cells):
                result.add("notebook-no-narrative", "info", rel, "notebook has no markdown cells explaining it")
    return result


def strip_docstrings(tree: ast.Module) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def _code_signature(path: str, text: str | None) -> list[str] | None:
    """Comparable code content: docstring-free AST dumps (per code cell for notebooks)."""
    if text is None:
        return None
    if path.endswith(".py"):
        tree = parse(text)
        return [strip_docstrings(tree)] if tree else None
    cells = []
    for cell in notebook_cells(text):
        if cell.get("cell_type") == "code":
            tree = parse(strip_magics(cell_text(cell.get("source"))))
            cells.append(strip_docstrings(tree) if tree else cell_text(cell.get("source")))
    return cells


def verify_docs_only(ctx: Context, ref: str = "HEAD") -> CheckResult:
    """Prove that edits since ref changed only docstrings, comments and doc files."""
    result = CheckResult("docs-only")
    changed = set((git(ctx.root, "diff", "--name-only", "-z", ref) or "").split("\0"))
    changed |= set((git(ctx.root, "ls-files", "-o", "--exclude-standard", "-z") or "").split("\0"))
    changed.discard("")
    for rel in sorted(changed):
        if rel.endswith(DOC_EXTS):
            continue
        if not rel.endswith((".py", ".ipynb")):
            result.add("non-doc-file-changed", "block", rel, "a non-documentation file changed")
            continue
        before = git(ctx.root, "show", f"{ref}:{rel}")
        after = (ctx.root / rel).read_text(errors="replace") if (ctx.root / rel).exists() else None
        if before is None:
            result.add("new-code-file", "block", rel, "new code file; a docs pass must not add code")
        elif after is None:
            result.add("file-deleted", "block", rel, "file deleted during a docs pass")
        elif _code_signature(rel, before) != _code_signature(rel, after):
            result.add("behavior-change", "block", rel, "code changed beyond docstrings and comments")
    result.notes.append(f"checked {len(changed)} changed file(s) against {ref}")
    return result
