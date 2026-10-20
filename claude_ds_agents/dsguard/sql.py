"""SQL anti-patterns in .sql files, SQL strings inside Python, and %%sql notebook cells.

These rules are heuristics over text, not a parser: they flag patterns that are
almost always a bug or a cost problem and leave the semantics to the sql-reviewer
agent. sqlfluff adds dialect-aware linting when installed.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess

from .core import CheckResult, Context, call_name, cell_text, notebook_cells, python_sources

SQL_START = re.compile(r"^\s*(?:--[^\n]*\n\s*)*(select|with|insert|update|delete|merge|create|replace)\b", re.I)
SQL_BODY = re.compile(r"\b(from|into|table)\b", re.I)
KEYWORDS = {"on", "using", "where", "left", "right", "inner", "outer", "full", "join", "cross", "group",
            "order", "limit", "union", "qualify", "window", "having", "lateral"}
SARGABLE = re.compile(
    r"\b(date|lower|upper|cast|trim|substr|substring|year|month|to_char|to_date|date_trunc|"
    r"coalesce|extract|format_date|timestamp_trunc)\s*\([^()]*\)\s*(=|<|>|<=|>=|!=|<>|between\b|in\b|like\b)",
    re.I,
)
CLAUSE_END = re.compile(r"\b(group\s+by|order\s+by|having|limit|qualify|union|window)\b", re.I)


def looks_like_sql(text: str) -> bool:
    return bool(SQL_START.match(text)) and bool(SQL_BODY.search(text))


def mask(sql: str) -> tuple[str, str]:
    """(no_comments, clean): comments blanked; clean also blanks string literal contents.

    Blanking with spaces keeps offsets and newlines aligned with the original text.
    """
    no_comments, clean = list(sql), list(sql)
    i, n = 0, len(sql)
    while i < n:
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                no_comments[k] = clean[k] = " "
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            for k in range(i, j):
                if sql[k] != "\n":
                    no_comments[k] = clean[k] = " "
            i = j
        elif sql[i] == "'":
            j = i + 1
            while j < n and not (sql[j] == "'" and not sql.startswith("''", j)):
                j += 2 if sql.startswith("''", j) else 1
            for k in range(i + 1, min(j, n)):
                if sql[k] != "\n":
                    clean[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(no_comments), "".join(clean)


def statements(clean: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(";", clean):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(clean)))
    return [(s, e) for s, e in spans if clean[s:e].strip()]


def _where_clauses(stmt: str):
    for m in re.finditer(r"\bwhere\b", stmt, re.I):
        end = CLAUSE_END.search(stmt, m.end())
        # Stop at the enclosing paren closing, if any, for subquery WHERE clauses.
        depth, stop = 0, end.start() if end else len(stmt)
        for k in range(m.end(), stop):
            if stmt[k] == "(":
                depth += 1
            elif stmt[k] == ")":
                if depth == 0:
                    stop = k
                    break
                depth -= 1
        yield m.end(), stmt[m.end():stop]


def _paren_groups(stmt: str) -> list[tuple[int, int]]:
    stack, groups = [], []
    for k, ch in enumerate(stmt):
        if ch == "(":
            stack.append(k)
        elif ch == ")" and stack:
            groups.append((stack.pop(), k))
    return groups


def lint_statement(stmt: str, raw: str, cfg: dict) -> list[tuple[str, str, int, str]]:
    """Rules over one statement. stmt has strings blanked; raw keeps literals."""
    hits = []

    def hit(rule, severity, offset, message):
        hits.append((rule, severity, offset, message))

    for m in re.finditer(r"\bselect\s+(?:distinct\s+)?(?:\w+\.)?\*", stmt, re.I):
        hit("select-star", "warn", m.start(), "SELECT * scans every column and breaks when the schema changes")
    for m in re.finditer(r"\bselect\s+distinct\b", stmt, re.I):
        hit("select-distinct", "info", m.start(), "DISTINCT often hides join fan-out; check the grain instead")
    for m in re.finditer(r"\bfrom\s+[\w.\"`\[\]]+(?:\s+(?:as\s+)?(\w+))?\s*,\s*(?!unnest\b|lateral\b|table\s*\()", stmt, re.I):
        if (m.group(1) or "").lower() not in KEYWORDS:
            hit("implicit-cross-join", "warn", m.start(), "comma join: a missing join condition becomes a cross join")
    for m in re.finditer(r"\bnot\s+in\s*\(\s*select\b", stmt, re.I):
        hit("not-in-subquery", "warn", m.start(), "NOT IN (SELECT …) returns no rows if the subquery yields a NULL; use NOT EXISTS")
    for m in re.finditer(r"(?:=|!=|<>)\s*null\b", stmt, re.I):
        hit("compare-null", "warn", m.start(), "comparison with NULL is never true; use IS [NOT] NULL")
    for m in re.finditer(r"\bunion\b(?!\s+all\b)", stmt, re.I):
        hit("union-distinct", "info", m.start(), "UNION de-duplicates (sort/hash); use UNION ALL unless dedup is intended")
    for m in re.finditer(r"\blike\s+'%", raw, re.I):
        hit("leading-wildcard", "info", m.start(), "LIKE '%…' cannot use an index or clustering")
    for m in re.finditer(r"\bbetween\s+'\d{4}-\d{2}-\d{2}'\s+and\s+'\d{4}-\d{2}-\d{2}'", raw, re.I):
        hit("date-between", "info", m.start(),
            "BETWEEN dates on a TIMESTAMP column drops the end day after 00:00; use >= start AND < end + 1 day")

    wheres = list(_where_clauses(stmt))
    for offset, clause in wheres:
        for m in SARGABLE.finditer(clause):
            hit("function-on-filter-column", "info", offset + m.start(),
                "function on a filtered column blocks index use and partition pruning")

    for m in re.finditer(r"\bleft\s+(?:outer\s+)?join\s+([\w.\"`]+)(?:\s+(?:as\s+)?(\w+))?", stmt, re.I):
        alias = m.group(2) if m.group(2) and m.group(2).lower() not in KEYWORDS else m.group(1).split(".")[-1]
        alias = alias.strip('"`')
        pattern = re.compile(rf"\b{re.escape(alias)}\.\w+\s*(=|<>|!=|<=|>=|<|>|\bin\b|\blike\b|\bbetween\b)", re.I)
        for offset, clause in wheres:
            if offset > m.end():
                f = pattern.search(clause)
                if f:
                    hit("left-join-filtered", "warn", offset + f.start(),
                        f"WHERE filter on LEFT JOINed '{alias}' turns it into an INNER JOIN; move it to ON")
                    break

    for open_, close in _paren_groups(stmt):
        inner = stmt[open_ + 1:close]
        if not re.match(r"\s*select\b", inner, re.I):
            continue
        order = list(re.finditer(r"\border\s+by\b", inner, re.I))
        if order and not re.search(r"\b(limit|top|fetch|qualify)\b", inner[order[-1].end():], re.I):
            nested = [g for g in _paren_groups(inner) if g[0] < order[-1].start() < g[1]]
            if not nested:
                hit("order-in-subquery", "info", open_ + 1 + order[-1].start(),
                    "ORDER BY in a subquery/CTE without LIMIT is discarded work")

    top = re.sub(r"\([^()]*\)", lambda m: " " * len(m.group(0)), stmt)
    for _ in range(5):
        top = re.sub(r"\([^()]*\)", lambda m: " " * len(m.group(0)), top)
    limit = re.search(r"\blimit\b", top, re.I)
    if limit and re.match(r"\s*(select|with)\b", stmt, re.I) and not re.search(r"\border\s+by\b", top, re.I):
        hit("limit-without-order", "info", limit.start(),
            "LIMIT without ORDER BY returns arbitrary rows; never use it as a sample")

    for table, column in cfg["partitioned_tables"].items():
        if re.search(rf"\b{re.escape(table)}\b", stmt, re.I):
            filtered = any(re.search(rf"\b{re.escape(column)}\b", c, re.I) for _, c in wheres)
            if not filtered:
                hit("missing-partition-filter", "warn", 0,
                    f"reads partitioned table {table} without filtering on {column}: full scan")
    return hits


def lint_sql(sql: str, cfg: dict) -> list[tuple[str, str, int, str]]:
    """Return (rule, severity, line_offset_from_0, message) for every statement."""
    no_comments, clean = mask(sql)
    hits = []
    for start, end in statements(clean):
        for rule, severity, offset, message in lint_statement(clean[start:end], no_comments[start:end], cfg):
            hits.append((rule, severity, clean.count("\n", 0, start + offset), message))
    return hits


def python_sql(tree: ast.Module):
    """Yield (sql_text, lineno, interpolated) for SQL-looking strings in Python."""
    consumed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = [v.value if isinstance(v, ast.Constant) else "{?}" for v in node.values]
            consumed.update(id(v) for v in node.values)
            text = "".join(str(p) for p in parts)
            if looks_like_sql(text):
                yield text, node.lineno, True
        elif isinstance(node, ast.Call) and call_name(node.func) == "format" and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Constant) and isinstance(base.value, str) and looks_like_sql(base.value):
                consumed.add(id(base))
                yield base.value, base.lineno, True
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, str) and looks_like_sql(side.value):
                    consumed.add(id(side))
                    yield side.value, side.lineno, True
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in consumed:
            if looks_like_sql(node.value):
                yield node.value, node.lineno, False


def _sqlfluff(ctx: Context, files: list[str], result: CheckResult) -> None:
    exe = shutil.which("sqlfluff")
    if not exe:
        result.notes.append("sqlfluff not installed: dialect-aware lint skipped (builtin heuristics only)")
        return
    if not files:
        return
    cfg = ctx.config["sql"]
    proc = subprocess.run([exe, "lint", "--dialect", cfg["dialect"], "--format", "json", "--nofail",
                           *[str(ctx.root / f) for f in files]], capture_output=True, text=True, check=False)
    try:
        reports = json.loads(proc.stdout)
    except ValueError:
        result.notes.append(f"sqlfluff failed: {proc.stderr.strip()[:300]}")
        return
    skip = tuple(cfg["sqlfluff_exclude_prefixes"])
    for report in reports:
        rel = report["filepath"].removeprefix(str(ctx.root) + "/")
        for v in report.get("violations", []):
            code = v.get("code", "")
            if code.startswith(skip) or code == "PRS":
                continue
            result.add(f"sqlfluff:{code}", "info", rel, v.get("description", ""),
                       v.get("start_line_no") or v.get("line_no"))


def run(ctx: Context) -> CheckResult:
    result = CheckResult("sql")
    cfg = ctx.config["sql"]
    sql_files = [f for f in ctx.files if f.endswith(".sql")]
    for rel in sql_files:
        text = ctx.read(rel)
        for rule, severity, line, message in lint_sql(text or "", cfg):
            result.add(rule, severity, rel, message, line + 1)
    for src in python_sources(ctx):
        if not src.tree:
            continue
        for sql, lineno, interpolated in python_sql(src.tree):
            if interpolated:
                result.add("interpolated-sql", "warn", src.path,
                           "SQL built by string interpolation: injection risk and no plan reuse; use bind parameters",
                           lineno, src.cell)
            for rule, severity, line, message in lint_sql(sql, cfg):
                result.add(rule, severity, src.path, message, lineno + line, src.cell)
    for rel in ctx.files:
        if rel.endswith(".ipynb"):
            for index, cell in enumerate(notebook_cells(ctx.read(rel) or "")):
                source = cell_text(cell.get("source"))
                if cell.get("cell_type") == "code" and source.lstrip().startswith("%%sql"):
                    body = source.split("\n", 1)[1] if "\n" in source else ""
                    for rule, severity, line, message in lint_sql(body, cfg):
                        result.add(rule, severity, rel, message, line + 2, index)
    _sqlfluff(ctx, sql_files, result)
    return result
