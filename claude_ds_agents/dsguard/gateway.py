"""LLM gateway policy: no direct provider SDKs, endpoints or keys outside the gateway."""

from __future__ import annotations

import ast
import re
from fnmatch import fnmatch

from .core import CheckResult, Context, call_name, const_str, match_any, python_sources

URL_RE = re.compile(r"https?://([A-Za-z0-9.\-]+)")
IMPORT_LINE_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)
MANIFEST_RE = re.compile(
    r"(^|/)(requirements[^/]*\.txt|pyproject\.toml|setup\.cfg|environment[^/]*\.ya?ml|Pipfile)$"
)
DOC_EXTS = (".md", ".rst")
TEXT_EXTS = (
    ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".env", ".sh", ".txt", ".js", ".ts",
    ".sql", ".r", ".R", ".conf", "Dockerfile", "Makefile",
) + DOC_EXTS

# (rule, severity, line, message)
Hit = tuple[str, str, int | None, str]


def host_status(host: str, cfg: dict) -> str | None:
    host = host.lower()
    if any(fnmatch(host, pattern) for pattern in cfg["allowed_hosts"]):
        return "allowed"
    if any(fnmatch(host, pattern) for pattern in cfg["banned_hosts"]):
        return "banned"
    return None


def module_banned(module: str, cfg: dict) -> bool:
    return any(module == b or module.startswith(b + ".") for b in cfg["banned_imports"])


def _docstring_nodes(tree: ast.Module) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and const_str(body[0].value) is not None:
                ids.add(id(body[0].value))
    return ids


def scan_python_tree(tree: ast.Module, cfg: dict) -> list[Hit]:
    hits: list[Hit] = []
    ban_mode = cfg["mode"] == "ban"
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for module in modules:
            if ban_mode and module_banned(module, cfg):
                hits.append(("direct-llm-import", "block", node.lineno,
                             f"imports provider SDK '{module}'; route LLM calls through the gateway client"))
                break

        if isinstance(node, ast.Call) and not ban_mode and call_name(node.func) in cfg["client_constructors"]:
            hits.extend(_check_client_call(node, cfg))

        value = const_str(node) if isinstance(node, ast.Constant) else None
        if value:
            for host in URL_RE.findall(value):
                if host_status(host, cfg) == "banned":
                    severity = "warn" if id(node) in docstrings else "block"
                    hits.append(("direct-llm-endpoint", severity, node.lineno,
                                 f"references provider endpoint '{host}' instead of the gateway"))
            if value in cfg["provider_key_env_vars"]:
                hits.append(("provider-api-key", "warn", node.lineno,
                             f"reads provider credential {value}; the gateway should own credentials"))
    return hits


def _check_client_call(node: ast.Call, cfg: dict) -> list[Hit]:
    name = call_name(node.func)
    base = next((k for k in node.keywords if k.arg in cfg["base_url_kwargs"]), None)
    if base is None:
        return [("client-without-gateway", "warn", node.lineno,
                 f"{name}() has no base_url; it falls back to the provider or an env var that cannot be verified")]
    url = const_str(base.value)
    if url is None:
        return [("client-gateway-unverified", "info", node.lineno,
                 f"{name}() base_url is computed at runtime; confirm it resolves to the gateway")]
    hosts = URL_RE.findall(url)
    if not hosts or host_status(hosts[0], cfg) != "allowed":
        return [("client-not-gateway", "block", node.lineno,
                 f"{name}() base_url '{url}' is not an allowed gateway host")]
    return []


def scan_python_fragment(text: str, cfg: dict) -> list[Hit]:
    """Regex fallback for code that does not parse on its own (e.g. an Edit snippet)."""
    hits: list[Hit] = []
    if cfg["mode"] == "ban":
        for match in IMPORT_LINE_RE.finditer(text):
            module = match.group(1) or match.group(2)
            if module_banned(module, cfg):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(("direct-llm-import", "block", line,
                             f"imports provider SDK '{module}'; route LLM calls through the gateway client"))
    return hits + scan_text(text, cfg, is_doc=False)


def scan_text(text: str, cfg: dict, is_doc: bool) -> list[Hit]:
    hits: list[Hit] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for host in URL_RE.findall(line):
            if host_status(host, cfg) == "banned":
                hits.append(("direct-llm-endpoint", "warn" if is_doc else "block", lineno,
                             f"references provider endpoint '{host}' instead of the gateway"))
    return hits


def _manifest_hits(text: str, cfg: dict) -> list[Hit]:
    banned = {re.sub(r"[-_.]+", "-", p).lower() for p in cfg["banned_packages"]}
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", line.split("#")[0])[:1] + re.findall(
            r"""["']([A-Za-z0-9][A-Za-z0-9._-]*)""", line
        ):
            if re.sub(r"[-_.]+", "-", token).lower() in banned:
                hits.append(("provider-sdk-dependency", "warn", lineno,
                             f"depends on provider SDK '{token}'; is this pulled in by the gateway client?"))
                break
    return hits


def run(ctx: Context) -> CheckResult:
    result = CheckResult("gateway")
    cfg = ctx.config["llm_gateway"]
    if cfg["mode"] != "ban" and not cfg["allowed_hosts"]:
        result.notes.append("llm_gateway.allowed_hosts is empty; client base_url checks cannot pass")
    in_scope = [f for f in ctx.files if not match_any(f, cfg["exempt_paths"])]

    for src in python_sources(ctx, in_scope):
        hits = scan_python_tree(src.tree, cfg) if src.tree else scan_python_fragment(src.source, cfg)
        for rule, severity, line, message in hits:
            result.add(rule, severity, src.path, message, line, src.cell)

    for rel in in_scope:
        if rel.endswith((".py", ".ipynb")) or not rel.endswith(TEXT_EXTS):
            continue
        text = ctx.read(rel)
        if text is None:
            continue
        hits = scan_text(text, cfg, is_doc=rel.endswith(DOC_EXTS))
        if MANIFEST_RE.search(rel) and cfg["mode"] == "ban":
            hits += _manifest_hits(text, cfg)
        for rule, severity, line, message in hits:
            result.add(rule, severity, rel, message, line)
    return result
