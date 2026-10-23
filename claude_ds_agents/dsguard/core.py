"""Shared plumbing for dsguard checks: config, file discovery, sources, findings."""

from __future__ import annotations

import ast
import copy
import json
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path

USER_CONFIG = Path.home() / ".claude" / "dsguard.toml"
STATE_DIR = ".dsguard"
MAX_SCAN_BYTES = 2_000_000
SEVERITY_ORDER = {"block": 0, "warn": 1, "info": 2}

# Only used when the project is not a git repo; with git, .gitignore decides.
WALK_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".ipynb_checkpoints", ".tox", "build", "dist",
    STATE_DIR,
}

DEFAULTS: dict = {
    "llm_gateway": {
        # "ban": provider SDKs may not be imported at all outside exempt_paths.
        # "require_gateway_base_url": provider SDKs are allowed, but every client
        # must be constructed with base_url pointing at an allowed gateway host.
        "mode": "ban",
        "allowed_hosts": [],
        "banned_hosts": [
            "api.openai.com", "*.openai.azure.com", "api.anthropic.com",
            "generativelanguage.googleapis.com", "*aiplatform.googleapis.com",
            "bedrock-runtime.*.amazonaws.com", "api.cohere.ai", "api.cohere.com",
            "api.mistral.ai", "api.groq.com", "api.together.xyz", "api.together.ai",
            "api.fireworks.ai", "api.deepseek.com", "api.perplexity.ai", "openrouter.ai",
            "api-inference.huggingface.co", "router.huggingface.co", "api.x.ai",
        ],
        "banned_imports": [
            "openai", "anthropic", "google.generativeai", "google.genai", "vertexai",
            "cohere", "mistralai", "groq", "together", "litellm", "replicate",
            "langchain_openai", "langchain_anthropic", "langchain_google_genai",
            "langchain_google_vertexai", "langchain_mistralai", "langchain_cohere",
            "llama_index.llms.openai", "llama_index.llms.anthropic",
        ],
        "banned_packages": [
            "openai", "anthropic", "google-generativeai", "google-genai", "cohere",
            "mistralai", "groq", "together", "litellm", "replicate", "langchain-openai",
            "langchain-anthropic", "langchain-google-genai", "langchain-mistralai",
        ],
        "client_constructors": [
            "OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI", "Anthropic",
            "AsyncAnthropic", "AnthropicBedrock", "AnthropicVertex", "ChatOpenAI",
            "AzureChatOpenAI", "ChatAnthropic", "OpenAIEmbeddings", "AzureOpenAIEmbeddings",
        ],
        "base_url_kwargs": [
            "base_url", "api_base", "openai_api_base", "azure_endpoint", "anthropic_api_url",
        ],
        "provider_key_env_vars": [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
            "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY",
        ],
        "exempt_paths": [],
    },
    "secrets": {"scan_history": True, "allow_paths": []},
    "exposure": {
        "data_extensions": [
            ".csv", ".tsv", ".parquet", ".feather", ".arrow", ".xlsx", ".xls", ".jsonl",
            ".avro", ".orc", ".sav", ".dta", ".sas7bdat", ".db", ".sqlite", ".duckdb",
        ],
        "model_extensions": [
            ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".safetensors",
            ".onnx", ".h5", ".keras", ".bin", ".gguf", ".npy", ".npz",
        ],
        "allowed_data_paths": ["**/fixtures/**", "**/testdata/**", "tests/data/**"],
        "pii_email_allow_domains": [
            "example.com", "example.org", "example.net", "test.com", "localhost",
            "users.noreply.github.com",
        ],
    },
    "licenses": {
        "python": "",
        # Families that fail the check: permissive, weak-copyleft, unknown, strong-copyleft,
        # network-copyleft, restricted. Empty until there is a company policy.
        "block_families": [],
        "exceptions": {},
    },
    "sql": {
        "dialect": "tsql",
        "partitioned_tables": {},
        "sqlfluff_exclude_prefixes": ["LT", "CP"],
    },
    "code": {
        "max_function_lines": 60,
        "max_complexity": 10,
        "max_nesting": 4,
        "max_args": 7,
        "min_duplicate_lines": 6,
        "ruff": True,
        "ruff_extend_select": ["B", "PERF", "SIM", "PD", "NPY", "C4", "RET", "PLW", "PLE"],
    },
    "docs": {"require_module_docstring": True, "min_public_coverage": 0.8},
    "cleanup": {"large_file_mb": 10, "entrypoints": [], "keep": []},
    "hooks": {"ask_on_recursive_rm": True},
    "daybook": {
        "dir": "docs/journal",
        "workdays": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "weekly_review_day": "Fri",
        "push": False,
        "code_extensions": [".py", ".ipynb", ".sql", ".R", ".r", ".sh", ".js", ".ts"],
        "todo_markers": ["TODO", "FIXME", "XXX", "HACK"],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(root: Path) -> dict:
    """Defaults < ~/.claude/dsguard.toml < [tool.dsguard] in pyproject < ./dsguard.toml."""
    config = copy.deepcopy(DEFAULTS)
    if USER_CONFIG.is_file():
        _deep_merge(config, tomllib.loads(USER_CONFIG.read_text()))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            tool = tomllib.loads(pyproject.read_text()).get("tool", {})
        except tomllib.TOMLDecodeError:
            tool = {}
        _deep_merge(config, tool.get("dsguard", {}))
    local = root / "dsguard.toml"
    if local.is_file():
        _deep_merge(config, tomllib.loads(local.read_text()))
    return config


def match_any(rel: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch(rel, pattern) or (pattern.startswith("**/") and fnmatch(rel, pattern[3:])):
            return True
    return False


def git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def find_root(start: Path) -> Path:
    top = git(start, "rev-parse", "--show-toplevel")
    return Path(top.strip()) if top else start.resolve()


def _split_z(out: str) -> list[str]:
    return [p for p in out.split("\0") if p]


def discover_files(root: Path) -> list[str]:
    """Tracked plus untracked-but-not-ignored files, as posix paths relative to root."""
    out = git(root, "ls-files", "-co", "--exclude-standard", "-z")
    if out is not None:
        files = [p for p in _split_z(out) if (root / p).is_file()]
    else:
        files = []
        for path in root.rglob("*"):
            rel = path.relative_to(root)
            if path.is_file() and not any(part in WALK_SKIP_DIRS for part in rel.parts):
                files.append(rel.as_posix())
    return sorted(p for p in files if not p.startswith(STATE_DIR + "/"))


def tracked_files(root: Path) -> set[str] | None:
    out = git(root, "ls-files", "-z")
    return None if out is None else set(_split_z(out))


def changed_files(root: Path, ref: str) -> set[str]:
    """Files committed since ref, plus staged, unstaged and untracked changes."""
    committed = git(root, "diff", "--name-only", "-z", f"{ref}...HEAD")
    if committed is None:
        raise SystemExit(f"dsguard: cannot diff against {ref!r}")
    changed = set(_split_z(committed))
    for args in (("diff", "--name-only", "-z"), ("diff", "--name-only", "-z", "--cached"),
                 ("ls-files", "-o", "--exclude-standard", "-z")):
        changed.update(_split_z(git(root, *args) or ""))
    return changed


def staged_files(root: Path) -> set[str]:
    return set(_split_z(git(root, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR") or ""))


def ensure_state_dir(root: Path) -> Path:
    state = root / STATE_DIR
    state.mkdir(exist_ok=True)
    ignore = state / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")
    return state


@dataclass
class Finding:
    rule: str
    severity: str
    path: str
    message: str
    line: int | None = None
    cell: int | None = None

    @property
    def location(self) -> str:
        loc = self.path
        if self.cell is not None:
            loc += f"[cell {self.cell}]"
        if self.line is not None:
            loc += f":{self.line}"
        return loc


@dataclass
class CheckResult:
    name: str
    status: str = "ok"  # ok | findings | skipped | error
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, rule, severity, path, message, line=None, cell=None) -> None:
        self.findings.append(Finding(rule, severity, path, message, line, cell))
        self.status = "findings"

    def skip(self, note: str) -> CheckResult:
        self.status = "skipped"
        self.notes.append(note)
        return self

    def to_dict(self) -> dict:
        data = asdict(self)
        data["findings"] = sorted(
            data["findings"], key=lambda f: (SEVERITY_ORDER[f["severity"]], f["path"], f["line"] or 0)
        )
        return data


@dataclass
class Context:
    root: Path
    config: dict
    files: list[str]       # in scope for this run
    all_files: list[str]   # whole project, for cross-file analysis
    tracked: set[str] | None
    _text: dict = field(default_factory=dict, repr=False)

    def size(self, rel: str) -> int:
        try:
            return (self.root / rel).stat().st_size
        except OSError:
            return 0

    def read(self, rel: str) -> str | None:
        """File text, or None for binary or oversized files."""
        if rel not in self._text:
            text = None
            path = self.root / rel
            if self.size(rel) <= MAX_SCAN_BYTES:
                try:
                    raw = path.read_bytes()
                except OSError:
                    raw = b"\0"
                if b"\0" not in raw[:8192]:
                    text = raw.decode("utf-8", errors="replace")
            self._text[rel] = text
        return self._text[rel]


def build_context(
    root: Path, changed_since: str | None = None, staged: bool = False, paths: list[str] | None = None
) -> Context:
    root = root.resolve()
    all_files = discover_files(root)
    files = all_files
    if changed_since:
        scope = changed_files(root, changed_since)
        files = [f for f in all_files if f in scope]
    elif staged:
        scope = staged_files(root)
        files = [f for f in all_files if f in scope]
    if paths:
        prefixes = [p.rstrip("/") for p in paths]
        files = [f for f in files if any(f == p or f.startswith(p + "/") for p in prefixes)]
    return Context(root, load_config(root), files, all_files, tracked_files(root))


@dataclass
class PySource:
    path: str
    cell: int | None
    source: str
    tree: ast.Module | None


_MAGIC = re.compile(r"^(\s*)[%!]")


def notebook_cells(text: str) -> list[dict]:
    try:
        cells = json.loads(text).get("cells", [])
    except (ValueError, AttributeError):
        return []
    return cells if isinstance(cells, list) else []


def cell_text(value) -> str:
    return "".join(value) if isinstance(value, list) else (value or "")


def strip_magics(source: str) -> str:
    """Blank IPython magics and shell escapes, keeping line numbers stable."""
    if source.lstrip().startswith("%%"):
        return ""
    return "\n".join("" if _MAGIC.match(line) else line for line in source.split("\n"))


def parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def python_sources(ctx: Context, files: list[str] | None = None):
    """Yield every .py file and every notebook code cell as a parsed PySource."""
    for rel in ctx.files if files is None else files:
        if rel.endswith(".py"):
            text = ctx.read(rel)
            if text is not None:
                yield PySource(rel, None, text, parse(text))
        elif rel.endswith(".ipynb"):
            text = ctx.read(rel)
            for index, cell in enumerate(notebook_cells(text or "")):
                if cell.get("cell_type") == "code":
                    source = strip_magics(cell_text(cell.get("source")))
                    yield PySource(rel, index, source, parse(source))


def call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def dotted_name(node: ast.expr) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def kwarg(call: ast.Call, name: str) -> ast.keyword | None:
    return next((k for k in call.keywords if k.arg == name), None)
