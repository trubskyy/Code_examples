"""Adoption helper: read an existing (prewritten) LLM gateway client and draft dsguard config.

  dsguard suggest-gateway path/to/gateway_client      # a package dir or a .py file

It reports what the gateway code itself does: which provider SDKs it wraps, which hosts
it targets, which env vars it reads, and whether projects consume it as an importable
package. From that it proposes [llm_gateway] settings. The user (or the setup command)
reviews the TOML before it goes into ~/.claude/dsguard.toml.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .core import DEFAULTS, call_name, const_str, parse
from .gateway import URL_RE, host_status, module_banned

ENV_CALLS = {"getenv", "get", "environ"}


def _py_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.py") if not any(part.startswith(".") or part == "__pycache__" for part in p.parts))


def _package_name(target: Path) -> str | None:
    """Import name if target is (inside) a package; None for loose scripts."""
    directory = target if target.is_dir() else target.parent
    name = None
    while (directory / "__init__.py").exists():
        name = directory.name
        directory = directory.parent
    if name is None and target.is_dir() and (target / "pyproject.toml").exists():
        src = target / "src"
        pkgs = [p.name for p in (src if src.is_dir() else target).iterdir() if (p / "__init__.py").exists()]
        name = pkgs[0] if len(pkgs) == 1 else None
    return name


def analyse(target: Path) -> dict:
    gw = DEFAULTS["llm_gateway"]
    sdks, hosts, banned_hosts, env_vars, base_url_kwargs = set(), set(), set(), set(), set()
    for path in _py_files(target):
        tree = parse(path.read_text(errors="replace"))
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                sdks.update(a.name.split(".")[0] for a in node.names if module_banned(a.name, gw))
            elif isinstance(node, ast.ImportFrom) and node.module and module_banned(node.module, gw):
                sdks.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if call_name(node.func) in ENV_CALLS and node.args and const_str(node.args[0]):
                    env_vars.add(const_str(node.args[0]))
                for kw in node.keywords:
                    if kw.arg in gw["base_url_kwargs"]:
                        base_url_kwargs.add(kw.arg)
            elif isinstance(node, ast.Subscript) and const_str(node.slice) and "environ" in ast.unparse(node.value):
                env_vars.add(const_str(node.slice))
            value = const_str(node) if isinstance(node, ast.Constant) else None
            if value:
                for host in URL_RE.findall(value):
                    (banned_hosts if host_status(host, gw) == "banned" else hosts).add(host.lower())
    package = _package_name(target)
    return {"target": str(target), "package": package, "provider_sdks": sorted(sdks), "hosts": sorted(hosts),
            "provider_hosts": sorted(banned_hosts), "env_vars": sorted(env_vars),
            "uses_base_url": sorted(base_url_kwargs), "files": len(_py_files(target))}


def suggest(target: Path, root: Path) -> str:
    a = analyse(target)
    try:
        rel = target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = None
    mode = "ban" if a["package"] else "require_gateway_base_url"
    lines = [
        "# Drafted by `dsguard suggest-gateway` from the gateway code. Review before saving to ~/.claude/dsguard.toml.",
        f"# Gateway code: {a['target']} ({a['files']} Python file(s))",
        f"# Wraps provider SDKs: {', '.join(a['provider_sdks']) or 'none found'}",
        f"# Hosts referenced: {', '.join(a['hosts']) or 'none found (maybe from env/config)'}",
        f"# Env vars read: {', '.join(a['env_vars']) or 'none'}",
        "",
        "[llm_gateway]",
    ]
    if a["package"]:
        lines += [f"# Projects import the gateway as `{a['package']}`, so provider SDKs are banned everywhere else.",
                  f'mode = "{mode}"']
    else:
        lines += ["# No importable package: projects use the provider SDK pointed at the gateway via base_url.",
                  f'mode = "{mode}"']
    hosts = [h for h in a["hosts"] if h not in ("localhost", "127.0.0.1", "example.com")]
    lines.append(f"allowed_hosts = {hosts!r}".replace("'", '"') + ("" if hosts else "   # TODO: add the gateway hostname"))
    exempt = [f"{rel}/**" if target.is_dir() else rel] if rel else []
    lines.append(f"exempt_paths = {exempt!r}".replace("'", '"')
                 + ("" if exempt else "   # gateway lives outside this repo (installed package): nothing to exempt"))
    gateway_keys = [v for v in a["env_vars"] if v in DEFAULTS["llm_gateway"]["provider_key_env_vars"]]
    if gateway_keys:
        remaining = [v for v in DEFAULTS["llm_gateway"]["provider_key_env_vars"] if v not in gateway_keys]
        lines += [f"# The gateway itself reads {', '.join(gateway_keys)}; stop flagging it in projects:",
                  f"provider_key_env_vars = {remaining!r}".replace("'", '"')]
    if a["provider_hosts"]:
        lines.append(f"# Note: the gateway code calls provider hosts directly: {', '.join(a['provider_hosts'])} (expected inside the gateway)")
    return "\n".join(lines) + "\n"
