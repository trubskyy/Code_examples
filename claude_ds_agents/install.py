#!/usr/bin/env python3
"""Install the DS agent toolkit into ~/.claude (user level, so it works in every project).

  python3 install.py                 # agents, commands, skills, Claude hook, CLAUDE.md block
  python3 install.py --git-hooks     # also gate commits in every git repo on this machine
  python3 install.py --dry-run       # show what would change
  python3 install.py --uninstall     # reverse all of it

Everything is user-level, so it applies to every project you open, not one repository.
Symlinks mean `git pull` here updates every agent with no reinstall. settings.json and
CLAUDE.md are backed up before they change; existing hooks and your own CLAUDE.md text are kept.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLAUDE = Path.home() / ".claude"
TOOLKIT_LINK = CLAUDE / "ds-toolkit"
HOOK_COMMAND = "python3 ~/.claude/ds-toolkit/hooks/pretooluse_guard.py"
HOOK_MATCHER = "Write|Edit|MultiEdit|NotebookEdit|Bash"
GLOBAL_HOOKS = TOOLKIT_LINK / "git-hooks" / "global"
BLOCK_START, BLOCK_END = "<!-- ds-toolkit:start (managed by install.py) -->", "<!-- ds-toolkit:end -->"


def _backup(path: Path) -> None:
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")


def update_claude_md(install: bool, dry_run: bool) -> None:
    """Keep a managed block in ~/.claude/CLAUDE.md so Claude knows the toolkit in every project."""
    path = CLAUDE / "CLAUDE.md"
    text = path.read_text() if path.exists() else ""
    pattern = re.compile(re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END) + r"\n?", re.S)
    stripped = pattern.sub("", text)
    if install:
        block = f"{BLOCK_START}\n{(HERE / 'claude' / 'CLAUDE-block.md').read_text().strip()}\n{BLOCK_END}\n"
        new = (stripped.rstrip() + "\n\n" if stripped.strip() else "") + block
    else:
        new = stripped
    if new == text:
        print(f"ok     {path} (toolkit block {'current' if install else 'absent'})")
        return
    print(f"{'update' if install else 'remove'} toolkit block in {path}")
    if not dry_run:
        _backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new)


def _git_global(*args: str) -> str | None:
    proc = subprocess.run(["git", "config", "--global", *args], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def update_git_hooks(install: bool, dry_run: bool) -> None:
    """Point git's global core.hooksPath at the toolkit's dispatcher (opt-in), or remove it if ours."""
    current = _git_global("--get", "core.hooksPath")
    ours = current is not None and Path(current).expanduser() in (GLOBAL_HOOKS, GLOBAL_HOOKS.resolve())
    if install:
        if ours:
            print(f"ok     git core.hooksPath -> {current}")
        elif current:
            print(f"SKIP   git core.hooksPath is already set to {current}; not overriding it.\n"
                  f"       Per repo instead: ln -s ~/.claude/ds-toolkit/git-hooks/pre-commit .git/hooks/pre-commit")
        else:
            print(f"set    git config --global core.hooksPath {GLOBAL_HOOKS}  (every repo; each repo's own hooks still run)")
            if not dry_run:
                _git_global("core.hooksPath", str(GLOBAL_HOOKS))
    elif ours:
        print("unset  git config --global core.hooksPath")
        if not dry_run:
            _git_global("--unset", "core.hooksPath")


def links() -> list[tuple[Path, Path]]:
    pairs = [(TOOLKIT_LINK, HERE)]
    pairs += [(CLAUDE / "agents" / f.name, f) for f in sorted((HERE / "agents").glob("*.md"))]
    pairs += [(CLAUDE / "commands" / f.name, f) for f in sorted((HERE / "commands").glob("*.md"))]
    pairs += [(CLAUDE / "skills" / d.name, d) for d in sorted((HERE / "skills").iterdir()) if (d / "SKILL.md").is_file()]
    return pairs


def _hook_entries(settings: dict) -> list:
    return settings.setdefault("hooks", {}).setdefault("PreToolUse", [])


def _has_hook(entries: list) -> bool:
    return any(h.get("command") == HOOK_COMMAND for e in entries for h in e.get("hooks", []))


def update_settings(install: bool, dry_run: bool) -> None:
    path = CLAUDE / "settings.json"
    settings = json.loads(path.read_text()) if path.exists() else {}
    entries = _hook_entries(settings)
    if install and not _has_hook(entries):
        entries.append({"matcher": HOOK_MATCHER, "hooks": [{"type": "command", "command": HOOK_COMMAND}]})
        action = "register PreToolUse hook"
    elif not install and _has_hook(entries):
        for entry in entries:
            entry["hooks"] = [h for h in entry.get("hooks", []) if h.get("command") != HOOK_COMMAND]
        settings["hooks"]["PreToolUse"] = [e for e in entries if e["hooks"]]
        action = "remove PreToolUse hook"
    else:
        print(f"settings.json: hook already {'present' if install else 'absent'}")
        return
    print(f"settings.json: {action}")
    if dry_run:
        return
    _backup(path)
    path.write_text(json.dumps(settings, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--git-hooks", action="store_true",
                        help="gate commits in every git repo (sets global core.hooksPath; repos' own hooks still run)")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        print("Python 3.11+ is required (dsguard uses tomllib).")
        return 1

    for dest, src in links():
        if args.uninstall:
            if dest.is_symlink() and dest.resolve() == src:
                print(f"unlink {dest}")
                if not args.dry_run:
                    dest.unlink()
            continue
        if dest.is_symlink() and dest.resolve() == src:
            print(f"ok     {dest}")
            continue
        if dest.exists() or dest.is_symlink():
            print(f"SKIP   {dest} exists and is not ours; move it away and re-run")
            continue
        print(f"link   {dest} -> {src}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(src)

    update_settings(install=not args.uninstall, dry_run=args.dry_run)
    update_claude_md(install=not args.uninstall, dry_run=args.dry_run)
    if args.git_hooks or args.uninstall:
        update_git_hooks(install=not args.uninstall, dry_run=args.dry_run)
    if not args.uninstall:
        config = CLAUDE / "dsguard.toml"
        if config.exists():
            print(f"ok     {config} (kept; compare with dsguard.example.toml for new options)")
        else:
            print(f"copy   {config} <- dsguard.example.toml (your editable policy file)")
            if not args.dry_run:
                shutil.copy2(HERE / "dsguard.example.toml", config)
        if not args.git_hooks:
            print("\nTip: add --git-hooks to gate commits in every repo on this machine.")
        print("\nNext: start Claude Code in any project and run /ds-setup once to adopt the gateway and SQL Server settings.")
        print("Optional tools: pip install pip-audit bandit sqlfluff; install gitleaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
