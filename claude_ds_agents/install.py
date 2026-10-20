#!/usr/bin/env python3
"""Install the DS agent toolkit into ~/.claude (user level, so it works in every project).

  python3 install.py              # symlink agents/commands, register hooks
  python3 install.py --dry-run    # show what would change
  python3 install.py --uninstall

Symlinks mean `git pull` in this repo updates every agent with no reinstall.
settings.json is backed up before it is changed, and existing hooks are kept.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLAUDE = Path.home() / ".claude"
TOOLKIT_LINK = CLAUDE / "ds-toolkit"
HOOK_COMMAND = "python3 ~/.claude/ds-toolkit/hooks/pretooluse_guard.py"
HOOK_MATCHER = "Write|Edit|MultiEdit|NotebookEdit|Bash"


def links() -> list[tuple[Path, Path]]:
    pairs = [(TOOLKIT_LINK, HERE)]
    pairs += [(CLAUDE / "agents" / f.name, f) for f in sorted((HERE / "agents").glob("*.md"))]
    pairs += [(CLAUDE / "commands" / f.name, f) for f in sorted((HERE / "commands").glob("*.md"))]
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
    if path.exists():
        backup = path.with_name(f"settings.json.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")
    path.write_text(json.dumps(settings, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
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
    if not args.uninstall:
        print("\nNext: copy dsguard.example.toml to ~/.claude/dsguard.toml and set your gateway hosts.")
        print("Optional tools: pip install pip-audit bandit sqlfluff; brew install gitleaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
