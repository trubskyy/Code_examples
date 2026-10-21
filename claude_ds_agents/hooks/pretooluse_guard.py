#!/usr/bin/env python3
"""Claude Code PreToolUse hook: stop gateway bypasses and secrets before they are written.

Wired to Write|Edit|MultiEdit|NotebookEdit|Bash. Reads the tool call on stdin and
answers with a permission decision:
  deny  - new content imports a provider SDK / hits a provider endpoint (ban mode),
          or contains a high-confidence secret; a Bash command calls a provider endpoint
  ask   - a recursive rm (so the cleanup agent goes through dsguard apply-cleanup)
Anything else, including an internal error, passes silently: a broken hook must
not stop work, and CI/pre-commit run the same rules as the backstop.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsguard import gateway, secrets  # noqa: E402
from dsguard.core import find_root, load_config, match_any, parse  # noqa: E402

RECURSIVE_RM = re.compile(r"(^|[;&|]\s*|\s)(sudo\s+)?rm\s+(-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)\b")


def decide(payload: dict) -> tuple[str, str] | None:
    tool = payload.get("tool_name", "")
    data = payload.get("tool_input", {}) or {}
    root = find_root(Path(payload.get("cwd") or "."))
    config = load_config(root)
    gw = config["llm_gateway"]

    if tool == "Bash":
        command = data.get("command", "")
        hosts = [h for h in gateway.URL_RE.findall(command) if gateway.host_status(h, gw) == "banned"]
        if hosts:
            return "deny", f"dsguard: '{hosts[0]}' is a direct LLM provider endpoint; go through the LLM gateway."
        if config["hooks"]["ask_on_recursive_rm"] and RECURSIVE_RM.search(command):
            return "ask", "dsguard: recursive delete. Cleanup should go through `dsguard apply-cleanup` (reversible)."
        return None

    path = data.get("file_path") or data.get("notebook_path") or ""
    if tool == "Write":
        content = data.get("content", "")
    elif tool == "Edit":
        content = data.get("new_string", "")
    elif tool == "MultiEdit":
        content = "\n".join(e.get("new_string", "") for e in data.get("edits", []))
    elif tool == "NotebookEdit":
        content = data.get("new_source", "")
    else:
        return None
    try:
        rel = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        rel = path

    problems = []
    if not match_any(rel, gw["exempt_paths"]):
        if rel.endswith((".py", ".ipynb")) or tool == "NotebookEdit":
            tree = parse(content)
            hits = gateway.scan_python_tree(tree, gw) if tree else gateway.scan_python_fragment(content, gw)
        else:
            hits = gateway.scan_text(content, gw, is_doc=rel.endswith(gateway.DOC_EXTS))
        problems += [f"line {line}: {msg}" for _, sev, line, msg in hits if sev == "block"]
    if not match_any(rel, config["secrets"]["allow_paths"]):
        problems += [f"line {line}: {rule} {detail}" for rule, sev, line, detail in secrets.scan_text(content)
                     if sev == "block"]
        if secrets.is_secret_file(rel):
            problems.append("this is a credential file; keep it out of the project and load secrets from the environment")
    if problems:
        return "deny", "dsguard blocked this edit:\n- " + "\n- ".join(problems[:10])
    return None


def main() -> int:
    try:
        decision = decide(json.load(sys.stdin))
    except Exception as exc:  # fail open; pre-commit and CI re-check
        print(f"dsguard hook error (allowed): {exc}", file=sys.stderr)
        return 0
    if decision:
        verdict, reason = decision
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": verdict, "permissionDecisionReason": reason,
        }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
