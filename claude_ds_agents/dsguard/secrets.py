"""Secret detection: high-confidence token patterns, credential files, gitleaks if present."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import PurePosixPath

from .core import CheckResult, Context, match_any

PATTERNS = [
    ("private-key", "block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----")),
    ("aws-access-key", "block", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("anthropic-key", "block", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", "block", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("github-token", "block", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})")),
    ("slack-token", "block", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("google-api-key", "block", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("huggingface-token", "block", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("jwt", "warn", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
]
URL_CREDENTIALS = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@'\"]+:([^\s@'\"/]{3,})@[^\s'\"]+", re.I)
ASSIGNMENT = re.compile(
    r"""(?i)\b[a-z0-9_]*(?:api[_-]?key|secret|token|passwd|password|pwd)[a-z0-9_]*["']?\s*[:=]\s*["']([^"'\s]{12,})["']"""
)
PLACEHOLDERS = (
    "example", "placeholder", "changeme", "your", "xxx", "<", ">", "${", "{", "dummy", "fake",
    "redacted", "***", "...", "password", "secret", "token", "test",
)
SECRET_FILES = [
    ".env", ".env.*", "*.pem", "*.key", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "*.p12", "*.pfx",
    ".netrc", ".pgpass", "credentials.json", "*service-account*.json", "*serviceaccount*.json",
    ".htpasswd", "*.keystore", "*.jks", "kubeconfig",
]
SAFE_SUFFIXES = (".example", ".sample", ".template", ".dist", ".tmpl")


def entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(c / len(value) * math.log2(c / len(value)) for c in counts.values())


def redact(value: str) -> str:
    return f"{value[:4]}…({len(value)} chars)"


def _placeholder(value: str) -> bool:
    lower = value.lower()
    return any(p in lower for p in PLACEHOLDERS) or re.fullmatch(r"[A-Z0-9_]+", value) is not None


def scan_text(text: str) -> list[tuple[str, str, int, str]]:
    """Return (rule, severity, line, redacted match). Never returns the raw secret."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        taken: list[tuple[int, int]] = []
        for rule, severity, pattern in PATTERNS:
            for m in pattern.finditer(line):
                if any(s <= m.start() < e for s, e in taken):
                    continue
                taken.append(m.span())
                hits.append((rule, severity, lineno, redact(m.group(0))))
        for m in URL_CREDENTIALS.finditer(line):
            if not _placeholder(m.group(1)):
                hits.append(("url-credentials", "block", lineno, f"password {redact(m.group(1))} in connection URL"))
        for m in ASSIGNMENT.finditer(line):
            value = m.group(1)
            if any(s <= m.start(1) < e for s, e in taken) or _placeholder(value) or entropy(value) < 3.5:
                continue
            hits.append(("hardcoded-secret", "warn", lineno, redact(value)))
    return hits


def is_secret_file(rel: str) -> bool:
    name = PurePosixPath(rel).name
    if name.endswith(SAFE_SUFFIXES):
        return False
    return any(PurePosixPath(name).match(p) for p in SECRET_FILES)


def _gitleaks(ctx: Context, result: CheckResult) -> None:
    exe = shutil.which("gitleaks")
    if not exe:
        result.notes.append("gitleaks not installed: git history was not scanned (builtin patterns only)")
        return
    history = ctx.config["secrets"]["scan_history"] and ctx.tracked is not None
    with tempfile.TemporaryDirectory() as tmp:
        report = f"{tmp}/report.json"
        cmd = [exe, "detect", "--source", str(ctx.root), "--report-format", "json",
               "--report-path", report, "--redact", "--exit-code", "0", "--no-banner"]
        if not history:
            cmd.append("--no-git")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        try:
            with open(report) as fh:
                leaks = json.load(fh) or []
        except (OSError, ValueError):
            result.notes.append(f"gitleaks failed: {proc.stderr.strip()[:300]}")
            return
    for leak in leaks:
        commit = leak.get("Commit", "")
        where = f" (in commit {commit[:8]})" if commit else ""
        result.add(f"gitleaks:{leak.get('RuleID', 'secret')}", "block", leak.get("File", "?"),
                   f"secret detected{where}; rotate it, then purge history", leak.get("StartLine"))


def run(ctx: Context) -> CheckResult:
    result = CheckResult("secrets")
    allow = ctx.config["secrets"]["allow_paths"]
    for rel in ctx.files:
        if match_any(rel, allow):
            continue
        if is_secret_file(rel):
            tracked = ctx.tracked is not None and rel in ctx.tracked
            result.add("credential-file", "block" if tracked else "warn", rel,
                       "credential file is committed" if tracked else "credential file is not gitignored")
        text = ctx.read(rel)
        if text is None:
            continue
        for rule, severity, line, detail in scan_text(text):
            result.add(rule, severity, rel, f"possible secret {detail}", line)
    _gitleaks(ctx, result)
    return result
