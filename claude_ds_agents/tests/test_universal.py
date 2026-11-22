"""The toolkit applies to every project: machine-wide git hooks, the CLAUDE.md block, the
cross-project registry. Everything runs against temporary homes and git configs."""

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT))

from dsguard import daybook  # noqa: E402
from dsguard.core import DEFAULTS  # noqa: E402

DISPATCH_DIR = TOOLKIT / "git-hooks" / "global"


def run(cmd, cwd, env=None, check=True):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=check)


@pytest.fixture
def hooked_repo(tmp_path):
    """A repo whose hooks go through the toolkit dispatcher (repo-local hooksPath, not global)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = os.environ | {"DSGUARD_BIN": str(TOOLKIT / "bin" / "dsguard"), "HOME": str(tmp_path / "home"),
                        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["config", "core.hooksPath", str(DISPATCH_DIR)]):
        run(["git", *args], repo, env)
    return repo, env


def commit(repo, env, name, text, extra_env=None):
    (repo / name).write_text(text)
    run(["git", "add", name], repo, env)
    return run(["git", "commit", "-qm", name], repo, env | (extra_env or {}), check=False)


def fake_key(suffix):
    return 'key = "' + "AKIA" + suffix * 16 + '"\n'      # assembled so this file has no literal key


def test_dispatcher_blocks_secrets_and_chains_repo_hooks(hooked_repo):
    repo, env = hooked_repo
    hooks = repo / ".git" / "hooks"
    (hooks / "pre-commit").write_text("#!/bin/sh\necho own-pre-commit >> hook.log\n")
    (hooks / "post-commit").write_text("#!/bin/sh\necho own-post-commit >> hook.log\n")
    for h in ("pre-commit", "post-commit"):
        (hooks / h).chmod(0o755)

    blocked = commit(repo, env, "leak.py", fake_key("A"))
    assert blocked.returncode != 0 and "commit blocked" in blocked.stderr
    assert not (repo / "hook.log").exists()                        # repo hook never reached

    run(["git", "rm", "-q", "--cached", "leak.py"], repo, env)
    (repo / "leak.py").unlink()
    assert commit(repo, env, "ok.py", "x = 1\n").returncode == 0
    assert (repo / "hook.log").read_text().split() == ["own-pre-commit", "own-post-commit"]


def test_dispatcher_opt_outs_and_missing_toolkit(hooked_repo):
    repo, env = hooked_repo
    assert commit(repo, env, "a.py", fake_key("B"), {"DSGUARD_SKIP": "1"}).returncode == 0
    run(["git", "config", "ds.guard", "false"], repo, env)
    assert commit(repo, env, "b.py", fake_key("C")).returncode == 0
    run(["git", "config", "--unset", "ds.guard"], repo, env)
    assert commit(repo, env, "c.py", "y = 2\n", {"DSGUARD_BIN": "/nonexistent"}).returncode == 0


def test_every_client_hook_has_a_stub():
    stubs = {p.name for p in DISPATCH_DIR.iterdir() if p.name != "_dispatch"}
    assert {"pre-commit", "commit-msg", "pre-push", "post-checkout", "post-merge", "post-commit"} <= stubs
    assert all(os.access(DISPATCH_DIR / s, os.X_OK) for s in stubs | {"_dispatch"})


def _install(home, gitconfig, *args):
    env = os.environ | {"HOME": str(home), "GIT_CONFIG_GLOBAL": str(gitconfig)}
    return run([sys.executable, str(TOOLKIT / "install.py"), *args], TOOLKIT, env)


def _hooks_path(gitconfig):
    proc = subprocess.run(["git", "config", "--file", str(gitconfig), "--get", "core.hooksPath"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or None


def test_installer_claude_md_and_git_hooks_round_trip(tmp_path):
    home, gitconfig = tmp_path / "home", tmp_path / "gitconfig"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "CLAUDE.md").write_text("# mine\nkeep this\n")
    _install(home, gitconfig, "--git-hooks")
    claude_md = (home / ".claude" / "CLAUDE.md").read_text()
    assert claude_md.startswith("# mine\nkeep this\n") and claude_md.count("ds-toolkit:start") == 1
    assert "applies to every project" in claude_md
    assert _hooks_path(gitconfig).endswith("ds-toolkit/git-hooks/global")
    assert (home / ".claude" / "skills" / "ds-start-day" / "SKILL.md").exists()

    _install(home, gitconfig, "--git-hooks")                                  # idempotent
    assert (home / ".claude" / "CLAUDE.md").read_text() == claude_md

    _install(home, gitconfig, "--uninstall")
    assert (home / ".claude" / "CLAUDE.md").read_text().strip() == "# mine\nkeep this"
    assert _hooks_path(gitconfig) is None and not (home / ".claude" / "ds-toolkit").exists()


def test_installer_never_overrides_someone_elses_hooks_path(tmp_path):
    home, gitconfig = tmp_path / "home", tmp_path / "gitconfig"
    (home / ".claude").mkdir(parents=True)
    gitconfig.write_text("[core]\n\thooksPath = /opt/company/hooks\n")
    out = _install(home, gitconfig, "--git-hooks").stdout
    assert "SKIP" in out and _hooks_path(gitconfig) == "/opt/company/hooks"
    _install(home, gitconfig, "--uninstall")
    assert _hooks_path(gitconfig) == "/opt/company/hooks"


def test_registry_records_checks_from_any_project(tmp_path):
    cfg = DEFAULTS["daybook"] | {"home_dir": str(tmp_path / "jhome")}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    daybook.touch(a, cfg, date(2026, 10, 22), dsguard={"verdict": "BLOCKED", "date": "2026-10-22"})
    daybook.touch(b, cfg, date(2026, 10, 23))
    daybook.touch(a, cfg, date(2026, 10, 23), mode="home", journal=str(tmp_path / "jhome" / "a-1"))
    registry = json.loads((tmp_path / "jhome" / "projects.json").read_text())
    assert registry[str(a.resolve())]["dsguard"]["verdict"] == "BLOCKED"       # earlier fields kept
    assert registry[str(a.resolve())]["mode"] == "home"
    overview = daybook.projects(cfg, date(2026, 10, 23))
    assert {p["name"] for p in overview} == {"a", "b"}
    assert next(p for p in overview if p["name"] == "b")["mode"] is None       # touched, no journal yet
    assert "no journal" in daybook.render_projects(overview)
