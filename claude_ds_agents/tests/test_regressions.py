"""Regressions from the review of 21e6acf: one group of tests per blocker (b01..b14), plus the
permission rules and the broken-config warning. These use real git, real hooks and the real CLI."""

import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT))

from dsguard import daybook  # noqa: E402
from dsguard import reviewloop as rl  # noqa: E402

DSGUARD = TOOLKIT / "bin" / "dsguard"
HOOKS = TOOLKIT / "git-hooks" / "global"


def run(*args, cwd, stdin=None, **env):
    return subprocess.run([str(a) for a in args], cwd=cwd, input=stdin, env=os.environ | env,
                          capture_output=True, text=True, errors="replace")


def ok(*args, cwd):
    proc = run(*args, cwd=cwd)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def commit(root, text="change\n", name="a.txt"):
    (root / name).write_text(text)
    ok("git", "add", name, cwd=root)
    ok("git", "commit", "-qm", "change", cwd=root)
    return ok("git", "rev-parse", "HEAD", cwd=root)


def fake_key():
    return 'key = "' + "AKIA" + "Q" * 16 + '"\n'


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@pytest.fixture
def user_config(isolated_home):
    def write(text):
        (isolated_home / ".claude" / "dsguard.toml").write_text(text)
    write('[review_loop]\ntest_command = "true"\n')
    return write


@pytest.fixture
def repo(tmp_path, user_config):
    root = tmp_path / "repo"
    root.mkdir()
    ok("git", "init", "-q", "-b", "main", cwd=root)
    commit(root, "base\n")
    ok("git", "checkout", "-qb", "feature", cwd=root)
    commit(root, "feature\n")
    return root


@pytest.fixture
def global_hooks():
    def enable():
        ok("git", "config", "--global", "core.hooksPath", HOOKS, cwd="/")
        os.environ["DSGUARD_BIN"] = str(DSGUARD)
    yield enable
    os.environ.pop("DSGUARD_BIN", None)


@pytest.fixture
def remote(tmp_path):
    path = tmp_path / "remote.git"
    ok("git", "init", "-q", "--bare", path, cwd=tmp_path)
    return path


def finding(severity, title="Missing input validation", line=2, **extra):
    return {"severity": severity, "category": "correctness", "file": "a.txt", "line": line, "title": title,
            "evidence": "e", "fix": "f", "failure_scenario": "s"} | extra


def review(verdict, *findings, **extra):
    return {"verdict": verdict, "findings": list(findings)} | extra


def reviewed(root, data):
    rl.gate(root)
    data = {"head": rl.review_brief(root)["head"]} | data
    return rl.record_review(root, data)


def approve(root):
    rl.start(root, "main")
    assert reviewed(root, review("approve"))["action"] == "DONE"


# b01: only a real finding blocks

@pytest.mark.parametrize("broken", ["broken = [", "secrets = 3"])
def test_b01_broken_config_does_not_block_commit(repo, user_config, global_hooks, broken):
    global_hooks()
    user_config(broken)
    (repo / "a.txt").write_text("clean\n")
    proc = run("git", "commit", "-qam", "clean", cwd=repo)
    assert proc.returncode == 0, proc.stderr


def test_b01_broken_config_does_not_block_push(repo, remote, user_config, global_hooks):
    global_hooks()
    user_config("broken = [")
    proc = run("git", "push", remote, "feature", cwd=repo)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("code", [1, 2, 127])
def test_b01_crashing_launcher_never_blocks(repo, tmp_path, global_hooks, code):
    global_hooks()
    executable(tmp_path / "broken-dsguard", f"exit {code}\n")
    (repo / "a.txt").write_text("clean\n")
    proc = run("git", "commit", "-qam", "clean", cwd=repo, DSGUARD_BIN=str(tmp_path / "broken-dsguard"))
    assert proc.returncode == 0, proc.stderr


def test_b01_per_repo_hook_does_not_block_on_crash(repo, isolated_home, user_config):
    (isolated_home / ".claude" / "ds-toolkit").symlink_to(TOOLKIT)
    user_config("broken = [")
    assert run(TOOLKIT / "git-hooks" / "pre-commit", cwd=repo).returncode == 0


def test_b01_per_repo_hook_still_blocks_findings(repo, isolated_home):
    (isolated_home / ".claude" / "ds-toolkit").symlink_to(TOOLKIT)
    (repo / "leak.py").write_text(fake_key())
    ok("git", "add", "leak.py", cwd=repo)
    assert run(TOOLKIT / "git-hooks" / "pre-commit", cwd=repo).returncode != 0


@pytest.mark.parametrize("name", [b"caf\xe9.txt", b"caf\xe9.py"])
def test_b01_odd_filenames_neither_block_nor_disable_the_gate(repo, global_hooks, name):
    global_hooks()
    with open(os.path.join(os.fsencode(repo), name), "w") as fh:
        fh.write("x = 1\n")
    (repo / "a.txt").write_text("clean\n")
    assert run("git", "commit", "-qam", "clean", cwd=repo).returncode == 0
    with open(os.path.join(os.fsencode(repo), name), "w") as fh:
        fh.write(fake_key())
    ok("git", "add", "-A", cwd=repo)
    assert run("git", "commit", "-qm", "leak", cwd=repo).returncode != 0


@pytest.mark.parametrize("content", ["[]", "{bad"])
def test_b01_bad_registry_does_not_affect_the_gate(repo, isolated_home, global_hooks, content):
    global_hooks()
    (isolated_home / ".claude" / "journal").mkdir()
    (isolated_home / ".claude" / "journal" / "projects.json").write_text(content)
    (repo / "a.txt").write_text("clean\n")
    assert run("git", "commit", "-qam", "clean", cwd=repo).returncode == 0
    (repo / "leak.py").write_text(fake_key())
    ok("git", "add", "leak.py", cwd=repo)
    assert run("git", "commit", "-qm", "leak", cwd=repo).returncode != 0


# b02: company hooksPath is never overridden

@pytest.mark.parametrize("where", ["system", "system_include", "include", "include_if", "nested_include_if"])
def test_b02_installer_respects_foreign_hooks_path(isolated_home, tmp_path, where):
    company = tmp_path / "company.gitconfig"
    company.write_text("[core]\n\thooksPath = /opt/company/hooks\n")
    system, user = isolated_home / "system.gitconfig", isolated_home / ".gitconfig"
    if where == "system":
        system.write_text(company.read_text())
    elif where == "system_include":
        system.write_text(f"[include]\n\tpath = {company}\n")
    elif where == "include":
        user.write_text(user.read_text() + f"[include]\n\tpath = {company}\n")
    else:
        target = company
        if where == "nested_include_if":
            (isolated_home / "work.gitconfig").write_text(f"[include]\n\tpath = {company}\n")
            target = "work.gitconfig"                                   # relative to the including file
        user.write_text(user.read_text() + f'[includeIf "gitdir:/work/"]\n\tpath = {target}\n')
    proc = run(sys.executable, TOOLKIT / "install.py", "--git-hooks", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "SKIP   git core.hooksPath" in proc.stdout
    assert "ds-toolkit" not in user.read_text()


# b03: push-to-checkout keeps git's behavior

def _push_to_checkout(tmp_path, dirty=None, own_hook=None, unborn=False):
    target, clone = tmp_path / "target", tmp_path / "clone"
    target.mkdir()
    ok("git", "init", "-q", "-b", "main", cwd=target)
    if not unborn:
        commit(target, "old\n")
    ok("git", "config", "receive.denyCurrentBranch", "updateInstead", cwd=target)
    ok("git", "clone", "-q", target, clone, cwd=tmp_path)
    ok("git", "checkout", "-q", "-B", "main", cwd=clone)
    commit(clone, "new\n")
    if own_hook:
        executable(target / ".git" / "hooks" / "push-to-checkout", own_hook)
    if dirty:
        (target / "a.txt").write_text("local edit\n")
        if dirty == "staged":
            ok("git", "add", "a.txt", cwd=target)
    return target, run("git", "push", "origin", "main", cwd=clone)


@pytest.mark.parametrize("unborn", [False, True])
def test_b03_updates_the_checked_out_tree(tmp_path, global_hooks, unborn):
    global_hooks()
    target, proc = _push_to_checkout(tmp_path, unborn=unborn)
    assert proc.returncode == 0, proc.stderr
    assert (target / "a.txt").read_text() == "new\n"
    assert ok("git", "status", "--porcelain", cwd=target) == ""


@pytest.mark.parametrize("dirty", ["unstaged", "staged"])
def test_b03_refuses_a_dirty_tree(tmp_path, global_hooks, dirty):
    global_hooks()
    target, proc = _push_to_checkout(tmp_path, dirty=dirty)
    assert proc.returncode != 0
    assert (target / "a.txt").read_text() == "local edit\n"


def test_b03_linked_worktree_is_updated(tmp_path, global_hooks):
    main, linked, clone = tmp_path / "main", tmp_path / "linked", tmp_path / "clone"
    main.mkdir()
    ok("git", "init", "-q", "-b", "main", cwd=main)
    commit(main, "old\n")
    ok("git", "config", "receive.denyCurrentBranch", "updateInstead", cwd=main)
    ok("git", "worktree", "add", "-q", "-b", "side", linked, cwd=main)
    ok("git", "clone", "-q", "-b", "side", main, clone, cwd=tmp_path)
    commit(clone, "new\n")
    global_hooks()
    ok("git", "push", "-q", "origin", "side", cwd=clone)
    assert (linked / "a.txt").read_text() == "new\n" and (main / "a.txt").read_text() == "old\n"


def test_b03_repos_own_hook_still_decides(tmp_path, global_hooks):
    global_hooks()
    target, proc = _push_to_checkout(tmp_path, own_hook="exit 1\n")
    assert proc.returncode != 0 and (target / "a.txt").read_text() == "old\n"


# b04: every hook git looks up in hooksPath is forwarded

ALL_HOOKS = """applypatch-msg pre-applypatch post-applypatch pre-commit pre-merge-commit prepare-commit-msg
commit-msg post-commit pre-rebase post-checkout post-merge pre-push pre-receive update proc-receive
post-receive post-update reference-transaction push-to-checkout pre-auto-gc post-rewrite sendemail-validate
p4-changelist p4-prepare-changelist p4-post-changelist p4-pre-submit post-index-change""".split()


@pytest.mark.parametrize("hook", ALL_HOOKS)
def test_b04_every_hook_has_a_stub(hook):
    assert os.access(HOOKS / hook, os.X_OK)


@pytest.mark.parametrize("hook", ["pre-receive", "update"])
def test_b04_server_hooks_can_still_reject(repo, remote, global_hooks, hook):
    executable(remote / "hooks" / hook, "exit 1\n")
    global_hooks()
    assert run("git", "push", remote, "feature", cwd=repo).returncode != 0


@pytest.mark.parametrize("hook", ["post-receive", "post-update"])
def test_b04_server_post_hooks_still_run(repo, remote, tmp_path, global_hooks, hook):
    executable(remote / "hooks" / hook, f'touch "{tmp_path}/ran"\n')
    global_hooks()
    ok("git", "push", "-q", remote, "feature", cwd=repo)
    assert (tmp_path / "ran").exists()


def test_b04_proc_receive_fails_closed_without_a_repo_hook(repo, remote, global_hooks):
    ok("git", "config", "receive.procReceiveRefs", "refs/for", cwd=remote)
    global_hooks()
    assert run("git", "push", remote, "HEAD:refs/for/main", cwd=repo).returncode != 0


# b05: no approval without a green gate on exactly this commit

def test_b05_red_gate_cannot_be_approved(repo, user_config):
    user_config('[review_loop]\ntest_command = "false"\n')
    rl.start(repo, "main")
    with pytest.raises(SystemExit, match="gate"):
        reviewed(repo, review("approve"))


def test_b05_missing_gate_cannot_be_approved(repo):
    rl.start(repo, "main")
    with pytest.raises(SystemExit, match="gate|brief"):
        rl.record_review(repo, review("approve"))


def test_b05_stale_gate_cannot_be_briefed(repo):
    rl.start(repo, "main")
    rl.gate(repo)
    commit(repo, "later\n")
    with pytest.raises(SystemExit, match="gate"):
        rl.review_brief(repo)


def test_b05_gate_failing_after_the_brief_revokes_it(repo, user_config):
    rl.start(repo, "main")
    rl.gate(repo)
    rl.review_brief(repo)
    user_config('[review_loop]\ntest_command = "false"\n')
    rl.gate(repo)
    with pytest.raises(SystemExit, match="gate"):
        rl.record_review(repo, review("approve"))


@pytest.mark.parametrize("name", ["a.txt", "untracked.py"])
def test_b05_gate_refuses_a_tree_that_is_not_head(repo, name):
    rl.start(repo, "main")
    (repo / name).write_text("not committed\n")
    assert not rl.gate(repo)["passed"]


def test_b05_gate_fails_when_a_check_crashes(repo, user_config):
    user_config('[review_loop]\ntest_command = "true"\n[secrets]\nallow_paths = 5\n')
    commit(repo, "x = 1\n", "code.py")
    rl.start(repo, "main")
    assert not rl.gate(repo)["passed"]


def test_b05_gate_cannot_pass_by_dirtying_head_midway(repo, user_config):
    user_config('[review_loop]\ntest_command = "git commit -q --allow-empty -m sneaky"\n')
    rl.start(repo, "main")
    rl.gate(repo)
    with pytest.raises(SystemExit, match="gate"):
        rl.review_brief(repo)


# b06: approve what was reviewed

def test_b06_commit_during_review_is_not_approved(repo):
    rl.start(repo, "main")
    rl.gate(repo)
    rl.review_brief(repo)
    commit(repo, "late\n")
    with pytest.raises(SystemExit, match="HEAD"):
        rl.record_review(repo, review("approve"))
    assert rl.load_state(repo)["approved_head"] is None


def test_b06_new_commit_with_its_own_gate_still_needs_its_own_brief(repo):
    rl.start(repo, "main")
    rl.gate(repo)
    rl.review_brief(repo)
    commit(repo, "late\n")
    rl.gate(repo)
    with pytest.raises(SystemExit, match="HEAD"):
        rl.record_review(repo, review("approve"))


def test_b06_brief_pins_commit_ids(repo):
    rl.start(repo, "main")
    rl.gate(repo)
    assert rl.review_brief(repo)["base"] == ok("git", "rev-parse", "main", cwd=repo)


# b07: pre-push judges what is being pushed

ENFORCE = '[review_loop]\ntest_command = "true"\nenforce_on_push = true\n'


@pytest.fixture
def enforced(repo, user_config, global_hooks):
    approve(repo)
    user_config(ENFORCE)
    ok("git", "checkout", "-qb", "other", "main", cwd=repo)
    commit(repo, "unreviewed\n")
    ok("git", "checkout", "-q", "feature", cwd=repo)
    global_hooks()
    return repo


@pytest.mark.parametrize("spec", [["other"], ["feature", "other"], ["--all"], ["other:feature"],
                                  ["other:refs/tags/sneaky"], ["other:refs/custom/sneaky"]])
def test_b07_unreviewed_refs_are_blocked(enforced, remote, spec):
    assert run("git", "push", remote, *spec, cwd=enforced).returncode != 0


@pytest.mark.parametrize("spec", ["feature", "HEAD:refs/heads/feature"])
def test_b07_approved_ref_is_allowed(enforced, remote, spec):
    proc = run("git", "push", remote, spec, cwd=enforced)
    assert proc.returncode == 0, proc.stderr


def test_b07_approved_ref_is_allowed_from_another_branch(enforced, remote):
    ok("git", "checkout", "-q", "other", cwd=enforced)
    proc = run("git", "push", remote, "feature", cwd=enforced)
    assert proc.returncode == 0, proc.stderr


def test_b07_ref_with_spaces_cannot_crash_the_check(enforced, remote):
    proc = run("git", "push", remote, "other@{0 seconds ago}:refs/heads/production", cwd=enforced)
    assert proc.returncode != 0, proc.stderr


def test_b07_delete_is_allowed(enforced, remote):
    ok("git", "push", "-q", "--no-verify", remote, "other", cwd=enforced)
    ok("git", "checkout", "-q", "other", cwd=enforced)
    proc = run("git", "push", remote, ":other", cwd=enforced)
    assert proc.returncode == 0, proc.stderr


def test_b07_tag_on_approved_commit_is_allowed(enforced, remote):
    ok("git", "tag", "-a", "v1", "-m", "v1", "feature", cwd=enforced)
    proc = run("git", "push", remote, "v1", cwd=enforced)
    assert proc.returncode == 0, proc.stderr


def test_b07_exemption_is_for_the_destination(enforced, remote, user_config):
    user_config(ENFORCE + 'exempt_branches = ["other"]\n')
    assert run("git", "push", remote, "other:production", cwd=enforced).returncode != 0
    assert run("git", "push", remote, "other", cwd=enforced).returncode == 0


def test_b07_similarly_named_branch_does_not_share_approval(enforced, remote):
    ok("git", "branch", "-m", "feature", "team_x", cwd=enforced)
    state = rl.load_state(enforced, "feature")
    state["branch"] = "team_x"
    rl.save_state(enforced, state)
    ok("git", "checkout", "-qb", "team/x", cwd=enforced)
    assert run("git", "push", remote, "team/x", cwd=enforced).returncode != 0


def test_b07_repo_hook_still_gets_the_refs(enforced, remote, tmp_path):
    executable(enforced / ".git" / "hooks" / "pre-push", f'cat > "{tmp_path}/refs"\n')
    ok("git", "push", "-q", remote, "feature", cwd=enforced)
    assert "refs/heads/feature" in (tmp_path / "refs").read_text()


# b08: the documented loop works with enforcement on

def test_b08_command_pushes_only_after_done():
    text = (TOOLKIT / "commands" / "ds-review-loop.md").read_text()
    assert "git push" not in text.split("## 5.")[0]


def test_b08_two_rounds_then_push(repo, remote, user_config, global_hooks):
    user_config(ENFORCE)
    global_hooks()
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker")))
    commit(repo, "fixed\n")
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}])
    assert reviewed(repo, review("approve", resolved_prior=["R1-1"]))["action"] == "DONE"
    assert run("git", "push", remote, "feature", cwd=repo).returncode == 0


# b09: distinct findings stay distinct

@pytest.mark.parametrize("order", [("blocker", "minor"), ("minor", "blocker")])
def test_b09_minor_cannot_overwrite_blocker_with_same_title(repo, order):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("minor")))
    rl.record_response(repo, [])
    decision = reviewed(repo, review("approve", *[finding(s, line=i) for i, s in enumerate(order, 2)]))
    assert decision["action"] != "DONE"
    assert len(rl.load_state(repo)["findings"]) == 2


def test_b09_fingerprint_match_cannot_lower_severity(repo):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker")))
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}])
    assert reviewed(repo, review("approve", finding("minor")))["action"] != "DONE"


def test_b09_linked_finding_can_lower_severity(repo):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker")))
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}])
    assert reviewed(repo, review("approve", finding("minor", prior_id="R1-1")))["action"] == "DONE"


def test_b09_each_same_titled_blocker_must_be_accounted_for(repo):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker", line=2), finding("blocker", line=90)))
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}, {"id": "R1-2", "action": "fixed"}])
    with pytest.raises(SystemExit, match="missing"):
        reviewed(repo, review("request_changes", finding("blocker", line=90)))


def test_b09_duplicate_prior_id_is_rejected(repo):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker")))
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}])
    with pytest.raises(SystemExit, match="prior_id"):
        reviewed(repo, review("approve", finding("blocker", prior_id="R1-1"), finding("minor", prior_id="R1-1")))


@pytest.mark.parametrize("a,b", [
    ("Ошибка в загрузке данных", "Утечка памяти сервера"),
    ("alpha bravo charlie delta echo foxtrot golf hotel zulu", "alpha bravo charlie delta echo foxtrot golf hotel yankee"),
    ("🔥", "💥"),
])
def test_b09_fingerprint_uses_the_whole_title(a, b):
    assert rl.fingerprint(finding("minor", a)) != rl.fingerprint(finding("minor", b))


# b10: deferred means a person decides

@pytest.mark.parametrize("second", [
    {"findings": [finding("minor")]},
    {"findings": [finding("minor", prior_id="R1-1")]},
    {"resolved_prior": ["R1-1"]},
    {"disputes": [{"prior_id": "R1-1", "ruling": "accept_dispute"}]},
])
def test_b10_deferred_always_escalates(repo, second):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker")))
    rl.record_response(repo, [{"id": "R1-1", "action": "deferred", "reason": "product decision"}])
    rl.gate(repo)
    assert "R1-1" in rl.review_brief(repo)["prior_findings"]
    assert reviewed(repo, {"verdict": "approve", "findings": []} | second)["action"] == "ESCALATE"


# b11: escalation survives a plain start

def test_b11_start_refuses_to_replace_an_escalation(repo):
    rl.start(repo, "main", max_rounds=1)
    assert reviewed(repo, review("request_changes", finding("blocker")))["action"] == "ESCALATE"
    with pytest.raises(SystemExit, match="escalated"):
        rl.start(repo, "main")
    assert run(DSGUARD, "loop", "start", "--base", "main", "--restart", cwd=repo).returncode != 0
    assert rl.load_state(repo)["findings"]


def test_b11_explicit_restart_archives(repo):
    rl.start(repo, "main", max_rounds=1)
    reviewed(repo, review("request_changes", finding("blocker")))
    proc = run(DSGUARD, "loop", "restart", "--base", "main", cwd=repo)
    assert proc.returncode == 0, proc.stderr
    assert rl.load_state(repo)["findings"] == {}
    assert len(list((repo / ".dsguard" / "loop").glob("feature-*.*.json"))) == 1


def test_b11_done_state_is_archived(repo):
    approve(repo)
    rl.start(repo, "main")
    assert len(list((repo / ".dsguard" / "loop").glob("feature-*.*.json"))) == 1


# b12: an interpreter that exists

def test_b12_python3_only_machine(tmp_path, monkeypatch):
    executable(tmp_path / "bin" / "python3", "exit 0\n")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert daybook.detect_tests(tmp_path, ["tests/test_x.py"], {"test_command": ""}) == "python3 -m pytest -q"


def test_b12_active_python_is_preferred(tmp_path, monkeypatch):
    for name in ("python", "python3"):
        executable(tmp_path / "bin" / name, "exit 0\n")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert daybook.detect_tests(tmp_path, ["tests/test_x.py"], {"test_command": ""}) == "python -m pytest -q"


@pytest.mark.parametrize("venv", [".venv", "venv"])
def test_b12_project_venv_wins(tmp_path, venv):
    executable(tmp_path / venv / "bin" / "python", "exit 0\n")
    (tmp_path / venv / "lib" / "python3.12" / "site-packages" / "pytest").mkdir(parents=True)
    assert daybook.detect_tests(tmp_path, ["tests/test_x.py"], {"test_command": ""}) == f"{venv}/bin/python -m pytest -q"


def test_b12_gate_runs_without_python_on_path(repo, user_config, tmp_path, monkeypatch):
    user_config("")
    commit(repo, "def test_x():\n    pass\n", "test_x.py")
    for tool in ("git", "sh"):
        executable(tmp_path / "bin" / tool, f'exec /usr/bin/{tool} "$@"\n')
    executable(tmp_path / "bin" / "python3", "exit 0\n")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    rl.start(repo, "main")
    result = rl.gate(repo)
    assert result["passed"], result


# b13: a date means from midnight

def test_b13_morning_commit_counts_for_today(repo, monkeypatch):
    today = date.today().isoformat()
    monkeypatch.setenv("GIT_COMMITTER_DATE", f"{today}T00:00:01")
    commit(repo, "x = 1\n", "morning.py")
    assert any(c["date"] for c in daybook.commits_since(repo, today) if c["subject"] == "change"
               and c["hash"] == ok("git", "rev-parse", "--short", "HEAD", cwd=repo))
    assert "morning.py" in daybook.changed_since(repo, today)


# b14: the registry survives parallel writers

def test_b14_parallel_touches_lose_nothing(tmp_path):
    home = tmp_path / "journal"
    home.mkdir()
    (home / "projects.json").write_text(json.dumps({f"seed{i}": {} for i in range(200)}))
    code = (f"import sys; sys.path.insert(0, {str(TOOLKIT)!r})\n"
            "from dsguard import daybook\nfrom pathlib import Path\nfrom datetime import date\n"
            f"for i in range(40): daybook.touch(Path(sys.argv[1] + str(i)), {{'home_dir': {str(home)!r}}}, date.today())\n")
    procs = [subprocess.Popen([sys.executable, "-c", code, f"/project{k}-"]) for k in range(6)]
    assert all(p.wait() == 0 for p in procs)
    assert len(json.loads((home / "projects.json").read_text())) == 200 + 6 * 40


@pytest.mark.parametrize("content", ["[]", "{bad", ""])
def test_b14_unreadable_registry_is_never_overwritten(tmp_path, content):
    (tmp_path / "projects.json").write_text(content)
    daybook.touch(tmp_path, {"home_dir": str(tmp_path)}, date.today())
    assert (tmp_path / "projects.json").read_text() == content


def test_b14_symlinked_registry_stays_a_link(tmp_path):
    (tmp_path / "real.json").write_text("{}")
    (tmp_path / "projects.json").symlink_to(tmp_path / "real.json")
    daybook.touch(tmp_path, {"home_dir": str(tmp_path)}, date.today())
    assert (tmp_path / "projects.json").is_symlink() and json.loads((tmp_path / "real.json").read_text())


# permissions: nothing pre-approves arbitrary git, shell, pushes or a loop restart

def _allowed(path):
    line = re.search(r"^allowed-tools: (.*)$", path.read_text(), re.M).group(1)
    return re.findall(r"Bash\(([^)]*)\)", line)


GRANTING = ["commands/ds-review-loop.md", "commands/ds-setup.md",
            "skills/ds-start-day/SKILL.md", "skills/ds-end-day/SKILL.md"]
DS = "~/.claude/ds-toolkit/bin/dsguard"


@pytest.mark.parametrize("path", GRANTING)
@pytest.mark.parametrize("command", [
    "git -C . -c alias.x=!id x", "git -c alias.x=!id x", "git push --force origin main", "git reset --hard HEAD~3",
    "git config --global alias.x !id", "gh pr merge 1", "python3 -c pass", "python -m pytest --rootdir=/ -p evil",
    "make test SHELL=/tmp/evil", "Rscript -e system('id')", f"{DS} loop restart --base main",
])
def test_permissions_do_not_preapprove(path, command):
    for rule in _allowed(TOOLKIT / path):
        prefix = rule.removesuffix(":*")
        matched = command == prefix or (rule.endswith(":*") and command.startswith(prefix + " "))
        assert not matched, f"{rule} pre-approves {command}"


# the edit hook says so when it is not guarding

@pytest.mark.parametrize("broken", ["broken = [", "llm_gateway = 3"])
def test_hook_warns_visibly_when_config_is_broken(repo, user_config, broken):
    user_config(broken)
    payload = {"tool_name": "Write", "cwd": str(repo), "tool_input": {"file_path": str(repo / "a.py"), "content": "x=1"}}
    proc = run(sys.executable, TOOLKIT / "hooks" / "pretooluse_guard.py", cwd=repo, stdin=json.dumps(payload))
    assert proc.returncode == 0
    message = json.loads(proc.stdout)["systemMessage"]
    assert "NOT" in message and "dsguard" in message


def test_hook_warns_visibly_when_it_cannot_load(tmp_path):
    copy = tmp_path / "hooks" / "guard.py"
    copy.parent.mkdir()
    copy.write_text((TOOLKIT / "hooks" / "pretooluse_guard.py").read_text())     # no dsguard package beside it
    proc = run(sys.executable, "-I", copy, cwd=tmp_path, stdin="{}")
    assert proc.returncode == 0 and "NOT" in json.loads(proc.stdout)["systemMessage"]
