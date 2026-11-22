"""Regressions from the second review, of the round 1 fixes (r01..r16, in that review's order)."""

import json
import re
import resource
import subprocess
import sys
from datetime import date
from types import SimpleNamespace

import pytest
from test_regressions import (DSGUARD, ENFORCE, TOOLKIT, _allowed, commit, enforced, executable, fake_key,  # noqa: F401
                              finding, global_hooks, ok, remote, repo, review, reviewed, run, user_config)

from dsguard import daybook
from dsguard import reviewloop as rl

# r01: ref names can hold Unicode whitespace and line separators


@pytest.mark.parametrize("name", ["feature v2", "z 0 x", "x y", "x\u0085y"])
def test_r01_unicode_ref_names_are_judged(enforced, remote, name):
    ok("git", "branch", name, "other", cwd=enforced)
    assert run("git", "push", remote, name, cwd=enforced).returncode != 0


def test_r01_odd_ref_does_not_release_the_others(enforced, remote):
    assert run("git", "push", remote, "other", "other:refs/tags/x y", cwd=enforced).returncode != 0


def test_r01_newline_in_local_refspec(enforced, remote):
    ok("git", "checkout", "-q", "other", cwd=enforced)
    (enforced / "a.txt").write_text("two lines\n")
    ok("git", "commit", "-qam", "first\nsecond", cwd=enforced)
    assert run("git", "push", remote, "other^{/first\nsecond}:refs/heads/prod", cwd=enforced).returncode != 0


@pytest.mark.parametrize("line", ["garbage", "a b c d", "refs/heads/x nothex refs/heads/x 0"])
def test_r01_unparseable_line_is_refused(enforced, line):
    assert not rl.check_push(enforced, line + "\n")[0]


# r02: a lookup failure refuses that ref


@pytest.mark.parametrize("name", ["review-1", "response-1", "/".join(["p" * 90] * 4)])
def test_r02_state_lookup_cannot_release_a_branch(enforced, remote, name):
    folder = enforced / ".dsguard" / "loop"
    (folder / "review-1.json").write_text('{"verdict": "approve"}')
    (folder / "response-1.json").write_text("[]")
    ok("git", "branch", name, "other", cwd=enforced)
    assert run("git", "push", remote, name, cwd=enforced).returncode != 0


def test_r02_corrupt_state_is_not_an_approval(enforced, remote):
    rl._state_path(enforced, "other").write_text("{bad")
    assert run("git", "push", remote, "other", cwd=enforced).returncode != 0


# r03: the refs reach the check and the repo's hook without a temp file


def test_r03_unwritable_tmp_does_not_empty_the_refs(enforced, remote):
    line = f"refs/heads/other {ok('git', 'rev-parse', 'other', cwd=enforced)} refs/heads/other {'0' * 40}\n"
    executable(enforced / ".git" / "hooks" / "pre-push", "grep -q refs/heads/other\n")
    no_files = lambda: resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # like a full /tmp
    proc = subprocess.run([str(TOOLKIT / "git-hooks" / "global" / "pre-push"), str(remote), str(remote)],
                          input=line, cwd=enforced, capture_output=True, text=True, preexec_fn=no_files)
    assert proc.returncode != 0 and "push blocked" in proc.stderr


# r04: exempt branch + its tag


@pytest.mark.parametrize("args", [["main", "v1"], ["--follow-tags", "main"]])
def test_r04_exempt_branch_with_its_tag(enforced, remote, user_config, args):
    user_config(ENFORCE + 'exempt_branches = ["main"]\n')
    ok("git", "checkout", "-q", "main", cwd=enforced)
    commit(enforced, "release\n")
    ok("git", "tag", "-a", "v1", "-m", "v1", cwd=enforced)
    options, names = [a for a in args if a.startswith("--")], [a for a in args if not a.startswith("--")]
    proc = run("git", "push", *options, remote, *names, cwd=enforced)
    assert proc.returncode == 0, proc.stderr


def test_r04_unrelated_tag_is_still_refused(enforced, remote, user_config):
    user_config(ENFORCE + 'exempt_branches = ["main"]\n')
    ok("git", "tag", "sneaky", "other", cwd=enforced)
    assert run("git", "push", remote, "main", "sneaky", cwd=enforced).returncode != 0


# r05: an existing override of company hooks is undone


def test_r05_reinstall_restores_company_hooks(isolated_home, tmp_path):
    install = [sys.executable, TOOLKIT / "install.py", "--git-hooks"]
    ok(*install, cwd=tmp_path)
    (isolated_home / "system.gitconfig").write_text("[core]\n\thooksPath = /opt/company/hooks\n")
    ok(*install, "--dry-run", cwd=tmp_path)
    assert "ds-toolkit" in ok("git", "config", "core.hooksPath", cwd="/")
    ok(*install, cwd=tmp_path)
    assert ok("git", "config", "core.hooksPath", cwd="/") == "/opt/company/hooks"


# r06: a new finding is never absorbed by a closed or deferred one


def test_r06_new_blocker_next_to_deferred(repo):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker", line=5)))
    rl.record_response(repo, [{"id": "R1-1", "action": "deferred", "reason": "human"}])
    decision = reviewed(repo, review("request_changes", finding("blocker", line=200)))
    found = rl.load_state(repo)["findings"]
    assert decision["blocking_open"] == ["R2-1"]
    assert found["R1-1"]["line"] == 5 and found["R1-1"]["status"] == "deferred"


@pytest.mark.parametrize("closing", [{"resolved_prior": ["R1-1"]},
                                     {"disputes": [{"prior_id": "R1-1", "ruling": "accept_dispute"}]}])
def test_r06_new_blocker_next_to_one_closed_this_round(repo, closing):
    rl.start(repo, "main")
    reviewed(repo, review("request_changes", finding("blocker", line=2)))
    rl.record_response(repo, [{"id": "R1-1", "action": "disputed", "reason": "r"}] if "disputes" in closing
                       else [{"id": "R1-1", "action": "fixed"}])
    decision = reviewed(repo, review("request_changes", finding("blocker", line=90), **closing))
    assert decision["blocking_open"] == ["R2-1"]
    assert rl.load_state(repo)["findings"]["R1-1"]["status"] in ("resolved", "dismissed")


# r07: the tests may not change what they test


@pytest.mark.parametrize("stage", ["", " && git add a.txt"])
def test_r07_tests_that_rewrite_the_tree_fail_the_gate(repo, user_config, stage):
    user_config(f"[review_loop]\ntest_command = 'echo good > a.txt{stage}'\n")
    rl.start(repo, "main")
    assert not rl.gate(repo)["passed"]


# r08: permissions, in every command and skill

GRANTS = sorted([*TOOLKIT.glob("commands/*.md"), *TOOLKIT.glob("skills/*/SKILL.md")])
DS = "~/.claude/ds-toolkit/bin/dsguard"


@pytest.mark.parametrize("path", GRANTS, ids=lambda p: f"{p.parent.name}/{p.name}")
@pytest.mark.parametrize("command", [
    "git -C . -c alias.x=!id x", "git fetch --upload-pack=id .", "git push --force", "git reset --hard",
    "python3 -c pass", "python -m pytest -q", "python3 -m pytest -q", "npm test", "make test", "gh pr merge 1",
    f"{DS} loop restart", f"{DS} loop gate", f"{DS} apply-cleanup plan.json",
])
def test_r08_permissions_everywhere(path, command):
    for rule in _allowed(path):
        prefix = rule.removesuffix(":*")
        matched = command == prefix or (rule.endswith(":*") and command.startswith(prefix + " "))
        assert not matched, f"{rule} pre-approves {command}"


# r09: NFS


def test_r09_registry_is_written_when_flock_fails(tmp_path, monkeypatch):
    def refuse(*_):
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(daybook, "fcntl", SimpleNamespace(flock=refuse, LOCK_EX=2))
    daybook.touch(tmp_path, {"home_dir": str(tmp_path)}, date.today())
    assert json.loads((tmp_path / "projects.json").read_text())


def test_r09_lock_is_taken_on_a_writable_file(tmp_path, monkeypatch):
    import fcntl

    modes = []
    record = lambda fd, _: modes.append(fcntl.fcntl(fd, fcntl.F_GETFL) & 3)
    monkeypatch.setattr(daybook, "fcntl", SimpleNamespace(flock=record, LOCK_EX=2))
    daybook.touch(tmp_path, {"home_dir": str(tmp_path)}, date.today())
    assert modes and modes[0] != 0  # 0 is O_RDONLY


# r10: branch and tag with the same name


def test_r10_branch_named_like_a_tag(repo, remote, user_config, global_hooks):
    ok("git", "tag", "feature", "main", cwd=repo)
    rl.start(repo, "main")
    reviewed(repo, review("approve"))
    user_config(ENFORCE)
    global_hooks()
    proc = run("git", "push", remote, "refs/heads/feature:refs/heads/feature", cwd=repo)
    assert proc.returncode == 0, proc.stderr


# r11: a review names the commit it reviewed


def test_r11_old_review_cannot_approve_a_newer_brief(repo):
    rl.start(repo, "main")
    rl.gate(repo)
    old = rl.review_brief(repo)["head"]
    commit(repo, "later\n")
    rl.gate(repo)
    rl.review_brief(repo)
    with pytest.raises(SystemExit, match="head"):
        rl.record_review(repo, review("approve", head=old))


def test_r11_review_without_a_head_is_refused(repo):
    rl.start(repo, "main")
    rl.gate(repo)
    rl.review_brief(repo)
    with pytest.raises(SystemExit, match="head"):
        rl.record_review(repo, review("approve"))


def test_r11_reviewer_is_told_to_return_the_head():
    assert re.search(r'^\s+"head":', (TOOLKIT / "agents" / "pr-reviewer.md").read_text(), re.M)


# r12: gate output and crashes


def test_r12_non_utf8_test_output(repo, user_config):
    user_config("[review_loop]\ntest_command = \"printf '\\\\351'\"\n")
    rl.start(repo, "main")
    assert rl.gate(repo)["passed"]


def test_r12_crashed_gate_revokes_the_previous_one(repo, monkeypatch):
    rl.start(repo, "main")
    rl.gate(repo)

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash")

    with monkeypatch.context() as patch:
        patch.setattr(daybook, "discover_files", crash)
        patch.setattr(rl, "load_config", lambda root: {"review_loop": {"test_command": ""}, "daybook": {}})
        with pytest.raises(RuntimeError):
            rl.gate(repo)
    with pytest.raises(SystemExit, match="gate"):
        rl.review_brief(repo)


# r13: colliding branch names


def test_r13_restart_leaves_the_other_branch_alone(repo):
    ok("git", "checkout", "-qb", "team_x", cwd=repo)
    rl.start(repo, "main")
    ok("git", "checkout", "-qb", "team/x", cwd=repo)
    rl.restart(repo, "main")
    ok("git", "checkout", "-q", "team_x", cwd=repo)
    assert rl.load_state(repo)["branch"] == "team_x"


# r14: a venv without pytest


def test_r14_venv_without_pytest_is_not_used(tmp_path):
    executable(tmp_path / ".venv" / "bin" / "python", "exit 0\n")
    assert ".venv" not in daybook.detect_tests(tmp_path, ["tests/test_x.py"], {"test_command": ""})


# r15: a crashed check says so


def test_r15_crashed_commit_check_warns(repo, user_config, global_hooks):
    user_config("secrets = 3\n")
    global_hooks()
    (repo / "k.py").write_text(fake_key())
    ok("git", "add", "k.py", cwd=repo)
    proc = run("git", "commit", "-qm", "k", cwd=repo)
    assert proc.returncode == 0 and "UNCHECKED" in proc.stderr


# r16: odd config values


@pytest.mark.parametrize("system,skipped", [
    (b'[includeIf "gitdir:/w/"]\n\tpath = ~nosuchuser/x\n', False),
    (b"[core]\n\thooksPath = /opt/caf\xe9\n", True),
    (b"[core]\n\thooksPath =\n", True),
])
def test_r16_installer_handles_odd_values(isolated_home, tmp_path, system, skipped):
    (isolated_home / "system.gitconfig").write_bytes(system)
    proc = run(sys.executable, TOOLKIT / "install.py", "--git-hooks", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert ("SKIP   git core.hooksPath" in proc.stdout) == skipped
