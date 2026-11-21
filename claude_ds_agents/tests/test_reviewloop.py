"""The review-loop referee: convergence, escalation rules, input validation, push enforcement."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT))

from dsguard import reviewloop as rl  # noqa: E402


def git(root, *args, env=None):
    return subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True, env=env)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("def f(x):\n    return x\n")
    (root / "dsguard.toml").write_text(f'[review_loop]\ntest_command = "true"\n[daybook]\nhome_dir = "{tmp_path}/jhome"\n')
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    git(root, "checkout", "-qb", "feature")
    (root / "app.py").write_text("def f(x):\n    return x * 2\n")
    git(root, "commit", "-qam", "change")
    return root


def finding(title, severity="major", file="app.py", **extra):
    base = {"severity": severity, "category": "correctness", "file": file, "line": 2, "title": title,
            "evidence": "returns x * 2", "fix": "handle None"}
    if severity in ("blocker", "major"):
        base["failure_scenario"] = "f(None) raises TypeError"
    return base | extra


def review(verdict, findings=(), resolved=(), disputes=()):
    return {"verdict": verdict, "summary": "s", "findings": list(findings),
            "resolved_prior": list(resolved), "disputes": list(disputes)}


def commit(root, text):
    (root / "app.py").write_text(text)
    git(root, "commit", "-qam", "fix")


def test_converges_to_done_and_approves_head(repo):
    rl.start(repo, "main")
    d1 = rl.record_review(repo, review("request_changes", [finding("None input crashes f")]))
    assert d1["action"] == "CONTINUE" and d1["blocking_open"] == ["R1-1"]
    with pytest.raises(SystemExit, match="author has not answered"):
        rl.record_review(repo, review("approve"))
    commit(repo, "def f(x):\n    return 0 if x is None else x * 2\n")
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed", "note": "guard"}])
    d2 = rl.record_review(repo, review("approve", resolved=["R1-1"]))
    assert d2["action"] == "DONE"
    state = rl.load_state(repo)
    assert state["status"] == "done" and state["approved_head"] == rl._head(repo)
    assert "DONE" in rl.report(state) and "R1-1" in rl.report(state)


def test_minor_findings_do_not_block(repo):
    rl.start(repo, "main")
    d = rl.record_review(repo, review("approve", [finding("Rename x to value", severity="nit")]))
    assert d["action"] == "DONE"


def test_review_validation(repo):
    rl.start(repo, "main")
    with pytest.raises(SystemExit, match="verdict"):
        rl.record_review(repo, {"verdict": "lgtm", "findings": []})
    no_scenario = finding("x")
    del no_scenario["failure_scenario"]
    with pytest.raises(SystemExit, match="failure_scenario"):
        rl.record_review(repo, review("request_changes", [no_scenario]))
    with pytest.raises(SystemExit, match="missing evidence"):
        rl.record_review(repo, review("request_changes", [finding("x") | {"evidence": ""}]))


def test_reviewer_must_account_for_every_open_finding_and_dispute(repo):
    rl.start(repo, "main")
    rl.record_review(repo, review("request_changes", [finding("None input crashes f"), finding("Overflow on big ints")]))
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"},
                              {"id": "R1-2", "action": "disputed", "reason": "Python ints do not overflow"}])
    with pytest.raises(SystemExit, match="missing: R1-2"):
        rl.record_review(repo, review("approve", resolved=["R1-1"]))
    d = rl.record_review(repo, review("approve", resolved=["R1-1"],
                                      disputes=[{"prior_id": "R1-2", "ruling": "accept_dispute", "reason": "right"}]))
    assert d["action"] == "DONE"
    assert rl.load_state(repo)["findings"]["R1-2"]["status"] == "dismissed"


def test_author_must_answer_every_blocking_finding(repo):
    rl.start(repo, "main")
    rl.record_review(repo, review("request_changes", [finding("A broken"), finding("B broken thing")]))
    with pytest.raises(SystemExit, match="missing: R1-2"):
        rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}])
    with pytest.raises(SystemExit, match="needs a reason"):
        rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}, {"id": "R1-2", "action": "disputed"}])


def test_no_progress_escalates(repo):
    rl.start(repo, "main")
    rl.record_review(repo, review("request_changes", [finding("None input crashes f")]))
    rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}])
    # Reviewer re-reports the same problem without linking it: the fingerprint still matches.
    d = rl.record_review(repo, review("request_changes", [finding("f crashes on None input", line=7)]))
    assert d["action"] == "ESCALATE" and "no progress" in d["reason"]
    with pytest.raises(SystemExit, match="escalated"):
        rl.record_review(repo, review("approve"))


def test_ping_pong_and_rejected_disputes_escalate(repo):
    rl.start(repo, "main", max_rounds=10)
    rl.record_review(repo, review("request_changes", [finding("None input crashes f"), finding("Another bug here")]))
    for _ in range(2):  # fixed, then reported again: reopened twice
        rl.record_response(repo, [{"id": "R1-1", "action": "fixed"}, {"id": "R1-2", "action": "fixed"}])
        d = rl.record_review(repo, review("request_changes", resolved=["R1-1", "R1-2"]))
        rl.record_response(repo, [])
        d = rl.record_review(repo, review("request_changes", [finding("x", prior_id="R1-1")]))
        if d["action"] == "ESCALATE":
            break
    assert d["action"] == "ESCALATE" and "ping-pong" in d["reason"]


def test_dispute_rejected_twice_escalates(repo):
    rl.start(repo, "main", max_rounds=10)
    rl.record_review(repo, review("request_changes", [finding("None input crashes f"), finding("Second issue")]))
    reject = [{"prior_id": "R1-1", "ruling": "reject_dispute", "reason": "callers pass None"}]
    rl.record_response(repo, [{"id": "R1-1", "action": "disputed", "reason": "never None"},
                              {"id": "R1-2", "action": "fixed"}])
    d = rl.record_review(repo, review("request_changes", [finding("None input crashes f", prior_id="R1-1")],
                                      resolved=["R1-2"], disputes=reject))
    assert d["action"] == "CONTINUE"
    rl.record_response(repo, [{"id": "R1-1", "action": "disputed", "reason": "still never None"}])
    d = rl.record_review(repo, review("request_changes", [finding("None input crashes f", prior_id="R1-1")],
                                      disputes=reject))
    assert d["action"] == "ESCALATE" and "disagree" in d["reason"]


def test_round_limit_and_deferred_escalate(repo):
    rl.start(repo, "main", max_rounds=1)
    d = rl.record_review(repo, review("request_changes", [finding("None input crashes f")]))
    assert d["action"] == "ESCALATE" and "round limit" in d["reason"]

    rl.main(type("A", (), {"root": str(repo), "action": "start", "base": "main", "max_rounds": 3, "restart": True})())
    rl.record_review(repo, review("request_changes", [finding("Needs product decision on rounding")]))
    rl.record_response(repo, [{"id": "R1-1", "action": "deferred", "reason": "PM decides rounding"}])
    d = rl.record_review(repo, review("approve"))
    assert d["action"] == "ESCALATE" and "deferred" in d["reason"]


def test_gate_runs_tests_and_dsguard(repo):
    rl.start(repo, "main")
    assert rl.gate(repo)["passed"]
    (repo / "dsguard.toml").write_text((repo / "dsguard.toml").read_text().replace('"true"', '"false"'))
    result = rl.gate(repo)
    assert not result["passed"] and result["tests"]["passed"] is False
    (repo / "dsguard.toml").write_text((repo / "dsguard.toml").read_text().replace('"false"', '"true"'))
    (repo / "leak.py").write_text('k = "' + "AKIA" + "Q" * 16 + '"\n')
    git(repo, "add", "leak.py")
    git(repo, "commit", "-qm", "leak")
    result = rl.gate(repo)
    assert not result["passed"] and any("secrets" in b for b in result["dsguard_blocks"])


def _enforce(repo):
    cfg = (repo / "dsguard.toml").read_text()
    (repo / "dsguard.toml").write_text(cfg.replace('test_command = "true"', 'test_command = "true"\nenforce_on_push = true'))


def test_check_push_enforcement(repo):
    assert rl.check_push(repo) == (True, "not enforced")
    _enforce(repo)
    ok, why = rl.check_push(repo)
    assert not ok and "not been through" in why
    rl.start(repo, "main")
    rl.record_review(repo, review("approve"))
    assert rl.check_push(repo)[0]
    commit(repo, "def f(x):\n    return x * 3\n")
    ok, why = rl.check_push(repo)
    assert not ok and "after the approved" in why


def test_pre_push_hook_blocks_unapproved_commits(repo, tmp_path):
    _enforce(repo)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    env = os.environ | {"DSGUARD_BIN": str(TOOLKIT / "bin" / "dsguard"), "HOME": str(tmp_path / "home"),
                        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    git(repo, "config", "core.hooksPath", str(TOOLKIT / "git-hooks" / "global"), env=env)
    git(repo, "remote", "add", "origin", str(remote), env=env)
    blocked = subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "feature"], capture_output=True, text=True, env=env)
    assert blocked.returncode != 0 and "push blocked" in blocked.stderr
    rl.start(repo, "main")
    rl.record_review(repo, review("approve"))
    allowed = subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "feature"], capture_output=True, text=True, env=env)
    assert allowed.returncode == 0, allowed.stderr


def test_brief_carries_prior_findings_and_responses(repo):
    rl.start(repo, "main")
    rl.record_review(repo, review("request_changes", [finding("None input crashes f")]))
    rl.record_response(repo, [{"id": "R1-1", "action": "disputed", "reason": "validated upstream"}])
    brief = rl.review_brief(repo)
    assert brief["round"] == 2 and brief["reviewer_model"] == "fable"
    assert "R1-1" in brief["prior_findings"] and brief["author_responses"][0]["action"] == "disputed"
    assert json.dumps(brief)
