"""Review loop: the deterministic referee between an author model and a stronger reviewer model.

The /ds-review-loop command runs the conversation: the current session (the author) writes and
fixes code, and a fresh reviewer subagent on a stronger model reviews the branch's diff. This
module decides everything that must not be left to either model:

    gate             dsguard checks + the project's tests, on a clean tree at HEAD
    brief            only after a green gate on HEAD; pins the commit that is being reviewed
    record-review    validate the reviewer's JSON, match findings across rounds (new / persisting
                     / reopened / resolved), and decide CONTINUE, DONE or ESCALATE
    record-response  require the author to answer every open blocking finding (fixed / disputed
                     / deferred) before the next round
    status, report   the loop's state, and a Markdown summary for the PR
    restart          archive the loop and begin again (a person's decision after ESCALATE)
    check-push       pre-push: is every ref being pushed a commit the loop approved?

Stopping rules (ESCALATE hands the decision to a person):
    DONE      no open blocking finding and the reviewer approves
    ESCALATE  round limit; no progress (same blocking set twice); a finding reopened twice;
              a dispute rejected twice; blocking count rising two rounds running; deferred items

State lives in <project>/.dsguard/loop/<branch>.json (self-ignoring, never committed).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .core import BLOCKED_EXIT, ensure_state_dir, git, load_config

SEVERITIES = ("blocker", "major", "minor", "nit")
STOP_WORDS = {"the", "a", "an", "of", "in", "on", "to", "is", "and", "or", "for", "with", "when", "not", "be"}


class LoopError(SystemExit):
    """Invalid input from a model or the user; message is shown, exit code 2."""

    def __init__(self, message: str):
        super().__init__(f"dsguard loop: {message}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _branch(root: Path) -> str:
    # not --abbrev-ref: that says "heads/x" when a tag is also called x
    ref = (git(root, "symbolic-ref", "-q", "HEAD") or "").strip()
    return ref.removeprefix("refs/heads/") or "HEAD"


def _head(root: Path) -> str:
    return (git(root, "rev-parse", "HEAD") or "").strip()


def _state_path(root: Path, branch: str) -> Path:
    # The hash keeps team/x apart from team_x, and any branch apart from review-<n>.json.
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", branch)[:80]
    digest = hashlib.sha1(branch.encode("utf-8", "surrogateescape")).hexdigest()[:10]
    return ensure_state_dir(root) / "loop" / f"{readable}-{digest}.json"


def load_state(root: Path, branch: str | None = None) -> dict | None:
    path = _state_path(root, branch or _branch(root))
    return json.loads(path.read_text()) if path.exists() else None


def save_state(root: Path, state: dict) -> None:
    path = _state_path(root, state["branch"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def fingerprint(finding: dict) -> str:
    """Line-insensitive identity of a finding: category, file and the key words of its title.

    Used when the reviewer does not link a re-reported finding to its earlier id itself."""
    title = finding.get("title", "").lower()
    words = sorted({w for w in re.findall(r"\w+", title) if w not in STOP_WORDS})
    key = "|".join([finding.get("category", "").lower(), finding.get("file", ""), " ".join(words) or title])
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def _cfg(root: Path) -> dict:
    return load_config(root)["review_loop"]


def _archive(root: Path, branch: str) -> None:
    path = _state_path(root, branch)
    if path.exists():
        path.rename(path.with_suffix(f".{datetime.now():%Y%m%d%H%M%S%f}.json"))


def restart(root: Path, base: str, max_rounds: int | None = None) -> dict:
    """Archive this branch's loop and begin again."""
    if git(root, "rev-parse", "--verify", f"{base}^{{commit}}") is None:
        raise LoopError(f"base {base!r} is not a commit (fetch it first?)")
    _archive(root, _branch(root))
    return start(root, base, max_rounds)


def start(root: Path, base: str, max_rounds: int | None = None) -> dict:
    cfg = _cfg(root)
    if git(root, "rev-parse", "--verify", f"{base}^{{commit}}") is None:
        raise LoopError(f"base {base!r} is not a commit (fetch it first?)")
    branch = _branch(root)
    existing = load_state(root, branch)
    if existing and existing["branch"] != branch:
        raise LoopError(f"state file for {branch!r} belongs to {existing['branch']!r}; rename one of the branches")
    if existing and existing["status"] == "active":
        return existing  # resume
    if existing and existing["status"] == "escalated":
        raise LoopError("loop is escalated and waits for a person; only they decide to run "
                        "`dsguard loop restart --base <ref>`")
    if existing:
        _archive(root, branch)
    state = {"branch": branch, "base": base, "started": _now(), "status": "active", "decision": None,
             "max_rounds": max_rounds or cfg["max_rounds"], "block_on": cfg["block_on"],
             "reviewer_model": cfg["reviewer_model"], "rounds": [], "findings": {}, "approved_head": None}
    save_state(root, state)
    return state


def _require_active(root: Path) -> dict:
    state = load_state(root)
    if state is None:
        raise LoopError("no review loop on this branch; run `dsguard loop start --base <ref>`")
    if state["branch"] != _branch(root):
        raise LoopError(f"this loop state belongs to {state['branch']!r}, not the current branch")
    if state["status"] == "escalated":
        raise LoopError("loop is escalated and waits for a person")
    if state["status"] != "active":
        raise LoopError(f"loop is {state['status']}; start a new one with `dsguard loop start --base <ref>`")
    return state


def gate(root: Path) -> dict:
    """Deterministic checks before a review round: dsguard on the diff plus the project's tests."""
    from . import daybook  # local import: daybook imports core only

    state = _require_active(root)
    # Until this run finishes, the last gate is a failed one: a crash must not leave an older pass standing.
    state.setdefault("gates", []).append({"at": _now(), "head": None, "passed": False})
    save_state(root, state)
    cfg = load_config(root)
    # What is tested must be what is approved: pin HEAD and require the tree to equal it, first.
    head = _head(root)
    dirty = git(root, "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude).dsguard")
    dsguard = Path(__file__).resolve().parent.parent / "bin" / "dsguard"
    run = subprocess.run([sys.executable, str(dsguard), "run", "all", "--changed-since", state["base"],
                          "--format", "json", "--no-save"], cwd=root, capture_output=True, text=True, errors="replace")
    try:
        report = json.loads(run.stdout)
        blocks = [f"{r['name']}: {f['path']}:{f.get('line') or ''} {f['message']}"
                  for r in report["results"] for f in r["findings"] if f["severity"] == "block"]
        blocks += [f"{r['name']}: check crashed: {'; '.join(r['notes'])}"
                   for r in report["results"] if r["status"] == "error"]
    except (ValueError, KeyError):
        blocks = [f"dsguard did not run cleanly: {run.stderr.strip()[:300]}"]
    if dirty is None:
        blocks.append("git status failed")
    elif dirty.strip():
        blocks.append("working tree differs from HEAD; commit, ignore or stash first:\n" + dirty.rstrip()[:1000])
    tests = cfg["review_loop"]["test_command"] or daybook.detect_tests(root, daybook.discover_files(root), cfg["daybook"])
    test_result = {"command": tests, "passed": None, "tail": ""}
    if tests:
        proc = subprocess.run(tests, shell=True, cwd=root, capture_output=True, text=True, errors="replace",
                              env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
        test_result.update(passed=proc.returncode == 0, tail=(proc.stdout + proc.stderr)[-1500:])
    after = git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude).dsguard")
    if after is None or (after.strip() and not (dirty or "").strip()):
        blocks.append("the checks or tests changed committed files, so they didn't test HEAD:\n" + (after or "")[:1000])
    if _head(root) != head:
        blocks.append("HEAD moved while the gate ran; gate again")
    result = {"at": _now(), "head": head, "dsguard_blocks": blocks, "tests": test_result,
              "passed": not blocks and test_result["passed"] is not False}
    state["gates"][-1] = result
    save_state(root, state)
    return result


def _require_green_gate(state: dict, head: str) -> None:
    latest = (state.get("gates") or [{}])[-1]
    if latest.get("head") != head or not latest.get("passed"):
        raise LoopError(f"the latest gate has not passed on {head[:8]}; run `dsguard loop gate`")


def _validate_review(review: dict) -> None:
    if review.get("verdict") not in ("approve", "request_changes"):
        raise LoopError("review.verdict must be 'approve' or 'request_changes'")
    for key in ("findings", "resolved_prior", "disputes"):
        if not isinstance(review.get(key, []), list):
            raise LoopError(f"review.{key} must be a list")
    for i, f in enumerate(review.get("findings", [])):
        missing = [k for k in ("severity", "category", "file", "title", "evidence", "fix") if not f.get(k)]
        if missing:
            raise LoopError(f"finding {i} is missing {', '.join(missing)}")
        if f["severity"] not in SEVERITIES:
            raise LoopError(f"finding {i}: severity must be one of {', '.join(SEVERITIES)}")
        if f["severity"] in ("blocker", "major") and not f.get("failure_scenario"):
            raise LoopError(f"finding {i}: a {f['severity']} needs a concrete failure_scenario")
    linked = [f["prior_id"] for f in review.get("findings", []) if f.get("prior_id")]
    if len(linked) != len(set(linked)):
        raise LoopError("each prior_id may be used by one finding only; report a second problem without prior_id")
    for d in review.get("disputes", []):
        if d.get("ruling") not in ("accept_dispute", "reject_dispute") or not d.get("prior_id"):
            raise LoopError("each dispute needs prior_id and ruling accept_dispute|reject_dispute")


def _blocking(state: dict) -> dict[str, dict]:
    return {fid: f for fid, f in state["findings"].items()
            if f["status"] == "open" and f["severity"] in state["block_on"]}


def _match_priors(findings: dict, items: list[dict], closing: set[str]) -> list[str | None]:
    """The earlier finding each reported item re-reports, using each earlier finding at most once.

    Explicit prior_ids claim first. Then equal fingerprints pair up, preferring the same severity
    and the nearest line. A closed finding is only revived by an item at least as severe. Deferred
    findings and those this review closes are never matched implicitly: that item is a new finding."""
    matched = [i["prior_id"] if i.get("prior_id") in findings else None for i in items]
    free = {fid for fid, f in findings.items() if f["status"] != "deferred"} - set(matched) - closing
    rank = SEVERITIES.index
    for n in sorted(range(len(items)), key=lambda n: rank(items[n]["severity"])):
        if matched[n]:
            continue
        item, fp = items[n], fingerprint(items[n])
        candidates = [
            fid for fid in free if findings[fid]["fingerprint"] == fp
            and (findings[fid]["status"] == "open"
                 or rank(item["severity"]) <= rank(findings[fid]["severity"]))
        ]
        if candidates:
            def distance(fid: str) -> tuple:
                prior = findings[fid]
                a, b = prior.get("line"), item.get("line")
                gap = abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0
                return prior["severity"] != item["severity"], gap, fid

            matched[n] = min(candidates, key=distance)
            free.remove(matched[n])
    return matched


def record_review(root: Path, review: dict) -> dict:
    """Merge a reviewer's round into the state and decide what happens next."""
    state = _require_active(root)
    _validate_review(review)
    rnd = len(state["rounds"]) + 1
    if rnd > 1 and state["rounds"][-1].get("responses") is None:   # [] is a valid answer: nothing was blocking
        raise LoopError("the author has not answered the previous round (record-response) yet")
    reviewing = state.get("reviewing")
    if not reviewing:
        raise LoopError("no review is in progress; run `dsguard loop gate`, then `dsguard loop brief`")
    reviewed_head = reviewing["head"]
    if _head(root) != reviewed_head:
        raise LoopError(f"HEAD moved since the brief ({reviewed_head[:8]} was reviewed); "
                        "discard this review, then gate, brief and review again")
    _require_green_gate(state, reviewed_head)
    if review.get("head") != reviewed_head:
        raise LoopError(f"review.head must be the brief's head {reviewed_head}; "
                        f"this review is for {review.get('head') or 'an unnamed commit'}")
    findings, items = state["findings"], review.get("findings", [])
    closing = set(review.get("resolved_prior", [])) | {
        d["prior_id"] for d in review.get("disputes", []) if d["ruling"] == "accept_dispute"}
    matched = _match_priors(findings, items, closing)
    events = {"new": [], "persisting": [], "reopened": [], "resolved": []}

    # The reviewer must account for every blocking finding still open (resolve it or re-report it)
    # and rule on every dispute the author raised, so nothing is dropped or waved through silently.
    if state["rounds"]:
        reported = set(matched)
        ruled = {d["prior_id"] for d in review.get("disputes", [])}
        accounted = set(review.get("resolved_prior", [])) | reported | ruled
        unaccounted = [fid for fid in _blocking(state) if fid not in accounted]
        if unaccounted:
            raise LoopError("the review must resolve, re-report or rule on every open blocking finding; "
                            f"missing: {', '.join(unaccounted)}")
        disputed = {r["id"] for r in state["rounds"][-1]["responses"] if r["action"] == "disputed"}
        if disputed - ruled:
            raise LoopError(f"the review must rule on every dispute; missing: {', '.join(sorted(disputed - ruled))}")

    for prior in review.get("resolved_prior", []):
        if prior in findings and findings[prior]["status"] == "open":
            findings[prior].update(status="resolved", resolved_round=rnd)
            events["resolved"].append(prior)
    for d in review.get("disputes", []):
        f = findings.get(d["prior_id"])
        if not f or f["status"] == "deferred":
            continue  # only a person closes a deferred finding
        f.setdefault("dispute_rulings", []).append({"round": rnd, "ruling": d["ruling"], "reason": d.get("reason", "")})
        if d["ruling"] == "accept_dispute":
            f.update(status="dismissed", resolved_round=rnd)

    seen_now = set()
    for item, fid in zip(items, matched, strict=True):
        fp = fingerprint(item)
        if fid:
            f = findings[fid]
            deferred = f["status"] == "deferred"
            severity = item["severity"]
            # Only an explicit prior_id may lower a severity, and never a deferred finding's.
            if deferred or item.get("prior_id") != fid:
                severity = min(severity, f["severity"], key=SEVERITIES.index)
            kind = "reopened" if f["status"] == "resolved" else "persisting"
            if kind == "reopened":
                f["reopened"] = f.get("reopened", 0) + 1
            f.update({k: item[k] for k in ("title", "evidence", "fix", "line", "failure_scenario") if k in item},
                     severity=severity, status="deferred" if deferred else "open", last_round=rnd)
        else:
            fid = f"R{rnd}-{len([k for k in findings if k.startswith(f'R{rnd}-')]) + 1}"
            findings[fid] = {**{k: item.get(k) for k in ("severity", "category", "file", "line", "title", "evidence",
                                                          "failure_scenario", "fix")},
                             "fingerprint": fp, "status": "open", "first_round": rnd, "last_round": rnd}
            kind = "new"
        events[kind].append(fid)
        seen_now.add(fid)

    blocking = _blocking(state)
    state["rounds"].append({"round": rnd, "head": reviewed_head, "at": _now(), "verdict": review["verdict"],
                            "summary": review.get("summary", ""), "events": events,
                            "blocking_open": sorted(blocking), "responses": None})
    state["reviewing"] = None
    decision = decide(state)
    state["decision"] = decision
    if decision["action"] == "DONE":
        state.update(status="done", approved_head=reviewed_head)
    elif decision["action"] == "ESCALATE":
        state["status"] = "escalated"
    save_state(root, state)
    return decision


def decide(state: dict) -> dict:
    """Pure function of the state: what the loop does next, and why."""
    rounds = state["rounds"]
    current = rounds[-1]
    blocking = _blocking(state)
    deferred = [fid for fid, f in state["findings"].items() if f["status"] == "deferred"]

    def out(action, reason):
        return {"action": action, "reason": reason, "round": current["round"], "blocking_open": sorted(blocking),
                "deferred": deferred}

    if not blocking:
        if deferred:
            return out("ESCALATE", f"no blocking findings left, but {len(deferred)} deferred to a person: {', '.join(deferred)}")
        if current["verdict"] == "approve":
            return out("DONE", "reviewer approves and no blocking finding is open")
        return out("CONTINUE", "no blocking finding open but the reviewer has not approved; address its remaining points")
    for fid, f in state["findings"].items():
        if f.get("reopened", 0) >= 2:
            return out("ESCALATE", f"{fid} was fixed and came back twice (ping-pong): {f['title']}")
        rejected = [r for r in f.get("dispute_rulings", []) if r["ruling"] == "reject_dispute"]
        if len(rejected) >= 2 and f["status"] == "open":
            return out("ESCALATE", f"author and reviewer disagree on {fid} after two rounds: {f['title']}")
    if len(rounds) >= 2 and set(rounds[-2]["blocking_open"]) == set(current["blocking_open"]) and not current["events"]["resolved"]:
        return out("ESCALATE", "no progress: the same blocking findings are open as last round")
    if len(rounds) >= 3 and len(rounds[-3]["blocking_open"]) < len(rounds[-2]["blocking_open"]) < len(current["blocking_open"]):
        return out("ESCALATE", "blocking findings grew two rounds running: fixes are introducing new problems")
    if current["round"] >= state["max_rounds"]:
        return out("ESCALATE", f"round limit ({state['max_rounds']}) reached with {len(blocking)} blocking finding(s) open")
    return out("CONTINUE", f"{len(blocking)} blocking finding(s) for the author to fix or dispute")


def record_response(root: Path, responses: list[dict]) -> dict:
    """The author's answer to every open blocking finding of the latest round."""
    state = _require_active(root)
    if not state["rounds"]:
        raise LoopError("no review round to respond to")
    current = state["rounds"][-1]
    findings = state["findings"]
    answered = {}
    for r in responses:
        fid, action = r.get("id"), r.get("action")
        if fid not in findings:
            raise LoopError(f"unknown finding id {fid!r}")
        if action not in ("fixed", "disputed", "deferred"):
            raise LoopError(f"{fid}: action must be fixed, disputed or deferred")
        if action in ("disputed", "deferred") and not r.get("reason"):
            raise LoopError(f"{fid}: a {action} finding needs a reason")
        answered[fid] = r
        if action == "deferred":
            findings[fid]["status"] = "deferred"
        findings[fid].setdefault("responses", []).append({"round": current["round"], **r})
    missing = [fid for fid in current["blocking_open"] if fid not in answered]
    if missing:
        raise LoopError(f"every open blocking finding needs a response; missing: {', '.join(missing)}")
    current["responses"] = list(answered.values())
    current["fix_head"] = _head(root)
    save_state(root, state)
    return {"answered": len(answered), "fixed": sum(r["action"] == "fixed" for r in answered.values()),
            "disputed": sum(r["action"] == "disputed" for r in answered.values()),
            "deferred": sum(r["action"] == "deferred" for r in answered.values())}


def review_brief(root: Path) -> dict:
    """What the next reviewer needs: base, head, prior findings still in play, and the author's answers.

    Pins both commits: record_review approves this head and no other."""
    state = _require_active(root)
    head = _head(root)
    _require_green_gate(state, head)
    base = (git(root, "rev-parse", "--verify", f"{state['base']}^{{commit}}") or "").strip()
    if not base:
        raise LoopError(f"base {state['base']!r} is no longer a commit")
    state["reviewing"] = {"base": base, "head": head, "at": _now()}
    save_state(root, state)
    last = state["rounds"][-1] if state["rounds"] else None
    in_play = {fid: {k: f.get(k) for k in ("severity", "category", "file", "line", "title", "fix", "status")}
               for fid, f in state["findings"].items() if f["status"] in ("open", "deferred", "resolved", "dismissed")}
    return {"round": len(state["rounds"]) + 1, "max_rounds": state["max_rounds"], "base": base,
            "head": head, "reviewer_model": state["reviewer_model"], "block_on": state["block_on"],
            "prior_findings": in_play, "author_responses": (last or {}).get("responses") or []}


def report(state: dict) -> str:
    lines = [f"## Review loop: {state['status'].upper()}",
             f"Branch `{state['branch']}` vs `{state['base']}` · reviewer: {state['reviewer_model']} · "
             f"{len(state['rounds'])} round(s) of max {state['max_rounds']}"]
    if state.get("decision"):
        lines.append(f"\n**Decision:** {state['decision']['action']}: {state['decision']['reason']}")
    lines += ["", "| Round | Verdict | New | Resolved | Reopened | Blocking open |", "|---|---|---|---|---|---|"]
    for r in state["rounds"]:
        e = r["events"]
        lines.append(f"| {r['round']} | {r['verdict']} | {len(e['new'])} | {len(e['resolved'])} | "
                     f"{len(e['reopened'])} | {len(r['blocking_open'])} |")
    by_status: dict[str, list] = {}
    for fid, f in state["findings"].items():
        by_status.setdefault(f["status"], []).append((fid, f))
    for status in ("open", "deferred", "dismissed", "resolved"):
        items = by_status.get(status, [])
        if items:
            lines.append(f"\n**{status.capitalize()} ({len(items)})**")
            for fid, f in sorted(items):
                loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
                note = ""
                if f.get("responses"):
                    last = f["responses"][-1]
                    note = f" · author: {last['action']}" + (f" ({last['reason']})" if last.get("reason") else "")
                lines.append(f"- `{fid}` [{f['severity']}] {f['title']} ({loc}){note}")
    return "\n".join(lines) + "\n"


def _approved_commits(root: Path) -> set[str]:
    folder = root / ".dsguard" / "loop"
    approved = set()
    for path in folder.glob("*.json") if folder.is_dir() else ():
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        # review-<n>.json, response-<n>.json and archives share the folder: current states only.
        if not isinstance(state, dict) or not isinstance(state.get("branch"), str):
            continue
        if path == _state_path(root, state["branch"]) and state.get("status") == "done":
            approved.add(state.get("approved_head"))
    return approved - {None}


def _judge_branch(root: Path, branch: str, sha: str) -> tuple[bool, str]:
    state = load_state(root, branch)
    if not state or state["branch"] != branch:
        return False, "this branch has not been through /ds-review-loop"
    if state["status"] == "done":
        if state["approved_head"] == sha:
            return True, "approved by the review loop"
        return False, (f"commits after the approved one ({(state['approved_head'] or '')[:8]}); "
                       "run /ds-review-loop again")
    if state["status"] == "escalated":
        return False, (f"the review loop escalated: {state['decision']['reason']} "
                       "(a person must decide; DSGUARD_SKIP=1 to push anyway)")
    return False, "this branch has not been through /ds-review-loop"


def _judge_ref(root: Path, line: str, exempt: list[str], published: list[str]) -> tuple[bool, str]:
    """One pre-push stdin line: <local ref> <local sha> <remote ref> <remote sha>."""
    fields = line.rsplit(" ", 3)  # only ASCII space separates; only the local ref can contain one
    if len(fields) != 4 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", fields[1]):
        return False, f"cannot parse pushed ref {line!r}"
    local_ref, sha, remote_ref, _ = fields
    if set(sha) == {"0"}:
        return True, "deletion"
    heads = "refs/heads/"
    if remote_ref.startswith(heads):
        destination = remote_ref.removeprefix(heads)
        if destination in exempt:  # the destination, so an exempt local branch can't be pushed elsewhere
            return True, f"{destination} is exempt"
        if local_ref.startswith(heads):
            branch = local_ref.removeprefix(heads)
        else:
            branch = _branch(root) if local_ref == "HEAD" else destination
        ok, why = _judge_branch(root, branch, sha)
        return ok, f"{destination}: {why}"
    # Tags and other refs: an approved commit, or one a remote already has.
    target = (git(root, "rev-parse", "--verify", f"{sha}^{{commit}}") or "").strip()
    if target and (target in _approved_commits(root)
                   or git(root, "rev-list", "-n", "1", target, "--not", "--remotes") == ""
                   or any(git(root, "merge-base", "--is-ancestor", target, tip) is not None for tip in published)):
        return True, f"{remote_ref}: already reviewed or published"
    return False, f"{remote_ref}: points at a commit the review loop has not approved"


def check_push(root: Path, refs: str | None = None) -> tuple[bool, str]:
    """pre-push: allow only commits the loop approved, when [review_loop] enforce_on_push is on.

    refs is what git gives the hook on stdin. Without it, judge the current branch at HEAD."""
    cfg = _cfg(root)
    if not cfg["enforce_on_push"]:
        return True, "not enforced"
    if refs is None:
        branch = _branch(root)
        if branch in cfg["exempt_branches"]:
            return True, f"{branch} is exempt"
        return _judge_branch(root, branch, _head(root))
    # Enforcement is known to be on from here: whatever can't be judged is not approved.
    def judge(line: str) -> tuple[bool, str]:
        try:
            return _judge_ref(root, line, cfg["exempt_branches"], published)
        except Exception as exc:
            return False, f"cannot check {line!r} ({type(exc).__name__}: {exc})"

    def to_branch(line: str) -> bool:
        return line.rsplit(" ", 2)[-2:][0].startswith("refs/heads/")

    # Branches first: a tag may point into a branch that this same push is allowed to publish.
    published: list[str] = []
    verdicts = []
    for line in sorted([r for r in refs.split("\n") if r], key=lambda r: not to_branch(r)):
        verdicts.append(judge(line))
        if verdicts[-1][0] and to_branch(line):
            sha = line.rsplit(" ", 3)[1]
            if set(sha) != {"0"}:
                published.append(sha)
    refused = [why for ok, why in verdicts if not ok]
    if refused:
        return False, "; ".join(refused)
    return True, "; ".join(why for _, why in verdicts) or "nothing to push"


def main(args) -> int:
    from .core import find_root

    root = find_root(Path(args.root))
    if args.action in ("start", "restart"):
        begin = restart if args.action == "restart" else start
        state = begin(root, args.base or "origin/main", args.max_rounds)
        print(json.dumps({k: state[k] for k in ("branch", "base", "max_rounds", "reviewer_model", "status")}, indent=2))
    elif args.action == "gate":
        result = gate(root)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    elif args.action == "brief":
        print(json.dumps(review_brief(root), indent=2))
    elif args.action == "record-review":
        data = json.loads(Path(args.file).read_text())
        decision = record_review(root, data)
        print(json.dumps(decision, indent=2))
    elif args.action == "record-response":
        data = json.loads(Path(args.file).read_text())
        print(json.dumps(record_response(root, data if isinstance(data, list) else data["responses"]), indent=2))
    elif args.action in ("status", "report"):
        state = load_state(root)
        if state is None:
            print("no review loop on this branch")
            return 0
        print(report(state) if args.action == "report" else json.dumps(state, indent=2))
    elif args.action == "check-push":
        refs = sys.stdin.buffer.read().decode("utf-8", "surrogateescape") if args.refs_stdin else None
        ok, why = check_push(root, refs)
        print(f"dsguard review loop: {'push allowed' if ok else 'push blocked'}: {why}", file=sys.stderr)
        return 0 if ok else BLOCKED_EXIT
    return 0
