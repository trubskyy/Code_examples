---
description: Author/reviewer loop before a push. A stronger reviewer model reviews the branch, this session fixes or disputes every finding, and a deterministic referee loops until done or escalates to you. Then pushes and updates the PR.
argument-hint: "[base-ref, default origin/main] [--max-rounds N]"
allowed-tools: Bash(~/.claude/ds-toolkit/bin/dsguard:*), Bash(git:*), Bash(gh pr:*), Read, Grep, Glob, Edit, Write, Agent, Task
---

Run the review loop for the current branch. You are the **author**: you wrote this code and
you fix it. A separate **reviewer** subagent on a stronger model judges it. The
`dsguard loop` referee decides when to stop. Never decide that yourself, and never edit the
loop state by hand.

Arguments: `$ARGUMENTS`. Base defaults to `origin/main` (falling back to `main`, then `master`).

## 0. Start

1. Commit any work in progress first. The reviewer reviews commits, not the working tree.
2. Run:
   ```bash
   ~/.claude/ds-toolkit/bin/dsguard loop start --base <base> [--max-rounds N]
   ```
   This resumes an active loop on this branch if there is one. It prints the reviewer model.

## 1. Gate (every round, before the reviewer)

```bash
~/.claude/ds-toolkit/bin/dsguard loop gate
```

It runs the dsguard checks on the diff and the project's tests. If it fails, fix the failures,
commit, and gate again. Don't spend a review on red checks. If you can't get it green after
two attempts, stop and report to the user: the loop escalates on a broken gate.

## 2. Review (fresh reviewer, stronger model)

```bash
~/.claude/ds-toolkit/bin/dsguard loop brief
```

Start the `pr-reviewer` agent with the Agent tool, setting **`model`** to the brief's
`reviewer_model` (default `fable`). Pass it only:
- the brief JSON (base, head, round, earlier findings, your responses);
- one line: "Review this branch per your instructions and return only the JSON block."

**Don't send** your reasoning, your plan, or arguments for why the code is right. The
reviewer's independence is the point.

Save its JSON block exactly as returned to `.dsguard/loop/review-<round>.json`, then:

```bash
~/.claude/ds-toolkit/bin/dsguard loop record-review .dsguard/loop/review-<round>.json
```

If the referee rejects the review as malformed (a missing field, a prior finding it didn't
account for, a dispute it didn't rule on), send the reviewer the error and ask for a corrected
block. Allow at most two corrections; after that, stop and report to the user.

## 3. Act on the decision

The referee returns one of three actions:

- **`DONE`**: go to step 5.
- **`ESCALATE`**: stop. Show the user the reason and `dsguard loop report`, then ask how to
  proceed. A person decides from here.
- **`CONTINUE`**: go to step 4.

## 4. Author round: answer every blocking finding

For each id in `blocking_open` (and, if `[review_loop] fix_minor` is true, the minor ones too):

| Action | When | What you do |
|---|---|---|
| `fixed` | The finding is right | Change the code, add or adjust a test that would have caught it, commit (`Review round N: fix R1-2 ...`) |
| `disputed` | You have a concrete technical reason it is wrong or not worth it | Write the reason |
| `deferred` | It needs a decision you can't make (product, data, access) | Write the reason. Deferred items always end in ESCALATE |

Rules:
- **Fix the cause, not the symptom.** A fix that only silences the finding will come back
  as a reopened finding, and two reopens escalate.
- **Dispute honestly.** Don't dispute to save effort; a dispute rejected twice escalates.
- **Keep fixes minimal.** Don't refactor beyond what the finding needs.

Write the responses, then record them:

```json
[{"id": "R1-2", "action": "fixed", "note": "guard empty frame; test_fit_empty"},
 {"id": "R1-3", "action": "disputed", "reason": "the input is validated upstream in load_orders()"}]
```

```bash
~/.claude/ds-toolkit/bin/dsguard loop record-response .dsguard/loop/response-<round>.json
```

Push the round's commits to the PR branch (`git push -u origin <branch>`), so the PR shows each
round. Then go back to step 1.

## 5. Done: push and update the PR

1. Make sure everything is pushed (`git push -u origin <branch>`).
2. **No PR exists:** open one (GitHub tools or `gh pr create`), with the loop report in the body.
3. **`post_to_pr` is true and a PR exists:** post `dsguard loop report` as a PR comment.
4. Tell the user:
   - the rounds and what was fixed;
   - what was disputed and how it was ruled;
   - any minor or nit findings left;
   - that merging is still their decision.

The approved commit is recorded. With `[review_loop] enforce_on_push = true`, the pre-push
hook blocks pushes of any later commit until the loop runs again.
