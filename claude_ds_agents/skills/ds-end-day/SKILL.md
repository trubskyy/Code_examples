---
name: ds-end-day
description: End-of-day wrap-up for the current project. Commits the day's work through the dsguard gate, runs through the Markdown files and updates them, writes today's entry in the project story (docs/journal/STORY.md), drafts the plan for the next working day, and commits. On the weekly review day (Friday by default) it first runs a comment pass over every code file changed that week, proven not to change behavior. Use when the user says "end of day", "wrap up" or "done for today".
allowed-tools: Bash(~/.claude/ds-toolkit/bin/dsguard:*), Bash(git status:*), Bash(git log:*), Bash(git diff:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(python -m pytest:*), Read, Grep, Glob, Write, Edit, Agent, Task
---

# End of day

Close the day so that tomorrow starts from an accurate picture: work committed, docs true,
story written, plan drafted. Facts come from `dsguard day`; you write. Ask at most one
question (step 2) unless something blocks.

## 1. Gather facts

```bash
~/.claude/ds-toolkit/bin/dsguard day status --format json
```

Note `is_weekly_review_day`, `next_workday`, `plans.today`, `commits_today`, `changed_today`,
`uncommitted`, `md_check_today` and `weekly_review`. If the journal is missing, run
`dsguard day init`.

## 2. Commit today's remaining work (one question)

If `uncommitted` lists work files, show them grouped (code, docs, notebooks, other) and ask:
**"Commit these as today's work?"** Leave out anything that looks like scratch, data, outputs
or credentials, and say what you left out.

On yes:
1. Run the project's fast tests if there are any (`tests` in the status). If they fail, say so.
   Ask whether to commit anyway (the user decides), and record the failure in the story.
2. Stage the files explicitly by name; never `git add -A`.
3. Run `~/.claude/ds-toolkit/bin/dsguard run commit --staged`. On a `block`, stop and report it.
4. Commit: `Work <date>: <one-line summary>`.

## 3. Weekly comment pass (only when `is_weekly_review_day`)

Scope: `weekly_review.files`, the code files changed since the last pass (or since Monday).
The tree must be clean after step 2; if it isn't, skip this step and say why.

Run the `documenter` agent on exactly those files with this brief: "Weekly comment pass. Make
every changed file very well commented: module docstrings, docstrings for public functions and
classes, and comments that explain *why*: business rules, data quirks, thresholds and their
origin, leakage guards, non-obvious pandas or SQL behavior. Don't restate code, don't change
behavior." For one or two small files, do it yourself under the same rules.

Then:

```bash
~/.claude/ds-toolkit/bin/dsguard verify-docs-only        # must pass: only docstrings/comments changed
~/.claude/ds-toolkit/bin/dsguard run docs --paths <files>  # coverage after the pass
```

If verify fails, revert the offending hunks and re-run until it passes. Commit as
`Weekly comment pass: week of <week_start>`, then run
`~/.claude/ds-toolkit/bin/dsguard day mark-review` (the marker file is committed in step 7).

## 4. Run through the Markdown files

Bring the docs in line with what actually changed. Base every edit on the diff
(`git diff <first commit today>~1..HEAD` or `git log --since=<date> -p`), not on memory.
- Every file in `md_check_today.mentions_changed_files` (and, on review day, the docs that
  mention files changed this week): check each statement about the changed code is still true.
  That covers flags, file names, steps, counts and status.
- Fix every entry in `md_check_today.broken_links`.
- README status and "next steps": move finished items out, and add new follow-ups discovered today.
- ARCHITECTURE (or similar), only if structure, contracts or data flow changed.
- Agent, skill or command docs, if their behavior changed.

Keep each file's tone and structure; this is maintenance, not a rewrite. If nothing needs
changing, say so rather than making cosmetic edits.

## 5. Update today's plan and write the story entry

**Plan**: in `docs/journal/plans/<today>.md`, tick what got done, using the evidence of
commits, files and the conversation. Add a short `## End of day` note: done, not done and why,
surprises.

**Story**: append to `docs/journal/STORY.md` (newest at the bottom):

```markdown
## 2026-10-23 (Fri)

<2–5 short paragraphs of narrative: what was worked on and why; decisions made and the reasoning
behind them; what was tried and dropped; problems found; anything a newcomer would need to
understand later. Only state numbers that appear in a report, test output or commit, and name
where they came from.>

**Next:** <one or two lines pointing at tomorrow's plan>
```

If an entry for today already exists, extend it instead of adding a second one. Write it as a
story someone can read top to bottom months later, not as a commit log. If there
was no activity today, write one line saying so.

On review day, also rewrite the `## Story so far` section at the top: at most 15 lines on where
the project stands, what has been built, what is in flight, and the open questions.

## 6. Draft the plan for the next working day

Create `docs/journal/plans/<next_workday>.md`, a draft that `ds-start-day` refines once the
user says how much time they have:
- Carry over the open items, with a reason where useful.
- Add the next logical steps from today's work, the README next steps, TODOs and open PRs.
- Use the same sections and estimates as the start-of-day plan: Needs you · Claude, this
  session · Unattended · Not today.
- First line: `Draft written at end of <today>; time available: to be confirmed at start of day.`

## 7. Commit the journal and docs

Stage explicitly:
- the updated Markdown files;
- `docs/journal/STORY.md`;
- today's and the next workday's plans;
- `docs/journal/.last-comment-review`, on review day.

Run `dsguard run commit --staged`, then commit:
`End of day <date>: docs, story, plan for <next_workday>`.

Push only if `push_after_commit` is true in the status (`[daybook] push = true` in
`dsguard.toml`). Otherwise say the commits are local and how many there are.

## 8. Report

```
End of day <date>
Committed: <n> commits (<subjects>)        Pushed: yes/no
Weekly comment pass: <n files, coverage before → after, verify-docs-only PASSED> | not review day
Docs updated: <files> | none needed
Story: entry added (<words> words)  Plan for <next_workday>: <n> items drafted
Needs attention: <failing tests, blocked items, anything left uncommitted on purpose>
```
