---
name: ds-start-day
description: Start-of-day planning for the current project. Finds what is still open (yesterday's plan, TODOs, next steps in the docs, uncommitted work), asks how much time you can give the project today, builds a plan that fits that time, and splits the work into what needs you, what Claude can do in this session, and what can run unattended. Use at the start of a working session, or when the user says "start of day", "plan my day" or "what should I work on".
allowed-tools: Bash(~/.claude/ds-toolkit/bin/dsguard:*), Bash(git status:*), Bash(git log:*), Bash(git diff:*), Read, Grep, Glob, Write, Edit
---

# Start of day

Plan today's work on this project around the time the user actually has. Facts come from
`dsguard day`; you decide priorities and wording. Keep the conversation short: one round of
questions, then the plan.

## 1. Gather facts (no questions yet)

```bash
~/.claude/ds-toolkit/bin/dsguard day status
~/.claude/ds-toolkit/bin/dsguard day status --format json     # when you need the detail
```

If it reports `story: missing`, run `~/.claude/ds-toolkit/bin/dsguard day init` (creates
`docs/journal/STORY.md` and `docs/journal/plans/`) and say you did.

Then read:
- `plans.today`: the draft written at the end of the last working day for today, if any. It is
  the starting point.
- `plans.previous`: its **open** items are carry-overs.
- The last one or two entries of `docs/journal/STORY.md`, for context and where things stand.
- `todos`, `next_steps` (from README/ARCHITECTURE/etc.) and `uncommitted` work.
- Open PRs and their CI or review state, if GitHub tools are available. A red PR outranks new work.

## 2. Show what is remaining, then ask

Give a compact picture (at most about 12 lines): carry-overs, blockers, the top items from
docs and TODOs, and uncommitted work. Then ask **in one message**:

1. **How much time can you give this project today?** (e.g. 30 min, 1 h, 2 h, half day, full day)
2. Anything fixed or urgent today (a meeting, a deadline, a priority change)? Optional.

Stop and wait for the answer. Don't plan before you know the time.

## 3. Build the plan to fit the time

**Order of priority**:
1. Blockers: red CI, broken builds, merge conflicts.
2. Whatever the user called urgent.
3. Carry-overs.
4. The next steps that unblock the most other work.
5. Nice-to-haves.

**Estimate every item**: S ≈ 30 min, M ≈ 1–2 h, L ≈ half a day. Split L items into steps that
fit; never plan an L for a 1-hour day.

**Budget only the user's time**. Their available time, minus a 20 % buffer for interruptions,
is the budget for "needs you" items. Each Claude item also costs review time: about 10–15 % of
its size for the user to check the result. Count that too. Whatever doesn't fit goes to
"Not today".

**Split by involvement**, which is the point of this plan:

| Bucket | What belongs there | Examples |
|---|---|---|
| **Needs you** | Decisions, domain knowledge, approvals, access, people | Modeling direction, metric or business definitions, approving a cleanup plan or merge, credentials, a stakeholder conversation, reviewing a draft |
| **Claude, this session** | Well-defined work Claude can finish and you review at the end | Docs, tests, refactors behind passing tests, code/SQL reviews, a dsguard audit, a leakage check or diagnostics on available data, drafting a summary |
| **Unattended** | Work that can run without anyone watching | Background subagents (audits, eval reruns with a known cost), scheduled `ds-monitor` drift checks, headless `claude -p "/ds-premerge"` |

For each Claude or unattended item, state what it needs from the user, if anything, before it
can start (data path, approval, budget for LLM calls). If an item needs a decision first, put
the decision in **Needs you** and the work after it.

## 4. Save the plan

Write `docs/journal/plans/<today>.md` (replacing the draft, but keep a line saying it was
drafted on <date>):

```markdown
# Plan for 2026-10-23 (Fri)

Time available: 2 h  ·  budget for your tasks: ~1 h 35 min  ·  drafted at end of 2026-10-22, refined this morning

## Needs you
- [ ] Decide churn label window: 30 vs 60 days (S, ~20 min). Why: changes the target definition

## Claude, this session
- [ ] Add tests for feature pipeline edge cases (M; your review ~10 min)

## Unattended
- [ ] Background: security-auditor on `src/` (no input needed)

## Not today
- Retrain with new features (L; blocked by the label decision)
```

Don't commit; the end-of-day skill commits the journal.

## 5. Kick off

Show the plan and ask once: **"Start the Claude and unattended items now?"** On yes:
- Start unattended items in the background (background subagents, or the headless and
  scheduled commands above) and tell the user how they'll hear back.
- Begin the first Claude item, or the first "needs you" item if the user prefers to do theirs
  first.

As work completes during the day, tick items in the plan file (`- [x]`).
