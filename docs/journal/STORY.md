# Project story

A running narrative of this project: what was built, why, what was tried and dropped, and
what comes next. One entry per working day, newest at the bottom. Written at end of day by
the `ds-end-day` skill from commits, the day's plan and the conversation; edit freely.

## Story so far

_Last refreshed 2026-10-23._

The repo began as a single runnable example comparing LLM fact extraction with vector search.
It is now mainly home to **`claude_ds_agents/`**, a personal Claude Code toolkit for a
data-science lead. The toolkit has 11 agents, 3 commands, 2 daily-rhythm skills, a
deterministic engine (`dsguard`) and harness scripts for models, leakage, LLM evals, drift and
verified summaries.

Its organising rule: **scripts decide whatever a script can decide; agents only judge.**

Built and tested so far:
- security and gateway guardrails;
- code, SQL (SQL Server), leakage and refactor reviews;
- model diagnostics for sklearn, LightGBM, XGBoost and PyTorch;
- LLM evals, drift monitoring and stakeholder summaries;
- docs and cleanup maintenance;
- start-of-day and end-of-day routines with this journal.

Still open:
- adopting the prewritten LLM gateway and the SQL Server table list on the work machine;
- a company license policy;
- verifying the PyTorch loader;
- running the agents in a live session;
- the `stats-reviewer` and `team-pr-reviewer` agents.

## 2026-10-18 (Sun)

The repository started with `llm_extraction_vs_vector_search/`. It is a small, offline-runnable
demo showing where vector search fails: negations ("compatible" vs "never compatible"),
near-identical IDs (`A4-99x` vs `A4-99y`), and facts that depend on section headings. It
contrasts that with structured LLM fact extraction and a hybrid of the two. It was written as
a reference example and calls the Anthropic API directly by design.

**Next:** turn day-to-day data-science chores into reusable agents.

## 2026-10-20 (Tue)

The question was which agents a data-science lead should have: someone who leads people,
works on tabular, text and LLM projects, and must keep LLM calls behind a company gateway. A
first list of six (data, code, refactoring, docs, cleanup, security) grew to cover what was
missing: leakage, LLM evaluation, model diagnostics, drift, stakeholder communication,
statistics and team review.

The defining decision came early: **make everything deterministic that can be**. Gateway
violations, secrets, license families, SQL anti-patterns and leakage signals became a
standard-library-only engine, `dsguard`. Agents run it first and can't overrule its blocks.
Two guardrails followed: a PreToolUse hook that denies gateway bypasses and secrets before
Claude writes them, and a git pre-commit gate. A missing external tool is always reported as
*skipped*, never as passed.

The day was built in four commits:
1. `dsguard` with six reviewer agents.
2. Leakage, model-diagnostics and LLM-eval harnesses with their agents.
3. Drift monitoring and fact-checked stakeholder summaries.
4. Adjustments for the work machine.

The tests caught real bugs along the way:
- creating a `KFold` object was being counted as "the split happened", hiding scalers fitted
  on all the data;
- `X_te` wasn't recognised as test data;
- a half-finished week raised false volume alarms on every drift run;
- nested-dict writes were flagged as pandas chained assignment.

The answers about the work environment shaped the last commit:
- **Gateway:** prewritten code, adopted on the work machine. So nothing company-specific was
  hardcoded; `dsguard suggest-gateway` and `/ds-setup` were added to read the gateway code and
  draft the config there.
- **SQL:** SQL Server, so T-SQL became the default, with rules for `NOLOCK`, cursors, INT
  division and `AVG(int)`.
- **Licenses:** no known policy, so packages are reported in two buckets, *always free* and
  *potential issues*, with plain-English obligations and no blocking.
- **Models:** sklearn, LightGBM/XGBoost and PyTorch, defined in code or stored as binaries.
  Hence `model_loader.py`, which identifies model files without executing them.

PyTorch downloads were blocked in the build environment, so the torch path is written but
unverified. PR #1 merged the toolkit.

**Next:** document what exists.

## 2026-10-22 (Thu)

Documentation caught up with the code:
- the toolkit README was rewritten into one coherent document (what it is, what it does by
  job, an honest status table);
- `ARCHITECTURE.md` was added, covering layers, contracts, flows, the trust model and known
  limitations;
- a root README now indexes the repository.

Writing the docs exposed a miscount: there are 11 agents, not 12, and four are strictly
read-only. Restarting the branch after PR #1 needed a force-push, which was blocked. Since
GitHub diffs a PR from the common commit, a plain follow-up PR worked instead. PR #2 merged
the docs.

**Next:** a daily working rhythm.

## 2026-10-23 (Fri)

Added the daily rhythm:
- **`ds-start-day`** reads what's open (yesterday's plan, TODOs, next-step sections, uncommitted
  work, open PRs). It asks how much time the user has, then builds a plan that fits that
  budget. The plan is split into what needs the user, what Claude can do in session, and what
  can run unattended.
- **`ds-end-day`** commits the day's work through the dsguard gate, brings the Markdown files
  in line with the diff, writes the day's entry in this story, drafts the next workday's
  plan, and commits.
- **Fridays** add a comment pass over every code file changed that week. It is proven not to
  change behavior by `dsguard verify-docs-only`, and a marker file records where the last
  review ended.

As before, the facts are deterministic: `dsguard day` computes workdays, open items, files
changed this week and broken Markdown links. Two of its first heuristics were fixed straight
away:
- a `# TODO` inside a Python string counted as a TODO, so Python now uses real comment tokens;
- Markdown table headers were listed as next-step items.

This file is the continuous story the user asked for, seeded from the git history. The PR #2
branch was fast-forwarded onto the merged default branch instead of being rewritten.

**Next:** run `/ds-end-day` at the end of the session, Friday edition, to exercise the full
loop on this repo. Then adopt the toolkit on the work machine with `/ds-setup`.
