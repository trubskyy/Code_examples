---
description: Pre-merge gate. Runs dsguard on the diff, then the relevant review agents in parallel, and writes one prioritized report.
argument-hint: "[base-ref, default origin/main]"
allowed-tools: Bash(~/.claude/ds-toolkit/bin/dsguard:*), Bash(git diff:*), Bash(git status:*), Bash(git rev-parse:*), Bash(git fetch:*), Read, Grep, Glob, Write, Agent, Task
---

Run the pre-merge gate for this repository. Base ref: `$ARGUMENTS` (if empty, use `origin/main`,
falling back to `main` and then `master`, whichever exists).

## Step 1: deterministic gate (always first)

```bash
~/.claude/ds-toolkit/bin/dsguard run all --changed-since <base> --format text --limit 30
```

Keep the output. Any `BLOCK` finding means the verdict is **BLOCKED**, whatever the agents say
later. Record every check reported as `skipped` or with a "not installed" note.

If no files changed, say so and stop.

## Step 2: pick reviewers from the diff

Run `git diff --name-only <base>...HEAD` plus the uncommitted changes, then dispatch **in
parallel** (one message, several Agent calls), passing each agent the base ref and the
changed file list:

| Agent | When |
|-------|------|
| `security-auditor` | always |
| `code-reviewer` | any `.py` or `.ipynb` changed |
| `sql-reviewer` | any `.sql` changed, the dsguard sql check found queries in changed files, or dbt models changed |
| `leakage-auditor` | changed files touch training, feature or split code (sklearn/xgboost/lightgbm/torch imports, `fit`, feature SQL), or dsguard `leakage` has findings. Tell it: **static and code review only**; no data materialization in a pre-merge run. |
| `llm-evaluator` | prompts, LLM call sites or parsing of LLM output changed **and** a golden set plus `.dsguard/evals/last.json` exist. Tell it: **re-run against the baseline only**, and ask before running more than 200 calls. |

Tell each agent: "Scope: changed since <base>. Files: <list>. Use `--changed-since <base>` for
dsguard. Return your report in your standard format."

Don't run `documenter`, `cleanup-agent`, `refactor-advisor` or `model-diagnostics` here. They
are on-demand (use `/ds-audit` or call them by name).

## Step 3: merge into one report

Write `.dsguard/reports/premerge-<YYYYMMDD-HHMM>.md` and print it:

```
# Pre-merge: <BLOCKED | FIX BEFORE MERGE | READY>
Base: <ref>  Files: <n>  Date: <date>

## Must fix (BLOCK + HIGH)
| # | Source | Location | Issue | Fix |
## Should fix (MEDIUM)
## Optional (LOW, top 10 only)
## Not covered
<skipped tools, agents not run, and anything an agent listed as not covered>
```

Rules for merging:
- De-duplicate: when the script and an agent report the same location and issue, keep one row and cite both sources.
- Don't drop or soften a dsguard `block`.
- Don't invent findings the agents didn't report.
- Verdict: any BLOCK means BLOCKED; else any HIGH means FIX BEFORE MERGE; else READY.
