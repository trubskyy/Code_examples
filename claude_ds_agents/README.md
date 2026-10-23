# Claude Code agents for data-science work

## What it is

A personal automation toolkit for a data-science lead who works on tabular, text and LLM
projects. It plugs into **Claude Code** on your own machine (`~/.claude`), so every project
gets the same specialist agents, commands and guardrails.

It has two halves that work together:

- **Deterministic scripts** (`dsguard`, `harness/`, hooks) decide everything a script can
  decide: gateway violations, secrets, license families, SQL anti-patterns, leakage signals,
  model metrics, drift statistics, whether a number in a summary is real. Their verdicts are
  repeatable and can block a commit.
- **Agents** (Claude subagents) make the judgement calls: is this join's grain right, is this
  feature known at prediction time, is this alert a pipeline break or a real shift, what should
  the executive summary say. Every agent runs its script first and can't overrule it.

How the pieces fit is described in [ARCHITECTURE.md](ARCHITECTURE.md).

## What it does

| Job | What happens | Main pieces |
|---|---|---|
| **Guard** | Before Claude writes a file or runs a command, a hook blocks code that bypasses the company LLM gateway or contains secrets. Commits are gated the same way. Security reviews cover data/PII exposure, unsafe code, package licenses (*always free* vs *potential issues*, explained in plain English) and known vulnerabilities. | `hooks/`, `git-hooks/`, `security-auditor` |
| **Review** | Finds bugs and inefficiencies in DS code and notebooks; correctness and cost problems in SQL (SQL Server by default); leakage in code and data; ranked refactoring proposals. | `code-reviewer`, `sql-reviewer`, `leakage-auditor`, `refactor-advisor` |
| **Analyze** | Loads models from sklearn, LightGBM, XGBoost or PyTorch (in code or as binaries) and diagnoses them: calibration, residual structure by feature, weak segments, drift over time, importance. Runs LLM evaluations through the gateway with regression diffs. | `model-diagnostics`, `llm-evaluator`, `harness/` |
| **Monitor** | Scheduled drift checks on deployed models; Claude is woken only on an ALERT, read-only, to triage it. | `drift-monitor`, `bin/ds-monitor` |
| **Communicate** | Writes executive, product or peer summaries in which every number is cited and machine-verified against the source reports. | `stakeholder-writer`, `harness/facts.py` |
| **Maintain** | Writes documentation, proven not to change code; proposes repository cleanup and applies it reversibly after your approval. | `documenter`, `cleanup-agent` |
| **Daily rhythm** | Start of day: what's left, how much time you have, a plan split into *needs you* / *Claude in session* / *unattended*. End of day: commit, update the Markdown files, add to the project story, draft tomorrow's plan, commit. Fridays add a comment pass over the week's changed files. | `ds-start-day`, `ds-end-day`, `dsguard day` |

## Status

| | |
|---|---|
| **Built** | 11 agents, 3 commands, 2 skills (start/end of day), 11 `dsguard` checks plus 5 tools, 6 harness scripts, the edit-time hook, the pre-commit hook, the scheduler wrapper, the installer |
| **Verified** | 65 tests; 64 pass with sklearn, LightGBM and XGBoost installed, and PyTorch skips. CLI runs end to end on synthetic data; installer install/uninstall in a throwaway home; the hook against sample tool calls. |
| **Not yet verified** | PyTorch model loading (implemented; PyTorch downloads were blocked in the build environment). Agents, commands and skills haven't been run inside a live Claude Code session. |
| **Waiting on the work machine** | Gateway adoption and SQL Server table list, both via `/ds-setup`. License policy (report-only until you set `block_families`). |
| **Next** | `stats-reviewer` (A/B tests) and `team-pr-reviewer` (see the end of this file) |

## Quick start

```bash
python3 claude_ds_agents/install.py --dry-run   # show what would change
python3 claude_ds_agents/install.py             # symlinks into ~/.claude, registers the hook, copies the policy file
pip install pip-audit bandit sqlfluff           # optional, recommended
# optional, recommended: install gitleaks to scan git history for secrets
```

Then in Claude Code:

1. `/ds-setup [path to the gateway client]` (once per machine) adopts the prewritten LLM
   gateway, the SQL Server tables, license reporting and model discovery, and writes
   `~/.claude/dsguard.toml`.
2. `/ds-premerge` before opening a PR.
3. `/ds-audit` when you inherit or hand over a project.
4. Any agent by name: "use the model-diagnostics agent on models/churn.txt", "use the
   stakeholder-writer for a 5-line exec email on the latest diagnostics".

Per repository, add the commit gate:
`ln -s ~/.claude/ds-toolkit/git-hooks/pre-commit .git/hooks/pre-commit`

Requirements: Python 3.11+. `dsguard`, the hooks, `llm_eval.py` and `facts.py` need only the
standard library; the other harness scripts need numpy and pandas. `install.py --uninstall`
reverses the install, and a `git pull` here updates every installed agent.

## Agents

| Agent | Writes | Use it for |
|---|---|---|
| `security-auditor` | nothing | Gateway compliance (including indirect bypasses), secrets, PII exposure, prompt-injection surface, licenses, vulnerabilities |
| `code-reviewer` | nothing | Silent pandas errors, leakage, reproducibility, fragile LLM calls, performance |
| `sql-reviewer` | nothing | Query grain, join fan-out, NULLs, time windows, SQL Server traps; then cost; optional estimated plan |
| `refactor-advisor` | nothing | Value/effort-ranked refactors with before/after sketches |
| `leakage-auditor` | scratch | The prediction moment, feature-by-feature checks, data-level leakage, eval contamination |
| `model-diagnostics` | scratch | Find and load the model, build the eval table, diagnose, rank next experiments |
| `llm-evaluator` | eval files | Golden sets, adapters to your gateway function, runs, regression diagnosis |
| `drift-monitor` | monitor config | Set up `monitor.toml`; triage ALERTs into pipeline break, shift, expected change or noise |
| `documenter` | docs only | Docstrings, README, data dictionary; must pass `verify-docs-only` |
| `cleanup-agent` | plan only | Verified cleanup plan; applies only after you approve; reversible |
| `stakeholder-writer` | summaries | Audience-specific summaries; `facts.py verify` must pass |

"Scratch" means helper scripts and extracts go under `.dsguard/scratch/`, never into project code.

## Daily skills

| Skill | What it does |
|---|---|
| `/ds-start-day` | Reads yesterday's plan, carry-overs, TODOs, next steps in the docs, uncommitted work and open PRs. Asks **how much time you have today**, then plans to fit it: estimates, a 20 % buffer, your review time for Claude's work. Splits the plan into **Needs you**, **Claude, this session** and **Unattended**, saves it to `docs/journal/plans/<today>.md`, and offers to start the work that doesn't need you. |
| `/ds-end-day` | Commits the remaining work (one confirmation, gated by `dsguard`). Updates the Markdown files that mention changed code and fixes broken links. Ticks today's plan, appends today's entry to **`docs/journal/STORY.md`**, drafts `plans/<next workday>.md`, and commits. Pushes only if `[daybook] push = true`. **On Fridays** it first runs a comment pass over every code file changed since the last review, proven behavior-neutral by `verify-docs-only`. |

The journal lives in the project and is committed with it:
- **`docs/journal/STORY.md`:** the continuous story of the project, one entry per working day, with a "Story so far" section refreshed weekly.
- **`docs/journal/plans/`:** one plan per day.

`dsguard day status` shows the facts both skills start from.

## Commands

| Command | What it does |
|---|---|
| `/ds-setup [gateway path]` | One-time adoption: install check → draft gateway config from the prewritten client (`dsguard suggest-gateway`) → validate it on a real project → SQL Server partitioned/large tables via read-only `sys.*` queries → license split → model scan → pre-commit links |
| `/ds-premerge [base]` | `dsguard` on the diff, then the relevant reviewers in parallel, merged into one report: BLOCKED / FIX BEFORE MERGE / READY |
| `/ds-audit [path]` | Whole-project review by all reviewers; cleanup in plan-only mode; suggests next agents |

## Deterministic vs judgement, per concern

| Concern | Script decides (`dsguard` / `harness/` / hooks) | Agent judges |
|---|---|---|
| **LLM gateway** | Provider SDK imports, provider hosts, clients without a gateway `base_url`, provider key env vars; hook denies at edit time | Indirect bypasses, whether the data sent is cleared for the gateway |
| **Secrets** | Token patterns, high-entropy assignments, credential files, gitleaks history; values masked | Fallback defaults, secrets printed in notebooks |
| **Data exposure** | Data/model files in git, notebook outputs, PII (email, SSN, Luhn-valid cards), open network binds, `share=True` | Whether outputs are sensitive, logging of prompts and responses |
| **Code security** | Unsafe deserialization, `eval`, `shell=True`, `verify=False`, `trust_remote_code`; bandit | Exploitability in context |
| **Licenses** | *Always free* vs *potential issues* (weak copyleft, unknown, strong copyleft, network copyleft, restricted), with the obligation in plain English; report-only until `block_families` is set | Model/dataset card terms |
| **Vulnerabilities** | pip-audit (skipped, not passed, if missing) | none |
| **SQL** | `SELECT *`, `NOT IN (SELECT)`, `= NULL`, a WHERE that kills a LEFT JOIN, non-sargable filters, TOP/LIMIT without ORDER BY, missing large-table filters, interpolated SQL; T-SQL: `GO`, `NOLOCK`, cursors, INT/INT division, `AVG(int)`, scalar UDFs; sqlfluff | Grain, fan-out, NULLs, collation, implicit conversion, point-in-time correctness, plans |
| **Code** | ruff, complexity, rename-insensitive duplicates, `iterrows`, concat in loops, chained assignment, `merge` without `validate=`, unseeded splits, swallowed exceptions | Silent data errors, reproducibility, LLM robustness |
| **Leakage** | Code: fit or resample before split, fit on holdout, `shift(-n)`, backfill. Data: shared ids and groups, duplicate rows and texts, temporal overlap, future timestamps, too-predictive columns, target copies, id/order leakage | What is known at prediction time, which fix |
| **Model diagnostics** | Metrics, calibration, thresholds, residual structure per feature, weak segments, trend over time, feature strength, permutation importance, worst rows | Baselines, causes, next experiments |
| **LLM evals** | Checks (exact, field, regex, JSON, numeric), pass rates by tag, consistency, latency, cost, judge-vs-human kappa, pass→fail diff | Golden-set design, judge rubric, regression diagnosis |
| **Drift** | PSI, KS, mean shift, out-of-range share, unseen categories, missingness, prediction and performance drift, schema, volume (partial periods excluded), OK/WATCH/ALERT | Cause and action |
| **Summaries** | Facts ledger; `verify` rejects uncited or mismatched numbers | Message, framing, caveats |
| **Docs** | Coverage, stale params; `verify-docs-only` proves only docs changed | Writing |
| **Cleanup** | Candidates; reversible apply/restore; hook asks on `rm -r` | Whether something is really unused |

## Scripts

`dsguard` (stdlib only):

```bash
~/.claude/ds-toolkit/bin/dsguard run all                        # or: security | quality | commit | <check>
~/.claude/ds-toolkit/bin/dsguard run all --changed-since origin/main --format json
~/.claude/ds-toolkit/bin/dsguard verify-docs-only
~/.claude/ds-toolkit/bin/dsguard suggest-gateway path/to/gateway_client
~/.claude/ds-toolkit/bin/dsguard day status                     # also: week-files | md-check | mark-review | init
~/.claude/ds-toolkit/bin/dsguard apply-cleanup .dsguard/cleanup-plan.json   # restore-cleanup <stamp> to undo
```

Checks: `gateway`, `secrets`, `exposure`, `pysec`, `licenses`, `vulns`, `sql`, `code`,
`leakage`, `docs`, `cleanup`. Exit code 1 on any `block` finding.

`harness/`: run these with the **project's** interpreter so they can load its data and code.

```bash
.venv/bin/python ~/.claude/ds-toolkit/harness/model_loader.py scan .              # never executes model files
.venv/bin/python ~/.claude/ds-toolkit/harness/model_loader.py inspect models/churn.txt
.venv/bin/python ~/.claude/ds-toolkit/harness/model_diagnostics.py --data eval.parquet --target y --pred p \
    --segments region plan --time scored_at --id customer_id --model models/churn.txt
.venv/bin/python ~/.claude/ds-toolkit/harness/leakage_check.py --train train.parquet --test test.parquet \
    --target y --id customer_id --time event_ts --text notes
.venv/bin/python ~/.claude/ds-toolkit/harness/llm_eval.py run --cases evals/golden.jsonl \
    --target evals.eval_targets:extract --baseline .dsguard/evals/last.json
.venv/bin/python ~/.claude/ds-toolkit/harness/drift_monitor.py --config monitor.toml
python3 ~/.claude/ds-toolkit/harness/facts.py collect .dsguard/diagnostics/*/diagnostics.json
python3 ~/.claude/ds-toolkit/harness/facts.py verify .dsguard/summaries/q3-review.md
```

Model access (`model_loader.py`, also used by `model_diagnostics.py` and the agents):
- **LightGBM text and XGBoost json/ubj**: load directly.
- **Pickles** (sklearn/joblib, whole pickled torch modules): need `--trust-pickle`, because loading runs code from the file.
- **PyTorch state_dicts and Lightning checkpoints**: `--model-class module:Class --model-init '{...}' --output logits|proba|value`.
- **Models defined in code**: `--factory module:function`, which returns a fitted model.

Each script writes JSON (for agents and CI) and Markdown (for you) under `.dsguard/`, which
ignores itself in git.

## Running without you

```bash
# headless pre-merge from a script or pre-push hook
cd ~/work/churn-model && claude -p "/ds-premerge origin/main" --output-format text > premerge.md

# CI without Claude
~/.claude/ds-toolkit/bin/dsguard run security --format json

# weekly drift check per deployed model (crontab -e)
0 7 * * 1  $HOME/.claude/ds-toolkit/bin/ds-monitor $HOME/work/churn-model
```

`ds-monitor` exits 0 on OK/WATCH and 1 on ALERT. On ALERT it runs `claude -p` with read-only
tools and saves a triage note under `.dsguard/drift/`.

## Configuration

`~/.claude/dsguard.toml` is your policy file (copied from `dsguard.example.toml` at install).
Projects can override it with `./dsguard.toml` or `[tool.dsguard]` in `pyproject.toml`. Key
sections:
- **`[llm_gateway]`**: mode, gateway hosts, exempt paths. Filled in by `/ds-setup`.
- **`[sql]`**: `dialect = "tsql"`, `partitioned_tables`.
- **`[licenses]`**: `block_families`, `exceptions`.

Thresholds for code, docs and cleanup have sensible defaults. Every option is listed in
`dsguard.example.toml`; the defaults are in `dsguard/core.py`.

## Layout

```
claude_ds_agents/
  README.md  ARCHITECTURE.md  install.py  dsguard.example.toml
  agents/      11 subagent definitions
  commands/    ds-setup, ds-premerge, ds-audit
  skills/      ds-start-day, ds-end-day
  dsguard/     deterministic engine (core, cli, 11 checks, onboarding, daybook)
  harness/     model_loader, model_diagnostics, leakage_check, llm_eval, drift_monitor, facts, common
  hooks/       pretooluse_guard.py (edit-time guard)
  git-hooks/   pre-commit
  bin/         dsguard launcher, ds-monitor scheduler wrapper
  tests/       65 tests
```

## Tests

```bash
python -m pytest -q claude_ds_agents/tests
```

## Next steps (recorded, not built)

| Agent | Deterministic part to build first | Agent judgement |
|---|---|---|
| `stats-reviewer` (A/B tests) | Sample-ratio-mismatch test, power/MDE calculator, Holm/BH correction, peeking check, CUPED helper, delta-method CIs for ratio metrics | Metric fits the decision, novelty effects, interference, practical vs statistical significance |
| `team-pr-reviewer` | dsguard on the PR diff, the team standards file as a pass/fail checklist, the author's recurring review themes | Coaching feedback; what to escalate to you |

Smaller follow-ups:
- A weekly model-health note: feed drift results to `stakeholder-writer`.
- Write model-diagnostics importance where `monitor.toml` reads it.
- Run the agents end to end in a live session and tune their prompts.
