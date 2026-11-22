# Claude Code agents for data-science work

## What it is

A personal automation toolkit for a data-science lead who works on tabular, text and LLM
projects. It is **one set of skills, agents and guardrails for all your projects**: install it
once on a machine and it's there in every repository you open in Claude Code. No per-project
setup is needed.

| Installed once, in `~/.claude` | What it gives every project |
|---|---|
| Agents, commands, skills | The same specialists, `/ds-premerge`, `/ds-audit`, `/ds-review-loop`, `/ds-start-day`, `/ds-end-day` everywhere |
| Managed block in `~/.claude/CLAUDE.md` | Claude knows, in any project, that the toolkit exists and when to use which part |
| PreToolUse hook | Gateway bypasses and secrets blocked before Claude writes them, in any project |
| Global git hooks (opt-in `--git-hooks`) | The commit gate in every repo on the machine; each repo's own hooks still run |
| `~/.claude/dsguard.toml` | Your policy (gateway, SQL dialect, licenses, daily rhythm) for all projects |
| `~/.claude/journal/projects.json` | One list of every project the toolkit has touched: last check verdict, journal, open plan items (`dsguard projects`) |

Each project only gets what's specific to it:
- an optional `dsguard.toml` to override settings;
- a self-ignoring `.dsguard/` folder for reports and scratch;
- its journal: story and daily plans, in the repo or privately in `~/.claude/journal/`.

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
| **Review** | A stronger model reviews every push in a loop with the author model until the findings are resolved (`/ds-review-loop`). Finds bugs and inefficiencies in DS code and notebooks; correctness and cost problems in SQL (SQL Server by default); leakage in code and data; ranked refactoring proposals. | `code-reviewer`, `sql-reviewer`, `leakage-auditor`, `refactor-advisor` |
| **Analyze** | Loads models from sklearn, LightGBM, XGBoost or PyTorch (in code or as binaries) and diagnoses them: calibration, residual structure by feature, weak segments, drift over time, importance. Runs LLM evaluations through the gateway with regression diffs. | `model-diagnostics`, `llm-evaluator`, `harness/` |
| **Monitor** | Scheduled drift checks on deployed models; Claude is woken only on an ALERT, read-only, to triage it. | `drift-monitor`, `bin/ds-monitor` |
| **Communicate** | Writes executive, product or peer summaries in which every number is cited and machine-verified against the source reports. | `stakeholder-writer`, `harness/facts.py` |
| **Maintain** | Writes documentation, proven not to change code; proposes repository cleanup and applies it reversibly after your approval. | `documenter`, `cleanup-agent` |
| **Daily rhythm** (every project) | Start of day: what's left, how much time you have, a plan split into *needs you* / *Claude in session* / *unattended*. End of day: commit, update the Markdown files, add to the project story, draft tomorrow's plan, commit. Fridays add a comment pass over the week's changed files. | `ds-start-day`, `ds-end-day`, `dsguard day` |

## Status

| | |
|---|---|
| **Built** | 12 agents, 4 commands, 2 skills (start/end of day), 11 `dsguard` checks plus 6 tools (including the review-loop referee), 6 harness scripts, the edit-time hook, the pre-commit hook, the scheduler wrapper, the installer |
| **Verified** | 94 tests. The 3 model-loader framework tests pass with sklearn, LightGBM and XGBoost installed and skip without them; PyTorch skips (unverified). CLI runs end to end on synthetic data; installer install/uninstall (CLAUDE.md block, global git hooks) in a throwaway home; the commit dispatcher in throwaway repos; the edit hook against sample tool calls. |
| **Not yet verified** | PyTorch model loading (implemented; PyTorch downloads were blocked in the build environment). Agents, commands and skills, including a full `/ds-review-loop` with a live reviewer model, haven't been run inside a live Claude Code session. Windows: never tested (see Next steps). |
| **Waiting on the work machine** | Gateway adoption and SQL Server table list, both via `/ds-setup`. Which reviewer model the gateway offers (Fable, otherwise Opus). License policy (report-only until you set `block_families`). |
| **Next** | Windows-ready installer, a first live review loop, then `stats-reviewer` (A/B tests) and `team-pr-reviewer` (see the end of this file) |

## Quick start

```bash
python3 claude_ds_agents/install.py --dry-run   # show what would change
python3 claude_ds_agents/install.py --git-hooks # into ~/.claude for every project; --git-hooks gates commits in every repo
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

There is no per-project setup. Open Claude Code in any project and everything above is
available.

The commit gate:
- **Without `--git-hooks`**: link it into chosen repos instead:
  `ln -s ~/.claude/ds-toolkit/git-hooks/pre-commit .git/hooks/pre-commit`
- **Opt out**: per repo with `git config ds.guard false`, or for one commit with
  `DSGUARD_SKIP=1 git commit …`.
- **Existing setting**: if `core.hooksPath` is already set globally (e.g. by company tooling),
  the installer leaves it alone and says so.

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
| `pr-reviewer` | nothing | The stronger-model reviewer in `/ds-review-loop`: strict JSON findings, verifies fixes, rules on disputes |

"Scratch" means helper scripts and extracts go under `.dsguard/scratch/`, never into project code.

## Daily skills

| Skill | What it does |
|---|---|
| `/ds-start-day` | Reads yesterday's plan, carry-overs, TODOs, next steps in the docs, uncommitted work and open PRs. Asks **how much time you have today**, then plans to fit it: estimates, a 20 % buffer, your review time for Claude's work. Splits the plan into **Needs you**, **Claude, this session** and **Unattended**, saves it to `docs/journal/plans/<today>.md`, and offers to start the work that doesn't need you. |
| `/ds-end-day` | Commits the remaining work (one confirmation, gated by `dsguard`). Updates the Markdown files that mention changed code and fixes broken links. Ticks today's plan, appends today's entry to **`docs/journal/STORY.md`**, drafts `plans/<next workday>.md`, and commits. Pushes only if `[daybook] push = true`. **On Fridays** it first runs a comment pass over every code file changed since the last review, proven behavior-neutral by `verify-docs-only`. |

Both skills are installed once in `~/.claude` and **work in every project you open**: any
repository, any language, with or without git. The current folder decides the project.
- **Each project has its own journal**, a continuous `STORY.md` (one entry per working day plus
  a weekly "Story so far") and `plans/` (one plan per day). The first time you use the skills in
  a project, they ask where the journal should live, and remember the answer:
  - **repo**: `<project>/docs/journal/`, committed with the project;
  - **home**: `~/.claude/journal/<project>-<id>/`, private, committed to your own journal
    repository. Better for shared work repos.
- **All projects in one view.** Every project is registered in `~/.claude/journal/projects.json`.
  `dsguard day projects` lists them with their last story entry and open plan items. Starting
  the day outside a project shows this list and asks which one to work on.
- **Tests and git.** The project's test command is detected (pytest, `npm test`, `make test`,
  R testthat) or set with `[daybook] test_command`. Projects without git still work: changes are
  found by file time, and the work commit is skipped.

`dsguard day status` shows the facts both skills start from; settings are under `[daybook]` in
`dsguard.toml`.

## Commands

| Command | What it does |
|---|---|
| `/ds-setup [gateway path]` | One-time adoption: install check → draft gateway config from the prewritten client (`dsguard suggest-gateway`) → validate it on a real project → SQL Server partitioned/large tables via read-only `sys.*` queries → license split → model scan → pre-commit links |
| `/ds-premerge [base]` | `dsguard` on the diff, then the relevant reviewers in parallel, merged into one report: BLOCKED / FIX BEFORE MERGE / READY |
| `/ds-audit [path]` | Whole-project review by all reviewers; cleanup in plan-only mode; suggests next agents |
| `/ds-review-loop [base]` | Author/reviewer loop before a push: a stronger model reviews the branch, this session fixes or disputes every finding, and the `dsguard loop` referee loops until DONE or escalates to you; then pushes and updates the PR |

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
~/.claude/ds-toolkit/bin/dsguard projects                       # every project: last verdict, journal, open items
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
  agents/      12 subagent definitions (11 specialists + pr-reviewer)
  commands/    ds-setup, ds-premerge, ds-audit, ds-review-loop
  skills/      ds-start-day, ds-end-day
  dsguard/     deterministic engine (core, cli, 11 checks, onboarding, daybook, reviewloop referee)
  harness/     model_loader, model_diagnostics, leakage_check, llm_eval, drift_monitor, facts, common
  hooks/       pretooluse_guard.py (edit-time guard)
  claude/      CLAUDE-block.md (managed block for ~/.claude/CLAUDE.md)
  git-hooks/   pre-commit (per repo), global/ (machine-wide dispatcher + stubs for every hook)
  bin/         dsguard launcher, ds-monitor scheduler wrapper
  tests/       94 tests
```

## Tests

```bash
python -m pytest -q claude_ds_agents/tests
```

## Review loop: a stronger model reviews every push

`/ds-review-loop` turns "write, push, hope" into a loop:
- **Author**: the model your session runs on. Sonnet 5.5 is a good author.
- **Reviewer**: a fresh `pr-reviewer` subagent on a stronger model (Claude Fable by default),
  which reviews the branch's diff against its base.
- **Referee**: `dsguard loop`, a script, decides when it's over.

```
gate (dsguard + tests green) ─▶ reviewer (fresh context: diff + earlier findings + author's replies)
      ▲                              │ strict JSON findings
      │                              ▼
author fixes / disputes ◀── referee: CONTINUE │ DONE ─▶ push + PR comment │ ESCALATE ─▶ you decide
```

The referee enforces what neither model should decide:
- **Gate first.** Every round starts with dsguard and the tests green, so no review is spent on
  broken code.
- **Findings must be concrete.** Every blocker or major needs a failure scenario. The reviewer
  must resolve, re-report or rule on every earlier finding and every dispute.
- **Every finding gets an answer.** The author must mark each blocking finding fixed, disputed
  (with a reason) or deferred.
- **Findings are tracked across rounds**: by id, or by a line-insensitive fingerprint when the
  reviewer re-words them.
- **DONE** when no blocker or major is open and the reviewer approves.
- **ESCALATE to you** when any of these happens:
  - the round limit is reached;
  - there's no progress (the same blocking set as the previous round);
  - a finding is fixed and comes back twice;
  - a dispute is rejected twice;
  - fixes keep adding problems;
  - anything is deferred.

The reviewer never sees the author's reasoning, only the code and the replies.

Because the loop runs inside your Claude Code session, every call goes through whatever LLM
gateway Claude Code uses at work. Settings are under `[review_loop]` in `dsguard.toml`:
- `reviewer_model`: set it to `opus` if your gateway doesn't offer Fable;
- `max_rounds`, `block_on`, `fix_minor`, `post_to_pr`;
- `enforce_on_push`: with the global git hooks, the pre-push hook then blocks any commit the
  loop hasn't approved, so every push goes through review. `DSGUARD_SKIP=1` bypasses it once.

## Next steps (recorded, not built)

| Agent | Deterministic part to build first | Agent judgement |
|---|---|---|
| `stats-reviewer` (A/B tests) | Sample-ratio-mismatch test, power/MDE calculator, Holm/BH correction, peeking check, CUPED helper, delta-method CIs for ratio metrics | Metric fits the decision, novelty effects, interference, practical vs statistical significance |
| `team-pr-reviewer` | dsguard on the PR diff, the team standards file as a pass/fail checklist, the author's recurring review themes. It can reuse the `dsguard loop` finding format and tracking | Coaching feedback for a person (unlike `pr-reviewer`, which reviews a model's work); what to escalate to you |

Windows readiness (asked for; not built yet):
- The installer finds a working Python (`python` / `py`) and writes it into the hook and
  launchers instead of `python3`.
- It copies files when links aren't allowed.
- `.cmd` launchers for `dsguard` and `ds-monitor`.
- A Task Scheduler command for drift checks.

Smaller follow-ups:
- A weekly model-health note: feed drift results to `stakeholder-writer`.
- Write model-diagnostics importance where `monitor.toml` reads it.
- Run the agents end to end in a live session and tune their prompts, starting with the
  reviewer prompt on a real `/ds-review-loop`.
