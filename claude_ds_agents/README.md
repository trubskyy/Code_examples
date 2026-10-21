# Claude Code agents for data-science work

A personal agent toolkit for a DS lead working on tabular, text and LLM projects. It installs
into `~/.claude`, so every project on your machine gets the same agents, commands and guardrails.

The design rule: **if a script can decide it, a script decides it.** Agents only make the
judgement calls. A deterministic verdict is repeatable, can block a commit, and can't be talked
out of a finding. Agents read the script output first, then do what scripts can't.

```
            ┌───────────── your machine is the harness ─────────────┐
 edit/run → │ PreToolUse hook (deterministic, every tool call)       │ → deny / ask / allow
 commit   → │ git pre-commit → dsguard run commit --staged           │ → block
 /ds-premerge → dsguard run all --changed-since <base>  ──┐          │
            │                                             ▼          │
            │   security-auditor ║ code-reviewer ║ sql-reviewer  (parallel, read-only)
            │                                             ▼          │
            │          merged report in .dsguard/reports/            │
            └────────────────────────────────────────────────────────┘
```

## What is deterministic and what is an agent

| Concern | Deterministic: `dsguard` / hooks | Agent judgement |
|---|---|---|
| **LLM gateway** | AST: provider SDK imports, provider hostnames, client constructors without a gateway `base_url`, provider key env vars, SDKs in requirements. The **hook denies** such edits and `curl`s to providers. | Indirect bypasses (HTTP calls with URLs from config, framework wrappers), whether data sent to the LLM is cleared for it |
| **Secrets** | Token patterns (AWS, Anthropic, OpenAI, GitHub, Slack, GCP, HF, JWT, URL passwords), entropy-checked assignments, credential files, gitleaks over git history. **Hook denies** writing them. Output is always redacted. | Fallback defaults, secrets printed in notebooks, keys on the CLI |
| **Data exposure** | Committed data/model files, notebooks with saved outputs, PII regex (email, SSN, Luhn-valid cards) in outputs and data files, `0.0.0.0` binds, `share=True`, `debug=True`, CORS `*` | Whether outputs are sensitive, prompt/response logging, injection surface |
| **Code security** | `pickle`/`torch.load`/`yaml.load`, `eval`, `shell=True`, `verify=False`, `trust_remote_code=True`; bandit if installed | Exploitability in context |
| **Licenses** | Every installed package sorted into **always free** (MIT, BSD, Apache, ISC, PSF, …) or **potential issues**, with the obligation in plain English per family (weak copyleft, unknown, strong copyleft, network copyleft, restricted/non-commercial); dual licenses take the freer option; nothing blocks until you set `block_families`; HF model and dataset IDs extracted | Model/dataset card terms (non-commercial, acceptable-use) |
| **Vulnerabilities** | pip-audit (reports **skipped**, never "pass", if missing) | none |
| **SQL** (SQL Server by default) | Comment- and string-aware rules: `SELECT *`, `NOT IN (SELECT)`, `= NULL`, a WHERE filter that turns a LEFT JOIN into an INNER JOIN, non-sargable filters (incl. `CONVERT`, `ISNULL`, `DATEPART`), ORDER BY in subqueries, TOP/LIMIT without ORDER BY, `BETWEEN` dates on date-times, missing filters on large/partitioned tables, interpolated SQL. T-SQL: `GO` batches, `NOLOCK`, cursors, INT/INT division, `AVG(int)`, scalar UDFs in WHERE. sqlfluff if installed. | Grain, fan-out, NULL semantics, collation, implicit conversions, point-in-time leakage, plans, cost |
| **Code quality** | ruff, complexity, nesting, length, args, duplicate functions (rename-insensitive), `iterrows`, concat in a loop, chained assignment, `merge` without `validate=`, unseeded splits, swallowed exceptions | Silent pandas errors, leakage, reproducibility, LLM call robustness, performance |
| **Refactoring** | Same metrics and duplicate fingerprints as evidence | What to extract, in what order, at what risk |
| **Docs** | Docstring coverage, **stale docstring params**, missing README, notebooks without narrative. **`verify-docs-only` proves an edit changed only docstrings and comments** (AST diff). | Writing the docs |
| **Leakage (code)** | AST, notebook cells in run order: preprocessor/selector/imputer fitted before any split or CV, `.fit()` on test/val-named data, oversampling before the split, `shift(-n)`, backfill | What is known at prediction time; snapshot tables; as-of joins |
| **Leakage (data)** | `harness/leakage_check.py`: shared ids/groups across splits, duplicate feature rows, normalized and word-set duplicate texts, temporal overlap, feature timestamps after event time, columns too predictive alone (raw and missingness), target copies, id/row-order leakage, PSI drift | Confirming each hit is a real leak, choosing the fix |
| **Model diagnostics** | `harness/model_diagnostics.py` on any framework's predictions: AUC/AP/log-loss/Brier/ECE or RMSE/MAE/R²/bias, calibration and threshold tables, residual structure per feature (eta²), weakest segments, metric over time, single-feature strength, redundant pairs, permutation importance (optional model), worst rows, plots | Baselines, causes, which experiments to run next |
| **LLM evals** | `harness/llm_eval.py`: golden-set runner calling *your* gateway function; exact/normalized/contains/regex/JSON/field/numeric checks, by-tag pass rates, error rate, consistency over repeats, p50/p95 latency, tokens and cost, judge-vs-human kappa, response cache, pass→fail diff vs baseline (exit 1) | Building the golden set, judge rubric, diagnosing regressions, significance |
| **Drift monitoring** | `harness/drift_monitor.py`: per-feature PSI, KS, mean shift in SDs, out-of-reference-range share, unseen categories, missing-rate change; prediction drift; performance with label coverage; schema/type changes; per-period trends and volume (partial periods excluded); importance-weighted OK/WATCH/ALERT; run history; exit 1 on ALERT. `bin/ds-monitor` runs it on a schedule and wakes Claude only on ALERT, read-only. | Pipeline break vs population shift vs noise; retrain or fix upstream |
| **Stakeholder summaries** | `harness/facts.py`: builds a numbered facts ledger from any report (and before/after deltas), then **`verify` fails any sentence whose numbers are not cited or do not match the cited fact** at the written precision (%, pp, k/M handled). `render` strips citations and adds sources. | What matters to the audience, plain-language framing, caveats |
| **Cleanup** | Junk dirs and files, suspect names, empty/large/identical files, unimported modules. **Apply is a script**: moves files to `.dsguard/trash/<stamp>` (restorable) or `git rm --cached`; refuses paths outside the repo. The hook **asks** before any `rm -r`. | Verifying a module is really unused, choosing what goes into the plan |

## Agents (in `agents/`)

| Agent | Edits? | Use it for |
|---|---|---|
| `security-auditor` | no | Gateway, secrets, exposure, licenses, CVEs, prompt-injection surface |
| `sql-reviewer` | no | Query correctness first, then cost |
| `code-reviewer` | no | Bugs and inefficiencies in DS code and notebooks |
| `refactor-advisor` | no | Ranked refactor proposals with sketches |
| `documenter` | docs only | Docstrings, README, data dictionary; must pass `verify-docs-only` |
| `cleanup-agent` | plan only | Writes `.dsguard/cleanup-plan.json`, applies only after you approve |
| `leakage-auditor` | scratch only | Code + data leakage, eval contamination for LLM projects |
| `model-diagnostics` | scratch only | Builds the eval table, runs the harness, ranks next experiments |
| `llm-evaluator` | eval files | Golden sets, adapters, runs and regression diagnosis via the gateway |
| `drift-monitor` | monitor config | Sets up `monitor.toml` for a deployed model; triages ALERTs |
| `stakeholder-writer` | summaries | Exec/product/peer summaries where every number is script-verified |

"Scratch only" agents write helper scripts and data extracts under `.dsguard/scratch/`, never
into project code.

Commands: `/ds-setup` (one-time adoption on a new machine), `/ds-premerge [base]` (gate on the diff) and `/ds-audit [path]` (whole project).
Call any agent directly too: "use the sql-reviewer on queries/".

## Adopting it on the work machine

1. Clone this repo there and run `python3 claude_ds_agents/install.py`.
2. In Claude Code, run `/ds-setup [path to the gateway client]`. It:
   - reads the **prewritten gateway client** with `dsguard suggest-gateway` and drafts the
     `[llm_gateway]` section (mode, gateway host, exempt path, the key env var the gateway owns);
   - validates it on a real project, including a dry run of the edit hook;
   - lists SQL Server **partitioned/large tables** with read-only `sys.*` queries (you approve)
     for `[sql.partitioned_tables]`;
   - shows the license split for a project and records exceptions legal has approved;
   - scans for model binaries;
   - links the pre-commit gate into the repos you name.
3. Until then the gateway rules use generic defaults. Nothing about your company is hardcoded.

## Install

```bash
python3 claude_ds_agents/install.py --dry-run   # review first
python3 claude_ds_agents/install.py             # symlinks + hook in ~/.claude/settings.json (backed up)
# install.py also copies dsguard.example.toml to ~/.claude/dsguard.toml (your policy file)
pip install pip-audit bandit sqlfluff           # optional, recommended
brew install gitleaks                           # optional, recommended: scans git history
```

Per repository, add the commit gate:

```bash
ln -s ~/.claude/ds-toolkit/git-hooks/pre-commit .git/hooks/pre-commit
```

Needs Python 3.11+. dsguard itself is stdlib only. Symlinks mean a `git pull` here updates
every agent; `install.py --uninstall` reverses everything it did.

## Using your machine as the harness

Interactive: run `/ds-premerge` before you open a PR, `/ds-audit` when you inherit a project.

Headless (cron, a git `pre-push` hook, or a script looping over your repos):

```bash
cd ~/work/churn-model && claude -p "/ds-premerge origin/main" --output-format text > /tmp/premerge.md
```

The scripts run without Claude too, e.g. in CI:

```bash
~/.claude/ds-toolkit/bin/dsguard run security --format json   # exit 1 on any block finding
~/.claude/ds-toolkit/bin/dsguard run all --changed-since origin/main
~/.claude/ds-toolkit/bin/dsguard verify-docs-only
```

Scheduled drift monitoring per deployed model (after the `drift-monitor` agent writes `monitor.toml`):

```bash
# crontab -e
0 7 * * 1  $HOME/.claude/ds-toolkit/bin/ds-monitor $HOME/work/churn-model
```

The monitor itself is deterministic; on ALERT the wrapper runs `claude -p` with Read/Grep/Glob
only and saves a triage note next to the report.

Reports land in `.dsguard/reports/` (git-ignored automatically).

## Tests

```bash
python -m pytest -q claude_ds_agents/tests
```

## Harness scripts (need numpy + pandas; llm_eval and facts are stdlib only)

Run them with the **project's** interpreter so they can load its data and code:

```bash
.venv/bin/python ~/.claude/ds-toolkit/harness/model_diagnostics.py --data eval.parquet --target y --pred p \
    --segments region plan --time scored_at --id customer_id
.venv/bin/python ~/.claude/ds-toolkit/harness/leakage_check.py --train train.parquet --test test.parquet \
    --target y --id customer_id --time event_ts --text notes
.venv/bin/python ~/.claude/ds-toolkit/harness/llm_eval.py run --cases evals/golden.jsonl \
    --target evals.eval_targets:extract --baseline .dsguard/evals/last.json
.venv/bin/python ~/.claude/ds-toolkit/harness/drift_monitor.py --config monitor.toml
.venv/bin/python ~/.claude/ds-toolkit/harness/model_loader.py scan .          # find models, never executes them
.venv/bin/python ~/.claude/ds-toolkit/harness/model_loader.py inspect models/churn.txt
python3 ~/.claude/ds-toolkit/harness/facts.py collect .dsguard/diagnostics/*/diagnostics.json --out .dsguard/summaries/facts.json
python3 ~/.claude/ds-toolkit/harness/facts.py verify .dsguard/summaries/q3-review.md
```

`model_loader.py` puts sklearn, LightGBM, XGBoost and PyTorch behind one `predict()`:
- **LightGBM text and XGBoost json/ubj** load directly.
- **Pickles** (joblib/pkl, and whole pickled torch modules) need `--trust-pickle`, because unpickling runs code.
- **Torch state_dicts** need `--model-class module:Class`.
- **Models defined in code** load through `--factory module:function`.

`model_diagnostics.py` and the agents use the same flags.

Each writes JSON (for agents and CI) and Markdown (for you) under `.dsguard/`. matplotlib,
pyarrow and joblib are optional. Metrics are implemented in `harness/common.py`, so sklearn is
not required.

## Next steps (recorded, not built yet)

| Agent | Deterministic part to build first | Agent judgement |
|---|---|---|
| `stats-reviewer` (A/B tests) | Sample-ratio-mismatch chi-square test, power and minimum-detectable-effect calculator, multiple-comparison correction (Holm/BH), peeking check from the analysis timeline, CUPED variance-reduction helper, CIs for ratio metrics (delta method) | Whether the metric matches the decision, novelty effects, interference between units, practical vs statistical significance |
| `team-pr-reviewer` | dsguard on the PR diff, the team standards file as a checklist with pass/fail per item, past review comments for the author to spot recurring themes | Coaching-style feedback that teaches; what to escalate to you |

Smaller follow-ups: feed `drift_monitor` alerts into `stakeholder-writer` for a weekly model
health note, and let `model-diagnostics` write its permutation importance to the path
`monitor.toml` reads.
