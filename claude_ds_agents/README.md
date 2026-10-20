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
| **Licenses** | Package licenses from installed metadata, classified allow/review/deny (dual licenses handled); HF model and dataset IDs extracted | Model/dataset card terms (non-commercial, acceptable-use) |
| **Vulnerabilities** | pip-audit (reports **skipped**, never "pass", if missing) | none |
| **SQL** | Comment- and string-aware rules: `SELECT *`, `NOT IN (SELECT)`, `= NULL`, a WHERE filter that turns a LEFT JOIN into an INNER JOIN, non-sargable filters, ORDER BY in subqueries, LIMIT without ORDER BY, `BETWEEN` dates on timestamps, missing partition filters, interpolated SQL; sqlfluff if installed | Grain, fan-out, NULL semantics, time zones, point-in-time leakage, incremental logic, cost |
| **Code quality** | ruff, complexity, nesting, length, args, duplicate functions (rename-insensitive), `iterrows`, concat in a loop, chained assignment, `merge` without `validate=`, unseeded splits, swallowed exceptions | Silent pandas errors, leakage, reproducibility, LLM call robustness, performance |
| **Refactoring** | Same metrics and duplicate fingerprints as evidence | What to extract, in what order, at what risk |
| **Docs** | Docstring coverage, **stale docstring params**, missing README, notebooks without narrative. **`verify-docs-only` proves an edit changed only docstrings and comments** (AST diff). | Writing the docs |
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

Commands: `/ds-premerge [base]` (gate on the diff) and `/ds-audit [path]` (whole project).
Call any agent directly too: "use the sql-reviewer on queries/".

## Install

```bash
python3 claude_ds_agents/install.py --dry-run   # review first
python3 claude_ds_agents/install.py             # symlinks + hook in ~/.claude/settings.json (backed up)
cp claude_ds_agents/dsguard.example.toml ~/.claude/dsguard.toml   # then set your gateway
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

Reports land in `.dsguard/reports/` (git-ignored automatically).

## Tests

```bash
python -m pytest -q claude_ds_agents/tests
```

## Roadmap: next agents (each needs its own deterministic harness)

| Agent | Deterministic part to build first |
|---|---|
| `model-diagnostics` | A script that loads a model and eval set and computes metrics by segment, calibration, residuals vs each feature, permutation/SHAP importance and learning curves, writing JSON and plots. The agent interprets them. |
| `leakage-auditor` | Train/test overlap and near-duplicate detection, features with suspiciously high single-feature AUC, timestamp ordering checks, fit-before-split AST rule |
| `llm-eval` | Golden-set runner through the gateway: exact/structured metrics, judge scores with agreement checks, cost/latency/token report, regression diff vs last run |
| `stats-reviewer` | Power and sample-size calculator, multiple-comparison correction, SRM check for A/B tests |
| `team-pr-reviewer` | dsguard on the PR diff plus your team standards file; the agent writes coaching feedback |
