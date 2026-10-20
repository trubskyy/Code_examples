---
description: One-time adoption on a new (work) machine. Checks the install, adopts the prewritten LLM gateway, configures SQL Server tables, license reporting and model discovery, and writes ~/.claude/dsguard.toml.
argument-hint: "[path to the gateway client code, optional]"
allowed-tools: Bash(~/.claude/ds-toolkit/bin/dsguard:*), Bash(python3:*), Bash(git:*), Bash(which:*), Bash(ls:*), Read, Grep, Glob, Edit, Write
---

Set up the DS agent toolkit on this machine. Work through the steps in order, show the user
what you found at each step, and **ask before writing** `~/.claude/dsguard.toml`. Don't send
anything off the machine.

## 1. Install check

- `~/.claude/ds-toolkit` resolves; agents are in `~/.claude/agents/`; the PreToolUse hook is in `~/.claude/settings.json`. If not, tell the user to run `python3 <repo>/claude_ds_agents/install.py`.
- `python3 --version` is 3.11+.
- Optional tools: `which gitleaks pip-audit bandit sqlfluff`. List the missing ones and what they add, but don't install anything yourself.
- `~/.claude/dsguard.toml` exists (the installer copies the template there). Read it.

## 2. Adopt the LLM gateway (most important)

The gateway client is prewritten; find it. Use `$ARGUMENTS` if given; otherwise ask where it
lives (an internal package, a shared repo, a snippet in the wiki). If it's an installed
package, locate its source with
`python3 -c "import <pkg>, os; print(os.path.dirname(<pkg>.__file__))"`.

```bash
~/.claude/ds-toolkit/bin/dsguard suggest-gateway <path>
```

Explain the proposed mode in one line each:
- **ban**: projects import the company package; provider SDKs (`openai`, `anthropic`, …) are blocked everywhere else.
- **require_gateway_base_url**: projects use the provider SDK directly, but every client must point at the gateway host.

Ask the user to confirm the mode, the gateway hostname(s), and any approved exceptions. Then
write the `[llm_gateway]` section into `~/.claude/dsguard.toml`.

Validate on one real project the user names:
- `~/.claude/ds-toolkit/bin/dsguard run gateway --root <project>`: walk through the findings and fix false positives with `exempt_paths`, not by weakening rules.
- Hook dry run (should print a deny):
  `echo '{"tool_name":"Write","cwd":"<project>","tool_input":{"file_path":"<project>/t.py","content":"import openai"}}' | python3 ~/.claude/ds-toolkit/hooks/pretooluse_guard.py`
  In require mode, use `content` = `from openai import OpenAI\nc = OpenAI(base_url='https://api.openai.com/v1')` instead.

## 3. SQL Server

Confirm `[sql] dialect = "tsql"`. Large or partitioned tables can't be guessed, so ask the user.
If they approve a **read-only** metadata query (via `sqlcmd` or their usual client, which they
run or let you run), these list candidates:

```sql
-- partitioned tables and their partition column
SELECT s.name + '.' + t.name AS table_name, c.name AS partition_column
FROM sys.tables t JOIN sys.schemas s ON s.schema_id = t.schema_id
JOIN sys.indexes i ON i.object_id = t.object_id AND i.index_id IN (0, 1)
JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id AND ic.partition_ordinal > 0
JOIN sys.columns c ON c.object_id = t.object_id AND c.column_id = ic.column_id;

-- largest tables by rows (filter candidates: usually a date column)
SELECT TOP 20 s.name + '.' + t.name AS table_name, SUM(p.rows) AS rows
FROM sys.tables t JOIN sys.schemas s ON s.schema_id = t.schema_id
JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
GROUP BY s.name, t.name ORDER BY rows DESC;
```

Write the agreed `table = "filter_column"` pairs under `[sql.partitioned_tables]`. Use the
names exactly as queries write them (e.g. `dbo.FactSales`).

## 4. Licenses

Run `~/.claude/ds-toolkit/bin/dsguard run licenses --root <project>` with that project's venv
active, or set `[licenses] python`. Show the split: *always free* vs *potential issues*, with
the plain-English obligation for each family. Leave `block_families` empty unless the user has
a company policy. Record any packages legal has approved under `[licenses.exceptions]`, with
the reason.

## 5. Models

`python3 ~/.claude/ds-toolkit/harness/model_loader.py scan <project>` lists model binaries by
framework without loading them. Point out pickles (`--trust-pickle` needed, only for the
project's own artifacts) and torch state_dicts (need `--model-class` from the project code).
For models defined in code, note the `--factory module:function` route.

## 6. Per-repository commit gate

For each repo the user names: `ln -s ~/.claude/ds-toolkit/git-hooks/pre-commit <repo>/.git/hooks/pre-commit`
(ask first; if a pre-commit hook already exists, show it and don't overwrite it).

## Finish

Print the final `~/.claude/dsguard.toml` and a short checklist: done, skipped, and to do later
(e.g. install gitleaks, set the license policy once known).
