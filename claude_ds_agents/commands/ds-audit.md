---
description: Full project audit with every reviewer agent. Use when inheriting a project, before a handover, or periodically.
argument-hint: "[path to limit the audit, optional]"
allowed-tools: Bash(~/.claude/ds-toolkit/bin/dsguard:*), Bash(git status:*), Bash(git ls-files:*), Read, Grep, Glob, Write, Agent, Task
---

Audit this whole repository (or only `$ARGUMENTS` if a path is given).

1. Run the deterministic baseline:
   ```bash
   ~/.claude/ds-toolkit/bin/dsguard run all --format text --limit 30 [--paths $ARGUMENTS]
   ```
2. Dispatch **in parallel** with full-project scope: `security-auditor`, `code-reviewer`,
   `refactor-advisor`, plus `sql-reviewer` if the project has SQL. Also run `cleanup-agent`
   with the instruction: **plan only, do not apply**.
3. Don't run `documenter`; it edits files. Instead, report the docs coverage numbers from
   dsguard and offer to run it.
4. Write `.dsguard/reports/audit-<YYYYMMDD-HHMM>.md` and print it:

```
# Project audit: <repo name>
## Health summary
| Area | Status | Key issue |
(security, LLM gateway, data exposure, licenses, code correctness, SQL, docs, hygiene, structure)
## Must fix
## Top refactors (from refactor-advisor, top 3)
## Cleanup plan (from cleanup-agent; awaiting approval at .dsguard/cleanup-plan.json)
## Not covered
```

Apply the same merge rules as `/ds-premerge`: never soften dsguard blocks, de-duplicate rows,
and don't invent findings.
