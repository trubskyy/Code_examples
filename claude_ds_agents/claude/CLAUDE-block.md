## DS toolkit (applies to every project)

A personal data-science toolkit is installed at `~/.claude/ds-toolkit`. Use it in whatever
project is open; nothing about it is specific to one repository.

- **Deterministic first.** `~/.claude/ds-toolkit/bin/dsguard` decides what scripts can decide;
  agents judge the rest. Never overrule a dsguard `block`. Report checks that were skipped
  as not covered, never as passed.
- **Before committing**, run `dsguard run commit --staged` unless the git hook already does.
  **Before a PR**, offer `/ds-premerge`.
- **LLM calls go through the company gateway** configured in `~/.claude/dsguard.toml`. Never
  add a direct provider SDK or endpoint.
- **Pick the specialist agent**:
  - security-auditor: leaks, gateway, licenses
  - code-reviewer: DS bugs
  - sql-reviewer: SQL Server first
  - leakage-auditor: before trusting any metric
  - model-diagnostics: after training
  - llm-evaluator: when prompts or models change
  - drift-monitor: deployed models
  - refactor-advisor, documenter, cleanup-agent
  - stakeholder-writer: summaries where every number is verified
- **Daily rhythm**: `/ds-start-day` and `/ds-end-day`. Each project has a journal (story +
  daily plans). `dsguard day status` says where it is.
- **Project memory**: when starting work in a project, read the last entry of its story
  (path from `dsguard day status`) for context, if one exists.
- Per-project settings live in the project's `dsguard.toml`; reports and scratch go in its
  self-ignoring `.dsguard/` folder.
