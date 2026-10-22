# Code examples

Working examples and personal tooling for data-science and LLM work.

| Directory | What it is |
|---|---|
| [`claude_ds_agents/`](claude_ds_agents/README.md) | **DS agent toolkit for Claude Code**: 11 specialist agents (security, code, SQL, leakage, model diagnostics, LLM evals, drift, docs, cleanup, refactoring, stakeholder summaries), 3 commands, and a deterministic engine (`dsguard`, harness scripts, hooks) that decides everything a script can decide. See [ARCHITECTURE.md](claude_ds_agents/ARCHITECTURE.md). |
| [`llm_extraction_vs_vector_search/`](llm_extraction_vs_vector_search/README.md) | Runnable comparison of LLM fact extraction vs vector search on a corpus built to trigger vector search's failure modes (negation, near-identical IDs, section context), plus a hybrid pipeline. |

`dsguard.toml` at the root configures the toolkit's checks for this repository. It exempts
the toolkit's own rule definitions and the demo that calls the Anthropic API directly by design.

## Quick start

```bash
python3 claude_ds_agents/install.py                 # install the agent toolkit into ~/.claude
python -m pytest -q claude_ds_agents/tests          # toolkit tests
cd llm_extraction_vs_vector_search && python demo.py   # offline demo
```
