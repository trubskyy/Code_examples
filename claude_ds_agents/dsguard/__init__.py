"""dsguard: deterministic checks that back the Claude Code data-science agents.

Every check here is a script decision (regex, AST, metadata, external linter),
so its verdict is repeatable and can gate a commit. Agents read the report and
handle the judgement calls the checks cannot make.
"""

__version__ = "0.1.0"
