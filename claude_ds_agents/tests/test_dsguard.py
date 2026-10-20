"""Offline tests for the deterministic layer. Run: python -m pytest -q claude_ds_agents/tests"""

import copy
import json
import tomllib
import subprocess
import sys
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT))

from dsguard import cleanup, code, docs, exposure, gateway, licenses, onboard, secrets, sql  # noqa: E402
from dsguard.core import DEFAULTS, build_context, parse  # noqa: E402

GW = DEFAULTS["llm_gateway"]
SQL = DEFAULTS["sql"]


def rules(hits):
    return [h[0] for h in hits]


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=True)
    return tmp_path


# --- gateway -------------------------------------------------------------------

def test_ban_mode_blocks_provider_imports_and_endpoints():
    tree = parse("import openai\nfrom langchain_openai import ChatOpenAI\nURL = 'https://api.anthropic.com/v1'\n")
    hits = gateway.scan_python_tree(tree, GW)
    assert rules(hits) == ["direct-llm-import", "direct-llm-import", "direct-llm-endpoint"]
    assert all(h[1] == "block" for h in hits)


def test_ban_mode_allows_gateway_client_and_unrelated_imports():
    tree = parse("from company_llm import Gateway\nimport pandas as pd\nimport openpyxl\n")
    assert gateway.scan_python_tree(tree, GW) == []


def test_require_base_url_mode():
    cfg = copy.deepcopy(GW) | {"mode": "require_gateway_base_url", "allowed_hosts": ["llm-gw.corp.net"]}
    tree = parse(
        "from openai import OpenAI\n"
        "ok = OpenAI(base_url='https://llm-gw.corp.net/v1')\n"
        "bad = OpenAI(base_url='https://api.openai.com/v1')\n"
        "implicit = OpenAI()\n"
        "dynamic = OpenAI(base_url=os.environ['GW'])\n"
    )
    hits = {(h[0], h[2]) for h in gateway.scan_python_tree(tree, cfg)}
    assert ("client-not-gateway", 3) in hits
    assert ("client-without-gateway", 4) in hits
    assert ("client-gateway-unverified", 5) in hits
    assert not any(line == 2 for _, line in hits)


def test_azure_wildcard_host_and_fragment_fallback():
    assert gateway.host_status("myco.openai.azure.com", GW) == "banned"
    hits = gateway.scan_python_fragment("    from anthropic import Anthropic", GW)
    assert rules(hits) == ["direct-llm-import"]


# --- secrets -------------------------------------------------------------------

def test_secret_patterns_are_redacted():
    # Fake secrets are assembled at runtime so this file does not trip secret scanners.
    text = 'key = "' + "sk-ant-" + 'api03-abcdefghijklmnopqrstuvwxyz"\nAWS = "' + "AKIA" + 'ABCDEFGHIJKLMNOP"\n'
    hits = secrets.scan_text(text)
    assert rules(hits) == ["anthropic-key", "aws-access-key"]
    assert all("abcdefghijklmnop" not in h[3] for h in hits)


def test_secret_placeholders_and_env_names_ignored():
    text = 'api_key = "YOUR_API_KEY_HERE"\npassword = "changeme-please"\nurl = "postgres://user:${PW}@db/x"\n'
    assert secrets.scan_text(text) == []
    assert rules(secrets.scan_text('db = "postgres://ana:' + 'Xk29!vq@db.corp/x"')) == ["url-credentials"]


def test_secret_files():
    assert secrets.is_secret_file("config/.env")
    assert secrets.is_secret_file("keys/gcp-service-account-prod.json")
    assert not secrets.is_secret_file(".env.example")


# --- sql -----------------------------------------------------------------------

def test_sql_rules():
    query = """
    SELECT DISTINCT * FROM orders o
    LEFT JOIN refunds r ON r.order_id = o.id
    WHERE r.status = 'done' AND o.id NOT IN (SELECT id FROM blocked)
      AND DATE(o.created_at) = '2024-01-01'
    UNION
    SELECT a, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM (SELECT a, b FROM t ORDER BY b) x
    """
    found = set(rules(sql.lint_sql(query, SQL)))
    assert {"select-star", "select-distinct", "left-join-filtered", "not-in-subquery",
            "function-on-filter-column", "union-distinct", "order-in-subquery"} <= found


def test_sql_window_order_and_null_check_not_flagged():
    query = "SELECT a, SUM(x) OVER (ORDER BY d) FROM t LEFT JOIN u ON u.id = t.id WHERE u.id IS NULL"
    assert sql.lint_sql(query, SQL) == []


def test_sql_comments_and_strings_do_not_trigger():
    query = "-- SELECT * FROM a, b\nSELECT id FROM t WHERE note = 'select * from x, y' -- = NULL\n"
    assert sql.lint_sql(query, SQL) == []


def test_sql_line_numbers_and_partition_filter():
    cfg = SQL | {"partitioned_tables": {"events": "event_date"}}
    hits = sql.lint_sql("SELECT 1;\n\nSELECT * FROM events WHERE user_id = 1", cfg)
    assert ("select-star", 2) in {(h[0], h[2]) for h in hits}
    assert "missing-partition-filter" in rules(hits)


def test_tsql_rules():
    query = """
    SELECT TOP 100 c.id, AVG(o.amount), SUM(o.refunds) / COUNT(*) AS rate
    FROM [dbo].[customers] c WITH (NOLOCK)
    LEFT JOIN [dbo].[orders] o ON o.customer_id = c.id
    WHERE o.status = 'open' AND CONVERT(date, c.created_at) = '2024-01-01' AND dbo.fnIsActive(c.id) = 1
    GROUP BY c.id
    GO
    DECLARE cur CURSOR FOR SELECT id FROM t
    """
    found = set(rules(sql.lint_sql(query, SQL)))
    assert {"nolock", "integer-division", "avg-int", "limit-without-order", "left-join-filtered",
            "function-on-filter-column", "scalar-udf-filter", "cursor"} <= found
    ansi = set(rules(sql.lint_sql(query, SQL | {"dialect": "ansi"})))
    assert "nolock" not in ansi and "avg-int" not in ansi


def test_go_separates_batches():
    hits = sql.lint_sql("SELECT a FROM t\nGO\nSELECT * FROM u\n", SQL)
    assert [(h[0], h[2]) for h in hits] == [("select-star", 2)]


def test_python_sql_extraction_flags_interpolation():
    tree = parse('q = f"SELECT * FROM {table} WHERE id = {i}"\nr = "SELECT id FROM t"\n')
    found = list(sql.python_sql(tree))
    assert [(line, interp) for _, line, interp in found] == [(1, True), (2, False)]


# --- code ----------------------------------------------------------------------

def test_pandas_pitfalls(tmp_path):
    root = make_repo(tmp_path, {"p.py": (
        "import pandas as pd\n"
        "def f(df, parts):\n"
        "    out = pd.DataFrame()\n"
        "    for p in parts:\n"
        "        out = pd.concat([out, p])\n"
        "    for _, row in df.iterrows():\n"
        "        pass\n"
        "    df['a']['b'] = 1\n"
        "    m = df.merge(df, on='k')\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception:\n"
        "        pass\n"
    )})
    ctx = build_context(root)
    ctx.config["code"]["ruff"] = False
    found = set(f.rule for f in code.run(ctx).findings)
    assert {"grow-in-loop", "row-iteration", "chained-assignment", "merge-unvalidated", "swallowed-exception"} <= found


def test_duplicate_functions_detected_despite_renames(tmp_path):
    body = "    total = 0\n    for v in {x}:\n        total += v * 2\n    total -= 1\n    total *= 3\n    return total\n"
    root = make_repo(tmp_path, {
        "a.py": "def f(xs):\n" + body.format(x="xs"),
        "b.py": "def g(values):\n" + body.format(x="values").replace("total", "acc"),
    })
    ctx = build_context(root)
    ctx.config["code"]["ruff"] = False
    assert "duplicate-function" in {f.rule for f in code.run(ctx).findings}


# --- docs ----------------------------------------------------------------------

def test_documented_params_styles():
    numpy = "Do it.\n\nParameters\n----------\nx : int\n    The x.\ny, z : float\n\nReturns\n-------\nint\n"
    google = "Do it.\n\nArgs:\n    x (int): the x.\n        more text: here\n    y: why.\n\nReturns:\n    int\n"
    assert docs.documented_params(numpy) == {"x", "y", "z"}
    assert docs.documented_params(google) == {"x", "y"}
    assert docs.documented_params(":param int x: a\n:param y: b") == {"x", "y"}
    assert docs.documented_params("No params section.") is None


def test_verify_docs_only(tmp_path):
    root = make_repo(tmp_path, {"m.py": "def f(x):\n    return x + 1\n", "README.md": "hi\n"})
    (root / "m.py").write_text('def f(x):\n    """Add one."""\n    # comment\n    return x + 1\n')
    (root / "README.md").write_text("hello\n")
    assert docs.verify_docs_only(build_context(root)).findings == []
    (root / "m.py").write_text('def f(x):\n    """Add two."""\n    return x + 2\n')
    assert [f.rule for f in docs.verify_docs_only(build_context(root)).findings] == ["behavior-change"]


# --- exposure / licenses -------------------------------------------------------

def test_pii_and_luhn():
    hits = exposure.pii_hits("a@corp.io 4539 1488 0343 6467\nb@example.com\n123-45-6789", ["example.com"])
    assert sorted(k for k, _ in hits) == ["SSN-like number", "email address", "payment card number"]
    assert not exposure.luhn("4111111111111112")


def test_notebook_outputs_flagged(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": "df.head()", "outputs": [
        {"output_type": "execute_result", "data": {"text/plain": "jane@corp.io  42"}}]}], "metadata": {}}
    root = make_repo(tmp_path, {"n.ipynb": json.dumps(nb)})
    found = {f.rule for f in exposure.run(build_context(root)).findings}
    assert found == {"notebook-outputs", "pii-in-notebook-output"}


@pytest.mark.parametrize("dist,expected", [
    ({"classifiers": ["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]}, "strong-copyleft"),
    ({"classifiers": ["License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"]}, "weak-copyleft"),
    ({"classifiers": ["License :: OSI Approved :: GNU Affero General Public License v3"]}, "network-copyleft"),
    ({"expression": "MIT OR GPL-3.0-only"}, "permissive"),
    ({"license": "BSD-3-Clause"}, "permissive"),
    ({"license": "CC-BY-NC-4.0"}, "restricted"),
    ({"license": "Mozilla Public License 2.0 (MPL 2.0)"}, "weak-copyleft"),
    ({"license": "Some custom terms"}, "unknown"),
    ({}, "unknown"),
])
def test_license_classification(dist, expected):
    assert licenses.classify(dist)[0] == expected


# --- cleanup -------------------------------------------------------------------

def test_cleanup_candidates_and_reversible_apply(tmp_path):
    root = make_repo(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "from .util import helper\n",
        "pkg/util.py": "def helper():\n    return 1\n",
        "pkg/old_stuff.py": "x = 1\n",
        "train.py": "import pkg.core\nif __name__ == '__main__':\n    pass\n",
        "debug.log": "noise\n",
        "analysis_copy.py": "y = 2\n",
    })
    found = {(f.rule, f.path) for f in cleanup.run(build_context(root)).findings}
    assert ("possibly-unused-module", "pkg/old_stuff.py") in found
    assert ("junk-file", "debug.log") in found
    assert ("suspect-name", "analysis_copy.py") in found
    assert not any(path in ("pkg/util.py", "pkg/core.py", "train.py") for rule, path in found
                   if rule == "possibly-unused-module")

    plan = root / "plan.json"
    plan.write_text(json.dumps({"items": [{"path": "pkg/old_stuff.py", "action": "delete"},
                                          {"path": "debug.log", "action": "untrack"}],
                                "gitignore": ["*.log"]}))
    cleanup.apply_plan(root, plan)
    assert not (root / "pkg/old_stuff.py").exists()
    assert (root / "debug.log").exists() and "*.log" in (root / ".gitignore").read_text()
    stamp = next((root / ".dsguard/trash").iterdir()).name
    cleanup.restore(root, stamp)
    assert (root / "pkg/old_stuff.py").read_text() == "x = 1\n"


def test_apply_refuses_paths_outside_root(tmp_path):
    root = make_repo(tmp_path, {"a.py": ""})
    plan = root / "plan.json"
    plan.write_text(json.dumps({"items": [{"path": "../etc", "action": "delete"}]}))
    with pytest.raises(SystemExit):
        cleanup.apply_plan(root, plan)


# --- gateway adoption -------------------------------------------------------------

def test_suggest_gateway_for_package(tmp_path):
    pkg = tmp_path / "corp_llm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .client import complete\n")
    (pkg / "client.py").write_text(
        "import os\nfrom openai import OpenAI\n"
        "_client = OpenAI(base_url='https://llm-gw.corp.internal/v1', api_key=os.environ['OPENAI_API_KEY'])\n"
        "def complete(prompt):\n    return _client.chat.completions.create(model='x', messages=[])\n")
    text = onboard.suggest(pkg, tmp_path)
    cfg = tomllib.loads(text)["llm_gateway"]
    assert cfg["mode"] == "ban" and cfg["allowed_hosts"] == ["llm-gw.corp.internal"]
    assert cfg["exempt_paths"] == ["corp_llm/**"]
    assert "OPENAI_API_KEY" not in cfg["provider_key_env_vars"]
    assert "`corp_llm`" in text


def test_suggest_gateway_for_snippet(tmp_path):
    snippet = tmp_path / "gateway_snippet.py"
    snippet.write_text("from openai import AzureOpenAI\nc = AzureOpenAI(azure_endpoint=os.getenv('GW_URL'))\n")
    cfg = tomllib.loads(onboard.suggest(snippet, tmp_path))["llm_gateway"]
    assert cfg["mode"] == "require_gateway_base_url" and cfg["allowed_hosts"] == []
