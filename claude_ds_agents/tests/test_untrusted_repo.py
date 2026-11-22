"""Checking a repository must not run its code or write outside it, whatever the repository contains."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parent.parent
DSGUARD = TOOLKIT / "bin" / "dsguard"


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "untrusted"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def dsguard(root, *args, **env):
    return subprocess.run([str(DSGUARD), "run", *args], cwd=root, env=os.environ | env, capture_output=True, text=True)


def licenses(root):
    proc = dsguard(root, "licenses", "--no-save", "--format", "json")
    return json.loads(proc.stdout)["results"][0]


def fake_dist(site, name="evil-gpl", license_="GPL-3.0"):
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\nLicense: {license_}\n")


@pytest.mark.parametrize("venv", [".venv", "venv"])
def test_venv_is_read_not_executed(repo, tmp_path, venv):
    script(repo / venv / "bin" / "python", f'touch "{tmp_path}/ran"\n')
    site = repo / venv / "lib" / "python3.12" / "site-packages"
    fake_dist(site)
    (site / "evil.pth").write_text(f'import os; open({str(tmp_path / "ran")!r}, "w").close()\n')
    result = licenses(repo)
    assert not (tmp_path / "ran").exists()
    assert [f["path"] for f in result["findings"]] == ["evil-gpl==1.0"]


@pytest.mark.parametrize("name,text", [("dsguard.toml", "[licenses]\n"), ("pyproject.toml", "[tool.dsguard.licenses]\n")])
def test_project_cannot_name_the_interpreter(repo, tmp_path, name, text):
    script(repo / "tool", f'touch "{tmp_path}/ran"\n')
    (repo / name).write_text(text + f'python = "{repo}/tool"\n')
    licenses(repo)
    assert not (tmp_path / "ran").exists()


@pytest.mark.parametrize("module", ["json", "importlib"])
def test_repo_modules_do_not_shadow_the_stdlib(repo, tmp_path, module):
    (repo / f"{module}.py").write_text(f'open({str(tmp_path / "ran")!r}, "w").close()\n')
    licenses(repo)
    assert not (tmp_path / "ran").exists()


def test_user_can_still_name_an_interpreter(repo, tmp_path, isolated_home):
    script(tmp_path / "python", f'touch "{tmp_path}/ran"\nexec "{sys.executable}" "$@"\n')
    (isolated_home / ".claude" / "dsguard.toml").write_text(f'[licenses]\npython = "{tmp_path}/python"\n')
    assert licenses(repo)["status"] != "error"
    assert (tmp_path / "ran").exists()


def test_relative_interpreter_in_user_config_is_not_resolved_in_the_repo(repo, tmp_path, isolated_home):
    script(repo / "tool", f'touch "{tmp_path}/ran"\n')
    (isolated_home / ".claude" / "dsguard.toml").write_text('[licenses]\npython = "./tool"\n')
    assert licenses(repo)["status"] == "error"
    assert not (tmp_path / "ran").exists()


@pytest.mark.parametrize("link", [".dsguard", ".dsguard/reports", ".dsguard/reports/latest.json", ".dsguard/.gitignore"])
def test_state_is_never_written_through_a_symlink(repo, tmp_path, link):
    outside = tmp_path / "outside"
    if link in (".dsguard", ".dsguard/reports"):
        outside.mkdir()
    (repo / link).parent.mkdir(parents=True, exist_ok=True)
    (repo / link).symlink_to(outside)
    proc = dsguard(repo, "commit")
    assert "symlink" in proc.stdout + proc.stderr and proc.returncode == 0
    assert not any(outside.iterdir()) if outside.is_dir() else not outside.exists()


def test_symlinked_state_does_not_release_a_push(repo, tmp_path, isolated_home):
    (isolated_home / ".claude" / "dsguard.toml").write_text("[review_loop]\nenforce_on_push = true\n")
    (repo / ".dsguard").symlink_to(tmp_path)
    proc = subprocess.run([str(DSGUARD), "loop", "check-push", "--refs-stdin"], cwd=repo, capture_output=True, text=True,
                          input=f"refs/heads/x {'a' * 40} refs/heads/x {'0' * 40}\n")
    assert proc.returncode == 10


@pytest.mark.parametrize("name,text", [("dsguard.toml", "[daybook]\n"), ("pyproject.toml", "[tool.dsguard.daybook]\n")])
def test_project_cannot_move_the_registry(repo, tmp_path, name, text):
    (repo / name).write_text(text + f'home_dir = "{tmp_path}/outside"\n')
    dsguard(repo, "commit", "--no-save")
    assert not (tmp_path / "outside").exists()


def test_sqlfluff_sees_no_project_config(repo, tmp_path):
    script(tmp_path / "bin" / "sqlfluff", f"""echo "$@" > "{tmp_path}/args"
for a; do if [ -f "$a" ]; then cat "$a" >> "{tmp_path}/seen"; fi; done
echo '[]'
""")
    (repo / "q.sql").write_text("-- sqlfluff:templater:jinja:library_path:lib\nSELECT 1\n")
    dsguard(repo, "sql", "--no-save", PATH=f"{tmp_path}/bin:{os.environ['PATH']}")
    args, seen = (tmp_path / "args").read_text(), (tmp_path / "seen").read_text()
    assert "--ignore-local-config" in args and "--templater raw" in args and str(repo) not in args
    assert "sqlfluff:" not in seen and seen.count("\n") == 2


def test_sqlfluff_findings_keep_their_paths(repo, tmp_path):
    script(tmp_path / "bin" / "sqlfluff", """for a; do last="$a"; done
printf '[{"filepath":"%s","violations":[{"code":"AM04","description":"d","start_line_no":2}]}]' "$last"
""")
    (repo / "q.sql").write_text("-- c\nSELECT 1\n")
    proc = dsguard(repo, "sql", "--no-save", "--format", "json", PATH=f"{tmp_path}/bin:{os.environ['PATH']}")
    found = [f for f in json.loads(proc.stdout)["results"][0]["findings"] if f["rule"] == "sqlfluff:AM04"]
    assert [(f["path"], f["line"]) for f in found] == [("q.sql", 2)]


def test_pip_audit_gets_only_pins_and_never_pip(repo, tmp_path):
    script(tmp_path / "bin" / "pip-audit", f"""echo "$@" > "{tmp_path}/args"
for a; do if [ -f "$a" ]; then cat "$a" > "{tmp_path}/seen"; fi; done
echo '{{"dependencies": []}}'
""")
    (repo / "requirements.txt").write_text(
        "-e ./pkg\n--index-url https://example.com/simple\n./pkg\nevil @ file:///tmp/pkg\n"
        "pandas>=2\nnumpy==2.1.0 ; python_version > '3.9'\nrequests[socks] == 2.32.3 # why\nwild==1.*\n")
    proc = dsguard(repo, "vulns", "--no-save", PATH=f"{tmp_path}/bin:{os.environ['PATH']}")
    args = (tmp_path / "args").read_text()
    assert "--no-deps" in args and "--disable-pip" in args
    assert (tmp_path / "seen").read_text().split() == ["numpy==2.1.0", "requests==2.32.3"]
    assert "NOT checked" in proc.stdout


def test_pip_audit_skips_when_nothing_is_pinned(repo, tmp_path):
    script(tmp_path / "bin" / "pip-audit", f'touch "{tmp_path}/ran"\n')
    (repo / "requirements.txt").write_text("-e .\npandas>=2\n")
    proc = dsguard(repo, "vulns", "--no-save", "--format", "json", PATH=f"{tmp_path}/bin:{os.environ['PATH']}")
    assert json.loads(proc.stdout)["results"][0]["status"] == "skipped" and not (tmp_path / "ran").exists()


def test_agents_use_isolated_python_for_snippets():
    for path in (TOOLKIT / "agents").glob("*.md"):
        assert "`python -c`" not in path.read_text(), path.name
