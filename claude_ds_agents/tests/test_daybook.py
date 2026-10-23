"""Tests for dsguard.daybook: the facts behind the start-of-day and end-of-day skills."""

import subprocess
import sys

import pytest
from datetime import date
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT))

from dsguard import daybook  # noqa: E402
from dsguard.core import DEFAULTS  # noqa: E402

CFG = DEFAULTS["daybook"]


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


def repo(tmp_path, files):
    tmp_path.mkdir(parents=True, exist_ok=True)
    git(tmp_path, "init", "-q")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_workdays_friday_rolls_to_monday():
    fri = daybook.workday_facts(date(2026, 10, 23), CFG)
    assert fri["weekday"] == "Fri" and fri["is_weekly_review_day"] and fri["next_workday"] == "2026-10-26"
    assert fri["week_start"] == "2026-10-19"
    tue = daybook.workday_facts(date(2026, 10, 20), CFG)
    assert tue["next_workday"] == "2026-10-21" and not tue["is_weekly_review_day"]
    sat = daybook.workday_facts(date(2026, 10, 24), CFG)
    assert not sat["is_workday"] and sat["next_workday"] == "2026-10-26"
    four_day = daybook.workday_facts(date(2026, 10, 22), CFG | {"workdays": ["Mon", "Tue", "Wed", "Thu"]})
    assert four_day["next_workday"] == "2026-10-26"


def test_checkbox_items_keep_sections():
    text = "# Plan\n## Needs you\n- [ ] decide window\n- [x] review PR\n## Claude, this session\n* [ ] add tests\n"
    items = daybook.checkbox_items(text)
    assert [(i["done"], i["section"]) for i in items] == [(False, "Needs you"), (True, "Needs you"),
                                                          (False, "Claude, this session")]


def test_todos_only_in_real_python_comments(tmp_path):
    root = repo(tmp_path, {
        "a.py": 'MSG = "# TODO not a comment"\nx = 1  # FIXME: real one\n',
        "q.sql": "SELECT 1 -- TODO add filter\n",
        "README.md": "TODO in docs is ignored here\n",
    })
    found = daybook.todo_markers(root, ["a.py", "q.sql", "README.md"], CFG["todo_markers"])
    assert [(t["path"], t["line"], t["marker"]) for t in found] == [("a.py", 2, "FIXME"), ("q.sql", 1, "TODO")]


def test_next_steps_sections_and_table_headers(tmp_path):
    root = repo(tmp_path, {"README.md": (
        "# Proj\n## Usage\n- not this\n## Next steps\n| Item | Why |\n|---|---|\n| retrain | drift |\n"
        "- add tests\n### Detail under next steps\n- still counts\n## Other\n- not this either\n")})
    items = [n["text"] for n in daybook.next_step_sections(root, ["README.md"], "docs/journal")]
    assert items == ["| retrain | drift |", "- add tests", "- still counts"]


def test_md_check_links_and_mentions(tmp_path):
    root = repo(tmp_path, {
        "docs/guide.md": "[ok](../src/model.py) [bad](missing.md) [web](https://x.io) [anchor](#top)\nSee model.py.\n",
        "src/model.py": "x = 1\n",
    })
    result = daybook.md_check(root, ["docs/guide.md"], ["src/model.py"])
    assert [b["target"] for b in result["broken_links"]] == ["missing.md"]
    assert result["mentions_changed_files"] == {"docs/guide.md": ["src/model.py"]}


def test_week_files_use_marker_after_review(tmp_path):
    root = repo(tmp_path, {"src/a.py": "a = 1\n", "notes.md": "x\n"})
    (root / "src/b.py").write_text("b = 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "b")
    journal = root / "docs/journal"
    first = daybook.week_files(root, CFG, date.today(), journal)
    assert set(first["files"]) == {"src/a.py", "src/b.py"} and first["basis"] == "start of this week"

    cfg = _isolated(root, tmp_path / "jhome")
    daybook.init_journal(root, cfg, "repo", date.today())
    daybook.mark_review(root, cfg)
    (root / "src/c.py").write_text("c = 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "c")
    after = daybook.week_files(root, CFG, date.today(), journal)
    assert after["files"] == ["src/c.py"] and after["basis"].startswith("last comment review")


def _isolated(root, home):
    """Point the journal home (registry, private journals) at a temp dir, never the real ~/.claude."""
    (root / "dsguard.toml").write_text(f'[daybook]\nhome_dir = "{home}"\n')
    return CFG | {"home_dir": str(home)}


def test_status_and_repo_journal(tmp_path):
    root = repo(tmp_path / "proj", {"src/a.py": "a = 1  # TODO tidy\n", "tests/test_a.py": "def test(): pass\n"})
    cfg = _isolated(root, tmp_path / "jhome")
    first = daybook.status(root, date(2026, 10, 23))
    assert first["journal"]["needs_setup"] and first["journal"]["mode"] is None

    journal, created = daybook.init_journal(root, cfg, "repo", date(2026, 10, 23))
    assert journal == root / "docs/journal" and created[0].endswith("docs/journal/STORY.md")
    (journal / "plans/2026-10-22.md").write_text("## Needs you\n- [ ] carry me\n- [x] done\n")
    (journal / "plans/2026-10-23.md").write_text("## Claude, this session\n- [ ] today item\n")
    s = daybook.status(root, date(2026, 10, 23))
    assert s["journal"]["mode"] == "repo" and s["journal"]["commit_root"] == str(root)
    assert s["is_weekly_review_day"] and s["next_workday"] == "2026-10-26"
    assert [i["text"] for i in s["plans"]["previous"]["open"]] == ["carry me"]
    assert [i["text"] for i in s["plans"]["today"]["open"]] == ["today item"]
    assert s["story"]["exists"] and s["story"]["entries"] == 0
    assert s["todos"][0]["path"] == "src/a.py" and s["tests"] == "python -m pytest -q"
    assert "docs/journal/STORY.md" in s["uncommitted"]
    assert "Fri 2026-10-23" in daybook.render_text(s)


def test_private_journal_registry_and_projects_overview(tmp_path):
    home = tmp_path / "jhome"
    a = repo(tmp_path / "alpha", {"x.py": "x = 1\n"})
    b = repo(tmp_path / "beta", {"y.py": "y = 1\n"})
    cfg_a, cfg_b = _isolated(a, home), _isolated(b, home)
    with pytest.raises(SystemExit):
        daybook.init_journal(a, cfg_a, None, date(2026, 10, 22))       # must choose a location first
    journal_a, created = daybook.init_journal(a, cfg_a, "home", date(2026, 10, 22))
    assert journal_a.parent == home and journal_a.name.startswith("alpha-") and str(home) in created[0]
    assert subprocess.run(["git", "-C", str(home), "rev-parse"], capture_output=True).returncode == 0
    assert not (a / "docs").exists()                                   # nothing written into the project
    daybook.init_journal(b, cfg_b, "repo", date(2026, 10, 23))
    (journal_a / "plans/2026-10-22.md").write_text("- [ ] alpha task\n")

    s = daybook.status(a, date(2026, 10, 23))
    assert s["journal"]["mode"] == "home" and s["journal"]["commit_root"] == str(home)
    overview = daybook.projects(cfg_a, date(2026, 10, 23))
    assert {p["name"] for p in overview} == {"alpha", "beta"}
    alpha = next(p for p in overview if p["name"] == "alpha")
    assert alpha["open_items"] == 1 and alpha["top_open"] == ["alpha task"] and alpha["mode"] == "home"
    assert "alpha" in daybook.render_projects(overview)


def test_registry_remembers_choice_across_runs(tmp_path):
    home = tmp_path / "jhome"
    root = repo(tmp_path / "gamma", {"z.py": "z = 1\n"})
    cfg = _isolated(root, home)
    daybook.init_journal(root, cfg, "home", date(2026, 10, 23))
    journal, mode = daybook.resolve_journal(root, cfg)
    assert mode == "home" and journal.parent == home


def test_non_git_project_uses_file_times(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    (root / "analysis.R").write_text("y <- 2  # FIXME units\n")
    _isolated(root, tmp_path / "jhome")
    s = daybook.status(root, date.today())
    assert not s["project"]["git"] and s["project"]["branch"] is None
    assert "analysis.R" in s["changed_today"] and s["commits_today"] == [] and s["uncommitted"] == []
    assert s["todos"][0]["marker"] == "FIXME"
    assert s["weekly_review"]["files"] == ["analysis.R"]


@pytest.mark.parametrize("files,contents,expected", [
    ({"tests/test_x.py": ""}, {}, "python -m pytest -q"),
    ({"package.json": ""}, {"package.json": '{"scripts": {"test": "jest"}}'}, "npm test"),
    ({"package.json": ""}, {"package.json": '{"scripts": {"test": "echo \\"Error: no test specified\\""}}'}, None),
    ({"Makefile": ""}, {"Makefile": "build:\n\techo\ntest:\n\tpytest\n"}, "make test"),
    ({"tests/testthat/test-a.R": ""}, {}, "Rscript -e 'testthat::test_dir(\"tests/testthat\")'"),
    ({"notes.txt": ""}, {}, None),
])
def test_detect_tests(tmp_path, files, contents, expected):
    for rel, text in (files | contents).items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert daybook.detect_tests(tmp_path, list(files), CFG) == expected
    assert daybook.detect_tests(tmp_path, list(files), CFG | {"test_command": "tox"}) == "tox"


def test_outside_any_project_lists_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(daybook.Path, "home", classmethod(lambda cls: tmp_path))
    s = daybook.status(tmp_path, date(2026, 10, 23))
    assert s["project"] is None and isinstance(s["projects"], list)
