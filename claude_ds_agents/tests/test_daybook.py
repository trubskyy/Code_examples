"""Tests for dsguard.daybook: the facts behind the start-of-day and end-of-day skills."""

import subprocess
import sys
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
    first = daybook.week_files(root, CFG, date.today())
    assert set(first["files"]) == {"src/a.py", "src/b.py"} and first["basis"] == "start of this week"

    daybook.mark_review(root, CFG)
    (root / "src/c.py").write_text("c = 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "c")
    after = daybook.week_files(root, CFG, date.today())
    assert after["files"] == ["src/c.py"] and after["basis"].startswith("last comment review")


def test_status_and_init(tmp_path):
    root = repo(tmp_path, {"src/a.py": "a = 1  # TODO tidy\n", "tests/test_a.py": "def test(): pass\n"})
    assert daybook.init_journal(root, CFG) == ["docs/journal/STORY.md"]
    (root / "docs/journal/plans/2026-10-22.md").write_text("## Needs you\n- [ ] carry me\n- [x] done\n")
    (root / "docs/journal/plans/2026-10-23.md").write_text("## Claude, this session\n- [ ] today item\n")
    s = daybook.status(root, date(2026, 10, 23))
    assert s["is_weekly_review_day"] and s["next_workday"] == "2026-10-26"
    assert [i["text"] for i in s["plans"]["previous"]["open"]] == ["carry me"]
    assert [i["text"] for i in s["plans"]["today"]["open"]] == ["today item"]
    assert s["story"]["exists"] and s["story"]["entries"] == 0
    assert s["todos"][0]["path"] == "src/a.py" and s["tests"]
    assert "docs/journal/" in " ".join(s["uncommitted"])
    assert "Fri 2026-10-23" in daybook.render_text(s)
