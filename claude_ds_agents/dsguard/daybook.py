"""Daybook: the deterministic facts behind the start-of-day and end-of-day skills.

The skills decide what to do; this module only gathers facts that a script can get right
every time: dates and workdays, what is still open (unchecked plan items, TODO markers,
"next steps" sections, uncommitted work), what changed today or since the last weekly
comment review, and Markdown files with broken links or that mention changed files.

Journal layout (default ``docs/journal/``, committed with the project):

    STORY.md              continuous narrative of the project, one ``## YYYY-MM-DD`` entry per day
    plans/YYYY-MM-DD.md   the plan for that day (drafted at end of day, refined at start of day)
    .last-comment-review  commit hash at which the last weekly comment pass finished
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath

from .core import discover_files, git, load_config

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CHECKBOX = re.compile(r"^\s*[-*]\s+\[( |x|X)\]\s+(.*)$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
STORY_ENTRY = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\b", re.M)
NEXT_STEPS = re.compile(r"(?i)^(next steps?|todo|to do|roadmap|backlog|open (questions|items)|follow-?ups?)\b")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
STORY_TEMPLATE = """# Project story

A running narrative of this project: what was built, why, what was tried and dropped, and
what comes next. One entry per working day, newest at the bottom. Written at end of day by
the `ds-end-day` skill from commits, the day's plan and the conversation; edit freely.

## Story so far

_Refreshed each week by the end-of-day review._
"""


def _journal(root: Path, cfg: dict) -> Path:
    return root / cfg["dir"]


def workday_facts(today: date, cfg: dict) -> dict:
    """Weekday, whether today is the weekly review day, and the next configured workday."""
    workdays = [DAY_NAMES.index(d) for d in cfg["workdays"]]
    nxt = today + timedelta(days=1)
    while nxt.weekday() not in workdays:
        nxt += timedelta(days=1)
    review_day = DAY_NAMES.index(cfg["weekly_review_day"])
    week_start = today - timedelta(days=today.weekday())
    return {
        "date": today.isoformat(),
        "weekday": DAY_NAMES[today.weekday()],
        "is_workday": today.weekday() in workdays,
        "is_weekly_review_day": today.weekday() == review_day,
        "next_workday": nxt.isoformat(),
        "week_start": week_start.isoformat(),
    }


def checkbox_items(text: str) -> list[dict]:
    """Markdown task items with the heading they sit under."""
    items, section = [], ""
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            section = heading.group(2)
            continue
        box = CHECKBOX.match(line)
        if box:
            items.append({"done": box.group(1).lower() == "x", "text": box.group(2).strip(), "section": section})
    return items


def latest_plan(journal: Path, today: date) -> tuple[Path | None, Path | None]:
    """(plan for today if it exists, most recent plan dated before today)."""
    plans = sorted((journal / "plans").glob("????-??-??.md")) if (journal / "plans").is_dir() else []
    todays = journal / "plans" / f"{today.isoformat()}.md"
    earlier = [p for p in plans if p.stem < today.isoformat()]
    return (todays if todays.exists() else None), (earlier[-1] if earlier else None)


def _comment_lines(rel: str, text: str):
    """(line number, text) pairs to search for markers. For Python, only real comments, so a
    '# TODO' inside a string literal is not reported; other languages fall back to every line."""
    if rel.endswith(".py"):
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            tokens = None
        if tokens is not None:
            return [(t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT]
    return list(enumerate(text.splitlines(), 1))


def todo_markers(root: Path, files: list[str], markers: list[str], limit: int = 200) -> list[dict]:
    pattern = re.compile(rf"(?:#|//|--|/\*|<!--)\s*({'|'.join(map(re.escape, markers))})\b[:\s-]*(.*)")
    found = []
    for rel in files:
        if rel.endswith((".md", ".ipynb")) or rel.startswith(("docs/journal/", ".dsguard/")):
            continue
        try:
            text = (root / rel).read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        if "\0" in text[:4096]:
            continue
        for lineno, line in _comment_lines(rel, text):
            m = pattern.search(line)
            if m:
                found.append({"path": rel, "line": lineno, "marker": m.group(1), "text": m.group(2).strip()[:160]})
                if len(found) >= limit:
                    return found
    return found


def next_step_sections(root: Path, md_files: list[str], journal_dir: str) -> list[dict]:
    """Bullets under headings like 'Next steps', 'TODO', 'Roadmap' in project Markdown."""
    out = []
    for rel in md_files:
        if rel.startswith(journal_dir + "/"):
            continue
        lines = (root / rel).read_text(errors="replace").splitlines()
        level, title = None, None
        for i, line in enumerate(lines):
            if i + 1 < len(lines) and re.match(r"^\s*\|[-:\s|]+\|?\s*$", lines[i + 1]):
                continue  # table header row
            heading = HEADING.match(line)
            if heading:
                depth = len(heading.group(1))
                if NEXT_STEPS.match(heading.group(2)):
                    level, title = depth, heading.group(2)
                elif level is not None and depth <= level:
                    level = title = None
                continue
            if title and re.match(r"^\s*([-*]|\d+\.|\|)\s*\S", line) and not re.match(r"^\s*\|[-\s|]+\|?\s*$", line):
                out.append({"path": rel, "section": title, "text": line.strip()[:200]})
    return out


def changed_since(root: Path, since: str) -> list[str]:
    """Files touched by commits since a date/ref, plus uncommitted and untracked changes."""
    files: set[str] = set()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", since):
        out = git(root, "log", f"--since={since}", "--name-only", "--pretty=format:")
    else:
        out = git(root, "diff", "--name-only", f"{since}..HEAD")
    files.update(p for p in (out or "").splitlines() if p)
    files.update(p for p in (git(root, "diff", "--name-only", "HEAD") or "").splitlines() if p)
    files.update(p for p in (git(root, "ls-files", "-o", "--exclude-standard") or "").splitlines() if p)
    return sorted(f for f in files if (root / f).is_file())


def commits_since(root: Path, since: str) -> list[dict]:
    out = git(root, "log", f"--since={since}", "--pretty=format:%h\t%ad\t%s", "--date=format:%Y-%m-%d %H:%M") or ""
    return [dict(zip(("hash", "date", "subject"), line.split("\t", 2), strict=False)) for line in out.splitlines() if line]


def week_files(root: Path, cfg: dict, today: date) -> dict:
    """Code files to review on the weekly comment pass: changed since the last pass, or this week."""
    marker = _journal(root, cfg) / ".last-comment-review"
    since, basis = workday_facts(today, cfg)["week_start"], "start of this week"
    if marker.exists():
        sha = marker.read_text().strip()
        if git(root, "cat-file", "-e", f"{sha}^{{commit}}") is not None:
            since, basis = sha, f"last comment review ({sha[:8]})"
    files = [f for f in changed_since(root, since)
             if PurePosixPath(f).suffix in cfg["code_extensions"] and not f.startswith((cfg["dir"] + "/", ".dsguard/"))]
    return {"since": since, "basis": basis, "files": files}


def mark_review(root: Path, cfg: dict) -> str:
    sha = (git(root, "rev-parse", "HEAD") or "").strip()
    marker = _journal(root, cfg) / ".last-comment-review"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(sha + "\n")
    return sha


def md_check(root: Path, md_files: list[str], changed: list[str]) -> dict:
    """Broken relative links, and Markdown that mentions files changed in the period."""
    broken, mentions = [], {}
    names = {f: (f, PurePosixPath(f).name) for f in changed if not f.endswith(".md")}
    for rel in md_files:
        text = (root / rel).read_text(errors="replace")
        for m in MD_LINK.finditer(text):
            target = m.group(1).split("#")[0]
            if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
                continue
            if not (root / PurePosixPath(rel).parent / target).exists():
                broken.append({"path": rel, "line": text.count("\n", 0, m.start()) + 1, "target": m.group(1)})
        hits = sorted(f for f, (full, base) in names.items() if full in text or re.search(rf"\b{re.escape(base)}\b", text))
        if hits:
            mentions[rel] = hits
    return {"broken_links": broken, "mentions_changed_files": mentions}


def init_journal(root: Path, cfg: dict) -> list[str]:
    journal = _journal(root, cfg)
    created = []
    (journal / "plans").mkdir(parents=True, exist_ok=True)
    story = journal / "STORY.md"
    if not story.exists():
        story.write_text(STORY_TEMPLATE)
        created.append(str(story.relative_to(root)))
    return created


def status(root: Path, today: date | None = None) -> dict:
    today = today or date.today()
    cfg = load_config(root)["daybook"]
    journal = _journal(root, cfg)
    files = discover_files(root)
    md_files = [f for f in files if f.endswith(".md") and not f.startswith(".dsguard/")]
    todays_plan, previous_plan = latest_plan(journal, today)
    plan_items = {}
    for label, plan in (("today", todays_plan), ("previous", previous_plan)):
        if plan:
            items = checkbox_items(plan.read_text())
            plan_items[label] = {"path": str(plan.relative_to(root)), "open": [i for i in items if not i["done"]],
                                 "done": [i for i in items if i["done"]]}
    story = journal / "STORY.md"
    story_dates = STORY_ENTRY.findall(story.read_text()) if story.exists() else []
    changed_today = changed_since(root, today.isoformat())
    porcelain = git(root, "status", "--porcelain", "--untracked-files=all") or ""   # files, not collapsed dirs
    return {
        **workday_facts(today, cfg),
        "branch": (git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
        "journal": cfg["dir"],
        "story": {"exists": story.exists(), "entries": len(story_dates), "last_entry": story_dates[-1] if story_dates else None},
        "plans": plan_items,
        "uncommitted": [line[3:] for line in porcelain.splitlines() if line.strip()],
        "commits_today": commits_since(root, today.isoformat()),
        "changed_today": changed_today,
        "todos": todo_markers(root, files, cfg["todo_markers"]),
        "next_steps": next_step_sections(root, md_files, cfg["dir"]),
        "md_files": md_files,
        "md_check_today": md_check(root, md_files, changed_today),
        "weekly_review": week_files(root, cfg, today),
        "push_after_commit": cfg["push"],
        "tests": "python -m pytest -q" if any(f.startswith("tests/") or "/tests/" in f for f in files) else None,
    }


def render_text(s: dict) -> str:
    lines = [f"{s['weekday']} {s['date']}  branch {s['branch']}  next workday {s['next_workday']}"
             + ("  ** weekly review day **" if s["is_weekly_review_day"] else "")]
    story = s["story"]
    lines.append(f"story: {'%d entries, last %s' % (story['entries'], story['last_entry']) if story['exists'] else 'missing (dsguard day init)'}")
    for label, plan in s["plans"].items():
        lines.append(f"plan ({label}) {plan['path']}: {len(plan['open'])} open, {len(plan['done'])} done")
        lines += [f"   [ ] {i['text']}  ({i['section']})" for i in plan["open"][:15]]
    if not s["plans"]:
        lines.append("plan: none yet")
    lines.append(f"uncommitted: {len(s['uncommitted'])}  commits today: {len(s['commits_today'])}  files changed today: {len(s['changed_today'])}")
    lines.append(f"TODO markers: {len(s['todos'])}")
    lines += [f"   {t['path']}:{t['line']} {t['marker']} {t['text']}" for t in s["todos"][:10]]
    lines.append(f"next-step items in docs: {len(s['next_steps'])}")
    lines += [f"   {n['path']} [{n['section']}] {n['text'][:100]}" for n in s["next_steps"][:10]]
    mc = s["md_check_today"]
    lines.append(f"markdown: {len(s['md_files'])} files, {len(mc['broken_links'])} broken links, "
                 f"{len(mc['mentions_changed_files'])} mention files changed today")
    wr = s["weekly_review"]
    lines.append(f"weekly comment review: {len(wr['files'])} code files changed since {wr['basis']}")
    return "\n".join(lines)


def main(args) -> int:
    from .core import find_root
    root = find_root(Path(args.root))
    today = datetime.strptime(args.date, "%Y-%m-%d").date() if getattr(args, "date", None) else date.today()
    cfg = load_config(root)["daybook"]
    if args.action == "status":
        s = status(root, today)
        print(json.dumps(s, indent=2) if args.format == "json" else render_text(s))
    elif args.action == "week-files":
        print(json.dumps(week_files(root, cfg, today), indent=2))
    elif args.action == "md-check":
        files = discover_files(root)
        md = [f for f in files if f.endswith(".md") and not f.startswith(".dsguard/")]
        print(json.dumps(md_check(root, md, changed_since(root, args.since or today.isoformat())), indent=2))
    elif args.action == "mark-review":
        print(f"weekly comment review marked at {mark_review(root, cfg)[:8]}")
    elif args.action == "init":
        created = init_journal(root, cfg)
        print("created: " + ", ".join(created) if created else f"journal already present in {cfg['dir']}")
    return 0
