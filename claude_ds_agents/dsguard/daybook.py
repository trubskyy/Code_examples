"""Daybook: the deterministic facts behind the start-of-day and end-of-day skills.

The skills decide what to do; this module only gathers facts that a script can get right
every time: dates and workdays, what is still open (unchecked plan items, TODO markers,
"next steps" sections, uncommitted work), what changed today or since the last weekly
comment review, and Markdown files with broken links or that mention changed files.

It works in any project: the skills are installed once in ~/.claude and run against the
current working directory. Each project gets its own journal, in one of two places (chosen
once per project, then remembered):

    repo   <project>/docs/journal/               committed with the project (visible to the team)
    home   ~/.claude/journal/<name>-<id>/         private; ~/.claude/journal is its own git repo

Journal layout (either place):

    STORY.md              continuous narrative of the project, one ``## YYYY-MM-DD`` entry per day
    plans/YYYY-MM-DD.md   the plan for that day (drafted at end of day, refined at start of day)
    .last-comment-review  commit hash at which the last weekly comment pass finished

~/.claude/journal/projects.json registers every project the skills have touched, so
``dsguard day projects`` (and start-of-day outside a project) can show them all.
Projects without git still work; changes are then detected by file modification time.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
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
STORY_TEMPLATE = """# Project story: {name}

A running narrative of this project: what was built, why, what was tried and dropped, and
what comes next. One entry per working day, newest at the bottom. Written at end of day by
the `ds-end-day` skill from commits, the day's plan and the conversation; edit freely.

## Story so far

_Refreshed each week by the end-of-day review._
"""
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _home(cfg: dict) -> Path:
    return Path(cfg["home_dir"]).expanduser()


def _is_git(root: Path) -> bool:
    return git(root, "rev-parse", "--is-inside-work-tree") is not None


def project_slug(root: Path) -> str:
    """Stable private-journal folder name: project name plus a short id from its remote (or path)."""
    remote = (git(root, "config", "--get", "remote.origin.url") or "").strip()
    digest = hashlib.sha1((remote or str(root.resolve())).encode()).hexdigest()[:6]
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', root.name)}-{digest}"


def load_registry(cfg: dict) -> dict:
    try:
        return json.loads((_home(cfg) / "projects.json").read_text())
    except (OSError, ValueError):
        return {}


def touch(root: Path, cfg: dict, today: date, **fields) -> None:
    """Record that a project was worked on (by any toolkit command), merging in extra fields.

    Never raises: the registry is a convenience, and failing to write it must not break a check.
    """
    try:
        registry = load_registry(cfg)
        key = str(root.resolve())
        registry[key] = registry.get(key, {"mode": None, "journal": None}) | {
            "name": root.name, "last_seen": today.isoformat(), **fields}
        _home(cfg).mkdir(parents=True, exist_ok=True)
        (_home(cfg) / "projects.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass


def register(root: Path, cfg: dict, journal: Path, mode: str, today: date) -> None:
    touch(root, cfg, today, mode=mode, journal=str(journal))


def resolve_journal(root: Path, cfg: dict) -> tuple[Path | None, str | None]:
    """Where this project's journal lives: explicit config, an existing journal, the registry, or unknown."""
    in_repo, private = root / cfg["dir"], _home(cfg) / project_slug(root)
    if cfg["location"] == "repo":
        return in_repo, "repo"
    if cfg["location"] == "home":
        return private, "home"
    if (in_repo / "STORY.md").exists():
        return in_repo, "repo"
    entry = load_registry(cfg).get(str(root.resolve()))
    if entry and entry.get("journal"):
        return Path(entry["journal"]), entry["mode"]
    if (private / "STORY.md").exists():
        return private, "home"
    return None, None


def journal_commit_root(root: Path, cfg: dict, mode: str | None) -> Path | None:
    """The git repository the journal is committed to."""
    if mode == "home":
        return _home(cfg)
    return root if mode == "repo" and _is_git(root) else None


def detect_tests(root: Path, files: list[str], cfg: dict) -> str | None:
    """The project's own test command, from config or from common conventions."""
    if cfg["test_command"]:
        return cfg["test_command"]
    names = set(files)
    pyproject = (root / "pyproject.toml").read_text(errors="replace") if "pyproject.toml" in names else ""
    if ({"pytest.ini", "conftest.py", "tox.ini"} & names or "[tool.pytest" in pyproject
            or any(re.search(r"(^|/)tests?/.*\.py$|(^|/)test_[^/]*\.py$", f) for f in files)):
        return "python -m pytest -q"
    if "package.json" in names:
        try:
            script = json.loads((root / "package.json").read_text()).get("scripts", {}).get("test", "")
        except ValueError:
            script = ""
        if script and "no test specified" not in script:
            return "npm test"
    if "Makefile" in names and re.search(r"^test:", (root / "Makefile").read_text(errors="replace"), re.M):
        return "make test"
    if any(f.startswith("tests/testthat/") for f in files):
        return "Rscript -e 'testthat::test_dir(\"tests/testthat\")'"
    return None


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


def todo_markers(root: Path, files: list[str], markers: list[str], skip: tuple = (".dsguard/",),
                 limit: int = 200) -> list[dict]:
    pattern = re.compile(rf"(?:#|//|--|/\*|<!--)\s*({'|'.join(map(re.escape, markers))})\b[:\s-]*(.*)")
    found = []
    for rel in files:
        if rel.endswith((".md", ".ipynb")) or rel.startswith(skip):
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


def next_step_sections(root: Path, md_files: list[str], skip: tuple = ()) -> list[dict]:
    """Bullets under headings like 'Next steps', 'TODO', 'Roadmap' in project Markdown."""
    out = []
    for rel in md_files:
        if skip and rel.startswith(skip):
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
    """Files touched by commits since a date/ref, plus uncommitted and untracked changes.

    Without git, falls back to files modified on or after the date."""
    if not _is_git(root):
        start = datetime.fromisoformat(since[:10]).timestamp() if DATE_RE.match(since) else 0.0
        return sorted(f for f in discover_files(root) if (root / f).stat().st_mtime >= start)
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


def week_files(root: Path, cfg: dict, today: date, journal: Path | None = None) -> dict:
    """Code files to review on the weekly comment pass: changed since the last pass, or this week."""
    since, basis = workday_facts(today, cfg)["week_start"], "start of this week"
    marker = journal / ".last-comment-review" if journal else None
    if marker and marker.exists():
        sha = marker.read_text().strip()
        if git(root, "cat-file", "-e", f"{sha}^{{commit}}") is not None:
            since, basis = sha, f"last comment review ({sha[:8]})"
    files = [f for f in changed_since(root, since)
             if PurePosixPath(f).suffix in cfg["code_extensions"] and not f.startswith((cfg["dir"] + "/", ".dsguard/"))]
    return {"since": since, "basis": basis, "files": files}


def mark_review(root: Path, cfg: dict) -> str:
    journal, _ = resolve_journal(root, cfg)
    if journal is None:
        raise SystemExit("dsguard: no journal for this project yet; run `dsguard day init --location repo|home`")
    sha = (git(root, "rev-parse", "HEAD") or "").strip()
    marker = journal / ".last-comment-review"
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


def init_journal(root: Path, cfg: dict, location: str | None, today: date) -> tuple[Path, list[str]]:
    """Create (or adopt) this project's journal and register it. location: repo | home."""
    journal, mode = resolve_journal(root, cfg)
    if journal is None:
        if location not in ("repo", "home"):
            raise SystemExit("dsguard: choose where this project's journal lives: --location repo|home")
        mode = location
        journal = root / cfg["dir"] if mode == "repo" else _home(cfg) / project_slug(root)
    created = []
    if mode == "home" and not _is_git(_home(cfg)):
        _home(cfg).mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(_home(cfg))], check=True)
        created.append(f"{_home(cfg)} (private journal git repo)")
    (journal / "plans").mkdir(parents=True, exist_ok=True)
    story = journal / "STORY.md"
    if not story.exists():
        story.write_text(STORY_TEMPLATE.format(name=root.name))
        created.append(str(story))
    register(root, cfg, journal, mode, today)
    return journal, created


def projects(cfg: dict, today: date) -> list[dict]:
    """Every project the toolkit has touched: last seen, journal state, open plan items, last dsguard verdict."""
    out = []
    for path, entry in load_registry(cfg).items():
        journal = Path(entry["journal"]) if entry.get("journal") else None
        story = journal / "STORY.md" if journal else None
        dates = STORY_ENTRY.findall(story.read_text()) if story and story.exists() else []
        todays, previous = latest_plan(journal, today) if journal else (None, None)
        plan = todays or previous
        open_items = [i for i in checkbox_items(plan.read_text()) if not i["done"]] if plan else []
        out.append({"name": entry["name"], "path": path, "exists": Path(path).is_dir(), "mode": entry.get("mode"),
                    "last_seen": entry.get("last_seen"), "last_story_entry": dates[-1] if dates else None,
                    "latest_plan": plan.stem if plan else None, "open_items": len(open_items),
                    "top_open": [i["text"] for i in open_items[:3]], "dsguard": entry.get("dsguard")})
    return sorted(out, key=lambda p: p["last_seen"] or "", reverse=True)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def status(root: Path, today: date | None = None) -> dict:
    today = today or date.today()
    cfg = load_config(root)["daybook"]
    if root.resolve() in (Path.home().resolve(), Path("/")):
        # Not inside a project: show every project the skills know about, so the user can pick one.
        return {**workday_facts(today, cfg), "project": None, "projects": projects(cfg, today)}

    journal, mode = resolve_journal(root, cfg)
    if journal is not None:
        register(root, cfg, journal, mode, today)
    files = discover_files(root)
    in_repo_journal = (cfg["dir"] + "/",) if mode == "repo" else ()
    md_files = [f for f in files if f.endswith(".md") and not f.startswith((".dsguard/", *in_repo_journal))]
    plan_items = {}
    story_dates: list[str] = []
    if journal is not None:
        todays_plan, previous_plan = latest_plan(journal, today)
        for label, plan in (("today", todays_plan), ("previous", previous_plan)):
            if plan:
                items = checkbox_items(plan.read_text())
                plan_items[label] = {"path": _rel(plan, root), "open": [i for i in items if not i["done"]],
                                     "done": [i for i in items if i["done"]]}
        story = journal / "STORY.md"
        story_dates = STORY_ENTRY.findall(story.read_text()) if story.exists() else []
    is_git = _is_git(root)
    changed_today = changed_since(root, today.isoformat())
    porcelain = (git(root, "status", "--porcelain", "--untracked-files=all") or "") if is_git else ""
    commit_root = journal_commit_root(root, cfg, mode)
    return {
        **workday_facts(today, cfg),
        "project": {"root": str(root), "name": root.name, "git": is_git,
                    "branch": (git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() if is_git else None},
        "journal": {"path": str(journal) if journal else None, "mode": mode,
                    "needs_setup": journal is None,
                    "commit_root": str(commit_root) if commit_root else None,
                    "story": _rel(journal / "STORY.md", root) if journal else None,
                    "plans_dir": _rel(journal / "plans", root) if journal else None},
        "story": {"exists": bool(journal and (journal / "STORY.md").exists()), "entries": len(story_dates),
                  "last_entry": story_dates[-1] if story_dates else None},
        "plans": plan_items,
        "uncommitted": [line[3:] for line in porcelain.splitlines() if line.strip()],
        "commits_today": commits_since(root, today.isoformat()) if is_git else [],
        "changed_today": changed_today,
        "todos": todo_markers(root, files, cfg["todo_markers"], skip=(".dsguard/", *in_repo_journal)),
        "next_steps": next_step_sections(root, md_files),
        "md_files": md_files,
        "md_check_today": md_check(root, md_files, changed_today),
        "weekly_review": week_files(root, cfg, today, journal),
        "push_after_commit": cfg["push"],
        "tests": detect_tests(root, files, cfg),
    }


def render_projects(items: list[dict]) -> str:
    if not items:
        return "no projects registered yet: run /ds-start-day inside a project"
    lines = [f"{len(items)} project(s), most recent first:"]
    for p in items:
        missing = "" if p["exists"] else "  (folder not found)"
        check = p.get("dsguard") or {}
        verdict = f"{check.get('verdict', '-')} {check.get('date', '')}".strip()
        lines.append(f"  {p['name']:<24} last seen {p['last_seen'] or '-':<10}  story {p['last_story_entry'] or '-':<10}"
                     f"  open {p['open_items']:<3} dsguard {verdict:<20} [{p['mode'] or 'no journal'}]  {p['path']}{missing}")
        lines += [f"      [ ] {t}" for t in p["top_open"]]
    return "\n".join(lines)


def render_text(s: dict) -> str:
    head = f"{s['weekday']} {s['date']}  next workday {s['next_workday']}" + (
        "  ** weekly review day **" if s["is_weekly_review_day"] else "")
    if s["project"] is None:
        return head + "\nnot inside a project\n" + render_projects(s["projects"])
    p, j = s["project"], s["journal"]
    lines = [head, f"project {p['name']}  ({p['root']})  " + (f"branch {p['branch']}" if p["git"] else "no git: changes by file time")]
    if j["needs_setup"]:
        lines.append("journal: not set up. Choose: dsguard day init --location repo|home")
    else:
        story = s["story"]
        lines.append(f"journal [{j['mode']}] {j['path']}  story: {story['entries']} entries, last {story['last_entry'] or '-'}")
    for label, plan in s["plans"].items():
        lines.append(f"plan ({label}) {plan['path']}: {len(plan['open'])} open, {len(plan['done'])} done")
        lines += [f"   [ ] {i['text']}  ({i['section']})" for i in plan["open"][:15]]
    if not s["plans"]:
        lines.append("plan: none yet")
    lines.append(f"uncommitted: {len(s['uncommitted'])}  commits today: {len(s['commits_today'])}  "
                 f"files changed today: {len(s['changed_today'])}  tests: {s['tests'] or 'none detected'}")
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
    elif args.action == "projects":
        items = projects(cfg, today)
        print(json.dumps(items, indent=2) if args.format == "json" else render_projects(items))
    elif args.action == "week-files":
        journal, _ = resolve_journal(root, cfg)
        print(json.dumps(week_files(root, cfg, today, journal), indent=2))
    elif args.action == "md-check":
        files = discover_files(root)
        md = [f for f in files if f.endswith(".md") and not f.startswith(".dsguard/")]
        print(json.dumps(md_check(root, md, changed_since(root, args.since or today.isoformat())), indent=2))
    elif args.action == "mark-review":
        print(f"weekly comment review marked at {mark_review(root, cfg)[:8]}")
    elif args.action == "init":
        journal, created = init_journal(root, cfg, args.location, today)
        print(("created: " + ", ".join(created)) if created else f"journal already set up at {journal}")
    return 0
