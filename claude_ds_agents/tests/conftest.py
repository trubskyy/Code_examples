"""Every test gets a throwaway home, so none can read or write the real ~/.claude or git config."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsguard import core  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    (home / ".claude").mkdir()
    (home / ".gitconfig").write_text("[user]\n\tname = T\n\temail = t@example.com\n[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(home / "system.gitconfig"))
    monkeypatch.delenv("DSGUARD_SKIP", raising=False)
    monkeypatch.delenv("DSGUARD_BIN", raising=False)
    monkeypatch.setattr(core, "USER_CONFIG", home / ".claude" / "dsguard.toml")
    return home
