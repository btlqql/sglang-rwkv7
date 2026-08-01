import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "scripts" / "git_identity_guard.py"
SPEC = importlib.util.spec_from_file_location("git_identity_guard", MODULE_PATH)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def identity():
    return guard.GitHubIdentity(
        login="alice",
        user_id=1234,
        emails=frozenset({"alice@example.com"}),
    )


@pytest.mark.parametrize(
    "email",
    [
        "alice@example.com",
        "alice@users.noreply.github.com",
        "1234+alice@users.noreply.github.com",
    ],
)
def test_accepts_verified_and_noreply_emails(email):
    guard.assert_ident_matches(identity(), "author", "alice", email)


def test_rejects_a_different_login():
    with pytest.raises(guard.GuardError, match="does not match"):
        guard.assert_ident_matches(identity(), "author", "bob", "alice@example.com")


def test_rejects_an_unverified_email():
    with pytest.raises(guard.GuardError, match="is not verified"):
        guard.assert_ident_matches(identity(), "author", "alice", "bob@example.com")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/acme/project.git", ("acme", "project")),
        ("git@github.com:acme/project.git", ("acme", "project")),
        ("ssh://git@github.com/acme/project", ("acme", "project")),
        ("ssh://host/path/project", None),
    ],
)
def test_parse_github_repo(url, expected):
    assert guard.parse_github_repo(url) == expected


def test_parse_push_updates():
    updates = list(
        guard.parse_push_updates(["refs/heads/work abc refs/heads/work def\n", "\n"])
    )
    assert updates == [("refs/heads/work", "abc", "refs/heads/work", "def")]


def test_accepts_branch_owned_by_active_account():
    guard.assert_branch_owned(identity(), "alice/feature")


@pytest.mark.parametrize("branch", ["main", "bob/feature", "feature"])
def test_rejects_shared_or_foreign_branch(branch):
    with pytest.raises(guard.GuardError):
        guard.assert_branch_owned(identity(), branch)
