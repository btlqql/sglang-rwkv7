#!/usr/bin/env python3
"""Keep local Git commits and pull-request updates bound to one GitHub account."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

ZERO_OID_RE = re.compile(r"^0+$")
IDENT_RE = re.compile(r"^(.*?) <([^<>]+)> \d+ [+-]\d{4}$")
GITHUB_REMOTE_RE = re.compile(
    r"^(?:https://(?:[^/@]+@)?github\.com/|ssh://git@github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?$"
)


class GuardError(RuntimeError):
    """A repository identity invariant was violated."""


@dataclass(frozen=True)
class GitHubIdentity:
    login: str
    user_id: int
    emails: frozenset[str]

    @property
    def allowed_emails(self) -> frozenset[str]:
        login = self.login.lower()
        return self.emails | frozenset(
            {
                f"{login}@users.noreply.github.com",
                f"{self.user_id}+{login}@users.noreply.github.com",
            }
        )


def run(*args: str, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            args,
            check=True,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GuardError(f"required command is not installed: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
        )
        raise GuardError(f"command failed ({' '.join(args)}): {detail}") from exc
    return result.stdout.strip()


def active_github_identity() -> GitHubIdentity:
    raw_user = run("gh", "api", "user")
    user = json.loads(raw_user)
    emails: set[str] = set()
    if user.get("email"):
        emails.add(str(user["email"]).lower())

    try:
        raw_emails = run("gh", "api", "user/emails", "--paginate")
        pages = json.loads(raw_emails)
        if pages and isinstance(pages[0], list):
            pages = [entry for page in pages for entry in page]
        for entry in pages:
            if entry.get("verified") and entry.get("email"):
                emails.add(str(entry["email"]).lower())
    except (GuardError, json.JSONDecodeError, TypeError):
        # Fine-grained tokens may not grant access to /user/emails. The two
        # canonical GitHub noreply forms remain valid and deterministic.
        pass

    return GitHubIdentity(
        login=str(user["login"]),
        user_id=int(user["id"]),
        emails=frozenset(emails),
    )


def parse_ident(value: str) -> tuple[str, str]:
    match = IDENT_RE.match(value.strip())
    if not match:
        raise GuardError(f"cannot parse Git identity: {value!r}")
    return match.group(1), match.group(2)


def assert_ident_matches(
    identity: GitHubIdentity, role: str, name: str, email: str
) -> None:
    if name.lower() != identity.login.lower():
        raise GuardError(
            f"{role} name {name!r} does not match active GitHub account "
            f"{identity.login!r}"
        )
    if email.lower() not in identity.allowed_emails:
        allowed = ", ".join(sorted(identity.allowed_emails))
        raise GuardError(
            f"{role} email {email!r} is not verified for {identity.login!r}; "
            f"allowed: {allowed}"
        )


def check_pending_commit(identity: GitHubIdentity) -> None:
    author = parse_ident(run("git", "var", "GIT_AUTHOR_IDENT"))
    committer = parse_ident(run("git", "var", "GIT_COMMITTER_IDENT"))
    assert_ident_matches(identity, "author", *author)
    assert_ident_matches(identity, "committer", *committer)


def parse_github_repo(remote_url: str) -> tuple[str, str] | None:
    match = GITHUB_REMOTE_RE.match(remote_url.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def commit_oids_for_update(
    remote_name: str, local_oid: str, remote_oid: str
) -> list[str]:
    if ZERO_OID_RE.match(local_oid):
        return []
    if not ZERO_OID_RE.match(remote_oid):
        return run("git", "rev-list", f"{remote_oid}..{local_oid}").splitlines()

    output = run(
        "git",
        "rev-list",
        local_oid,
        "--not",
        f"--remotes={remote_name}",
    )
    return output.splitlines()


def check_commit(identity: GitHubIdentity, oid: str) -> None:
    fields = run("git", "show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", oid).split(
        "\0"
    )
    if len(fields) != 4:
        raise GuardError(f"cannot read author and committer for {oid}")
    assert_ident_matches(identity, f"commit {oid[:12]} author", fields[0], fields[1])
    assert_ident_matches(identity, f"commit {oid[:12]} committer", fields[2], fields[3])


def open_pr_authors(repo: str, owner: str, branch: str) -> set[str]:
    endpoint = (
        f"repos/{repo}/pulls?state=open&head={quote(owner + ':' + branch)}&per_page=100"
    )
    pulls = json.loads(run("gh", "api", endpoint))
    return {str(pull["user"]["login"]) for pull in pulls}


def assert_branch_owned(identity: GitHubIdentity, branch: str) -> None:
    if branch in {"main", "master"}:
        raise GuardError(
            f"direct push to {branch!r} is disabled; push a personal branch and use a PR"
        )
    required_prefix = f"{identity.login}/"
    if not branch.lower().startswith(required_prefix.lower()):
        raise GuardError(
            f"branch {branch!r} is not owned by active account {identity.login!r}; "
            f"rename it with the {required_prefix!r} prefix"
        )


def check_pr_ownership(
    identity: GitHubIdentity, remote_url: str, remote_ref: str
) -> None:
    parsed = parse_github_repo(remote_url)
    if parsed is None or not remote_ref.startswith("refs/heads/"):
        return
    repo_owner, repo_name = parsed
    branch = remote_ref.removeprefix("refs/heads/")
    assert_branch_owned(identity, branch)

    repo = f"{repo_owner}/{repo_name}"
    authors = open_pr_authors(repo, repo_owner, branch)
    foreign = sorted(
        author for author in authors if author.lower() != identity.login.lower()
    )
    if foreign:
        raise GuardError(
            f"branch {branch!r} has an open PR owned by {', '.join(foreign)}; "
            f"active account {identity.login!r} may only update its own PRs"
        )


def parse_push_updates(lines: Iterable[str]) -> Iterable[tuple[str, str, str, str]]:
    for line in lines:
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 4:
            raise GuardError(f"cannot parse pre-push update: {line.rstrip()!r}")
        yield fields[0], fields[1], fields[2], fields[3]


def check_push(identity: GitHubIdentity, remote_name: str, remote_url: str) -> None:
    checked: set[str] = set()
    for _local_ref, local_oid, remote_ref, remote_oid in parse_push_updates(sys.stdin):
        check_pr_ownership(identity, remote_url, remote_ref)
        for oid in commit_oids_for_update(remote_name, local_oid, remote_oid):
            if oid not in checked:
                check_commit(identity, oid)
                checked.add(oid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("commit", help="validate the pending commit identity")
    push = subparsers.add_parser("push", help="validate commits and PR ownership")
    push.add_argument("remote_name")
    push.add_argument("remote_url")
    args = parser.parse_args()

    try:
        identity = active_github_identity()
        if args.command == "commit":
            check_pending_commit(identity)
        else:
            check_push(identity, args.remote_name, args.remote_url)
    except GuardError as exc:
        print(f"identity guard: BLOCKED: {exc}", file=sys.stderr)
        print(
            "identity guard: select the correct account with `gh auth switch`, "
            "then run `scripts/install_git_identity_hooks.sh`.",
            file=sys.stderr,
        )
        return 1

    print(f"identity guard: OK ({identity.login})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
