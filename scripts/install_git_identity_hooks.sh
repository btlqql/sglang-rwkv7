#!/bin/sh
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

login=$(gh api user --jq .login)
user_id=$(gh api user --jq .id)

git config --local user.name "$login"
git config --local user.email "${user_id}+${login}@users.noreply.github.com"
git config --local core.hooksPath .githooks

chmod +x .githooks/pre-commit .githooks/pre-push scripts/git_identity_guard.py
python3 scripts/git_identity_guard.py commit

printf 'Installed identity hooks for %s <%s>\n' \
    "$login" "${user_id}+${login}@users.noreply.github.com"
