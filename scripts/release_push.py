#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


VERSION_RE = re.compile(
    r"^V?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<pre>[0-9A-Za-z][0-9A-Za-z.-]*))?$"
)
_PRE_RELEASE_ALIASES = {"a": "alpha", "alpha": "alpha", "b": "beta", "beta": "beta", "rc": "rc"}
_PEP440_PRE_RELEASE = {"alpha": "a", "beta": "b", "rc": "rc"}
RELEASE_FILES = (
    Path("pyproject.toml"),
    Path("marvis/__init__.py"),
    Path("README.md"),
    Path("README.zh-CN.md"),
    Path("docs/runbook.md"),
    Path("docs/对notebook的要求.md"),
)


def normalize_version(value: str) -> str:
    major, minor, patch, pre = _parse_version(value)
    suffix = f"-{pre}" if pre else ""
    return f"V{major}.{minor}.{patch}{suffix}"


def bump_version_tag(current_tag: str, bump: str) -> str:
    normalized = normalize_version(current_tag)
    if "-" in normalized:
        raise ValueError("bump only supports stable V<MAJOR>.<MINOR>.<PATCH> tags; use --version for pre-releases")
    major, minor, patch = (int(part) for part in normalized[1:].split("."))
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    elif bump == "patch":
        patch += 1
    else:
        raise ValueError("bump must be one of: major, minor, patch")
    return f"V{major}.{minor}.{patch}"


def update_release_text(text: str, old_plain: str, new_plain: str) -> str:
    old_python = python_version(old_plain)
    new_python = python_version(new_plain)
    replacements = (
        (f'version = "{old_plain}"', f'version = "{new_python}"'),
        (f'version = "{old_python}"', f'version = "{new_python}"'),
        (f'__version__ = "{old_plain}"', f'__version__ = "{new_python}"'),
        (f'__version__ = "{old_python}"', f'__version__ = "{new_python}"'),
        (f"current V{old_plain} release", f"current V{new_plain} release"),
        (f"当前 V{old_plain} 版本", f"当前 V{new_plain} 版本"),
        (f"MARVIS 本地运行手册（V{old_plain}）", f"MARVIS 本地运行手册（V{new_plain}）"),
        (f"当前 V{old_plain} 公开版", f"当前 V{new_plain} 公开版"),
        (f"MARVIS V{old_plain} 当前内置", f"MARVIS V{new_plain} 当前内置"),
    )
    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated


def python_version(value: str) -> str:
    major, minor, patch, pre = _parse_version(value)
    base = f"{major}.{minor}.{patch}"
    if not pre:
        return base
    name, number = pre.split(".", 1)
    return f"{base}{_PEP440_PRE_RELEASE[name]}{number}"


def _parse_version(value: str) -> tuple[int, int, int, str | None]:
    match = VERSION_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(
            "version must match V<MAJOR>.<MINOR>.<PATCH> or V<MAJOR>.<MINOR>.<PATCH>-<alpha|beta|rc>.<N>"
        )
    pre = _normalize_prerelease(match.group("pre"))
    return int(match.group("major")), int(match.group("minor")), int(match.group("patch")), pre


def _normalize_prerelease(value: str | None) -> str | None:
    if value is None:
        return None
    parts = value.lower().split(".")
    if len(parts) != 2 or parts[0] not in _PRE_RELEASE_ALIASES or not parts[1].isdigit():
        raise ValueError("pre-release suffix must match alpha.N, beta.N, or rc.N")
    return f"{_PRE_RELEASE_ALIASES[parts[0]]}.{int(parts[1])}"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_output(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def latest_version_tag() -> str:
    tags = git_output("tag", "--list", "V[0-9]*", "--sort=-v:refname").splitlines()
    for tag in tags:
        if VERSION_RE.fullmatch(tag) and "-" not in normalize_version(tag):
            return tag
    raise RuntimeError("no existing V<MAJOR>.<MINOR>.<PATCH> tag found")


def ensure_clean_worktree() -> None:
    if git_output("status", "--short"):
        raise RuntimeError("working tree must be clean before release_push")


def ensure_on_branch(expected_branch: str) -> None:
    current_branch = git_output("branch", "--show-current")
    if current_branch != expected_branch:
        raise RuntimeError(
            f"release_push must run on {expected_branch}; current branch is {current_branch}"
        )


def tag_exists(tag: str) -> bool:
    return bool(git_output("tag", "--list", tag))


def update_release_files(old_tag: str, new_tag: str) -> list[Path]:
    old_plain = old_tag[1:]
    new_plain = new_tag[1:]
    changed: list[Path] = []
    for path in RELEASE_FILES:
        text = path.read_text(encoding="utf-8")
        updated = update_release_text(text, old_plain, new_plain)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)
    if not changed:
        raise RuntimeError(f"no release metadata changed from {old_tag} to {new_tag}")
    return changed


def release_commit_message(tag: str) -> str:
    return f"""Advance MARVIS release to {tag}

Constraint: Public release tags must match tracked release metadata.
Rejected: Move an existing release tag | immutable tags keep published versions auditable.
Confidence: high
Scope-risk: narrow
Directive: Use scripts/release_push.py for release pushes instead of raw git push.
Tested: release_push updated release metadata and created tag {tag}.
Not-tested: Full application regression is expected before invoking release_push.
"""


def create_release_commit(tag: str, changed: list[Path]) -> None:
    run(["git", "add", *[str(path) for path in changed]])
    commit = subprocess.run(
        ["git", "commit", "-F", "-"],
        input=release_commit_message(tag),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    sys.stdout.write(commit.stdout)


def push_release(remote: str, branch: str, tag: str) -> None:
    run(["git", "push", "--atomic", remote, branch, tag])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bump MARVIS release metadata, tag, and push.")
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument("--version", help="Explicit version tag, e.g. V1.0.1")
    version_group.add_argument(
        "--bump",
        choices=("patch", "minor", "major"),
        default="patch",
        help="Bump type based on the latest V tag. Defaults to patch.",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    old_tag = latest_version_tag()
    new_tag = normalize_version(args.version) if args.version else bump_version_tag(old_tag, args.bump)
    if tag_exists(new_tag):
        raise RuntimeError(f"tag already exists: {new_tag}")

    ensure_clean_worktree()
    ensure_on_branch(args.branch)
    if args.dry_run:
        print(f"verified clean worktree on branch {args.branch}")
        print(f"would update release metadata: {old_tag} -> {new_tag}")
        print(f"would create annotated tag: {new_tag}")
        if not args.no_push:
            print(f"would push {args.branch} and {new_tag} to {args.remote}")
        return 0

    changed = update_release_files(old_tag, new_tag)
    create_release_commit(new_tag, changed)
    run(["git", "tag", "-a", new_tag, "-m", f"MARVIS-Agent {new_tag}"])
    if not args.no_push:
        push_release(args.remote, args.branch, new_tag)
    print(f"released {new_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
