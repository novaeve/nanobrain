#!/usr/bin/env python3
"""Prove that an old nanoBrain source can be retired without losing anything.

A source is retirable only when everything it holds is reachable from the
unified repository's target ref:

  git source   every branch and tag is an ancestor of TARGET; no uncommitted
               change, no stash, no untracked non-ignored file.
  plain dir    every non-ignored file has an identical blob at the same path
               in ARCHIVE_REF, and ARCHIVE_REF is an ancestor of TARGET.

Ignored files (results, runtime, venvs, caches) are never covered by git.
They are reported separately and turn the verdict into
RETIRABLE_AFTER_DATA_MOVE; this tool never declares them disposable.

Usage:
  retire_check.py --repo UNIFIED --source PATH [--archive-ref REF]
                  [--target main] [--manifest out.json]
Exit code 0 only for RETIRABLE.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def git(*args, cwd=None, env=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, check=False)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r


def out(*args, **kw):
    return git(*args, **kw).stdout.decode()


def is_ancestor(repo, sha, target):
    if git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo, check=False).returncode:
        return False
    return git("merge-base", "--is-ancestor", sha, target, cwd=repo, check=False).returncode == 0


def check_git(repo, src, target, file_list=None):
    problems, refs = [], {}
    for line in out(
        "for-each-ref", "--format=%(objectname) %(refname)", "refs/heads", "refs/tags", cwd=src
    ).splitlines():
        sha, name = line.split(" ", 1)
        sha = out("rev-parse", f"{sha}^{{commit}}", cwd=src).strip()
        ok = is_ancestor(repo, sha, target)
        refs[name] = {"commit": sha, "reachable": ok}
        if not ok:
            problems.append(f"{name} {sha[:7]} is not an ancestor of {target}")
    # Tracked changes only: this stats tracked files and never walks untracked directories.
    status = out("status", "--porcelain", "--untracked-files=no", cwd=src).splitlines()
    if status:
        problems.append(f"{len(status)} uncommitted tracked paths")
    stashes = out("stash", "list", cwd=src).splitlines()
    if stashes:
        problems.append(f"{len(stashes)} stash entries")
    untracked, ignored = [], []
    if file_list is None:
        problems.append("untracked files not checked: pass --file-list from the NAS index")
    else:
        tracked = set(out("ls-files", "-z", cwd=src).split("\0"))
        others = [p for p in read_list(file_list) if p not in tracked and p != ".git" and not p.startswith(".git/")]
        if others:
            r = subprocess.run(
                ["git", "check-ignore", "-z", "--stdin"],
                cwd=src,
                input="\0".join(others).encode(),
                capture_output=True,
                check=False,
            )
            ignored = [p for p in r.stdout.decode().split("\0") if p]
        # The index is a snapshot: a path deleted since is not a loss. Stat only these few candidates.
        untracked = sorted(p for p in set(others) - set(ignored) if os.path.lexists(os.path.join(src, p)))
        if untracked:
            problems.append(f"{len(untracked)} untracked non-ignored files")
    return (
        {"kind": "git", "refs": refs, "dirty": status[:50], "untracked": untracked[:50], "stashes": stashes},
        problems,
        ignored,
    )


def blob_sha(path):
    """Git blob id of the exact bytes on disk; a symlink is its link text, as git stores it."""
    try:
        data = os.readlink(path).encode() if os.path.islink(path) else Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_list(path):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def check_dir(repo, src, archive_ref, target, file_list=None, ignored_list=None):
    problems = []
    if not archive_ref:
        return {"kind": "dir"}, ["plain directory needs --archive-ref"], []
    if not is_ancestor(repo, out("rev-parse", archive_ref, cwd=repo).strip(), target):
        problems.append(f"{archive_ref} is not an ancestor of {target}")
    if file_list:
        # On the shared NAS the file list comes from the path index, never from a directory walk.
        files, ignored = read_list(file_list), read_list(ignored_list) if ignored_list else []
    else:
        gitdir = out("rev-parse", "--absolute-git-dir", cwd=repo).strip()
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
            base = ["--git-dir", gitdir, "--work-tree", src]
            files = out(*base, "ls-files", "-z", "--others", "--exclude-standard", env=env).split("\0")
            ignored = out(
                *base, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", env=env
            ).splitlines()
    files = [f for f in files if f]
    archived = {}
    for line in out("ls-tree", "-r", "-z", archive_ref, cwd=repo).split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            archived[path] = meta.split()[2]
    missing = [p for p in files if archived.get(p) != blob_sha(os.path.join(src, p))]
    if missing:
        problems.append(f"{len(missing)} files differ from or are absent in {archive_ref}")
    return {"kind": "dir", "archive_ref": archive_ref, "files": len(files), "missing": missing[:200]}, problems, ignored


def data_summary(src, ignored):
    total, tops = 0, {}
    for rel in ignored:
        path = os.path.join(src, rel.rstrip("/"))
        size = 0
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for n in names:
                    try:
                        size += os.lstat(os.path.join(root, n)).st_size
                    except OSError:
                        pass
        elif os.path.lexists(path):
            size = os.lstat(path).st_size
        total += size
        top = rel.split("/", 1)[0]
        tops[top] = tops.get(top, 0) + size
    return {"paths": len(ignored), "bytes": total, "by_top_level": dict(sorted(tops.items(), key=lambda kv: -kv[1]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--archive-ref")
    ap.add_argument("--target", default="main")
    ap.add_argument("--manifest")
    ap.add_argument("--skip-data-size", action="store_true", help="list ignored paths without walking them")
    ap.add_argument(
        "--file-list", help="paths under SOURCE from the NAS index (no directory walk); plain dir: the kept paths"
    )
    ap.add_argument("--ignored-list", help="plain dir: ignored paths from the NAS index")
    a = ap.parse_args()
    if a.file_list:
        a.skip_data_size = True
    repo, src = os.path.abspath(a.repo), os.path.abspath(a.source)
    is_git = git("rev-parse", "--is-inside-work-tree", cwd=src, check=False).returncode == 0 and os.path.realpath(
        out("rev-parse", "--show-toplevel", cwd=src).strip()
    ) == os.path.realpath(src)
    report, problems, ignored = (
        check_git(repo, src, a.target, a.file_list)
        if is_git
        else check_dir(repo, src, a.archive_ref, a.target, a.file_list, a.ignored_list)
    )
    data = {"paths": len(ignored), "list": ignored[:100]} if a.skip_data_size else data_summary(src, ignored)
    verdict = "BLOCKED" if problems else ("RETIRABLE_AFTER_DATA_MOVE" if ignored else "RETIRABLE")
    result = {
        "source": src,
        "target": a.target,
        "verdict": verdict,
        "problems": problems,
        "uncovered_data": data,
        **report,
    }
    if a.manifest:
        with open(a.manifest, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps({k: result[k] for k in ("source", "verdict", "problems")}, ensure_ascii=False))
    print(json.dumps({"uncovered_data": {k: v for k, v in data.items() if k != "list"}}, ensure_ascii=False))
    sys.exit(0 if verdict == "RETIRABLE" else 1)


if __name__ == "__main__":
    main()
