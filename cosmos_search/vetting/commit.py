"""Commit Stage A verdicts, then reveal the batch's key (docs/blinding.md §3).

A batch's verdicts are written once, hashed, and the hash recorded with the
time and the code commit. Only then does reveal() open the key: the
packet -> candidate map with decoy and control labels and held-out truths. In a
sealed batch the key is age ciphertext made by the truth repo for the reveal
key (env COSMOS_REVEAL_KEY); in a tuning batch it is plaintext JSON. Either
way reveal() refuses to run if the verdict file has changed since its commit.

The hash alone shows the file did not change after commit.json was written, but commit.json is local. To make
"verdicts fixed before the reveal" checkable by anyone, commit(archive=...) also copies both files into the
repository and makes a git commit of them (published.json in the batch records it). That commit has to reach the
remote before reveal() will open the key: reveal() fetches and refuses unless a remote branch contains the commit
and the archived file there has the recorded hash. GitHub keeps every pushed PR commit under refs/pull/N/head and
shows its push time, even after a squash merge, so the order "verdicts pushed, then results" is public.
"""

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .verdict import Verdict


class CommitError(RuntimeError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()


def _git_head(cwd=None):
    try:
        return _git("rev-parse", "HEAD", cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return None


def _existing(path):
    """The path, or its nearest ancestor that exists (an archive directory is created only on commit)."""
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def _repo_root(path):
    return Path(_git("rev-parse", "--show-toplevel", cwd=path))


def _dirty(cwd=None):
    """Tracked files with uncommitted changes (untracked files do not change what HEAD means)."""
    try:
        return _git("status", "--porcelain", "--untracked-files=no", cwd=cwd).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def commit(verdicts, batch_dir, archive=None, allow_dirty=False, extra=None):
    """Write verdicts.jsonl and commit.json in batch_dir. Refuses to overwrite an existing commit, and refuses a
    repository with uncommitted changes to tracked files, since code_commit would then not say what code ran.

    archive: a directory inside a git working tree. Both files are copied there and committed (nothing else is
    staged), and published.json in batch_dir records that commit for reveal() to check once it is pushed.
    extra: more fields for commit.json (for example the sealed plan's sha256)."""
    batch_dir = Path(batch_dir)
    where = _existing(archive) if archive else None
    dirty = _dirty(where)
    if dirty and not allow_dirty:
        raise CommitError("uncommitted changes to tracked files; commit them first so code_commit means something:\n"
                          + "\n".join(dirty[:20]))
    batch_dir.mkdir(parents=True, exist_ok=True)
    vpath, cpath = batch_dir / "verdicts.jsonl", batch_dir / "commit.json"
    if cpath.exists():
        raise CommitError(f"{cpath} exists; a committed batch is immutable, start a new batch")
    with vpath.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(v.to_dict() if isinstance(v, Verdict) else v, sort_keys=True) + "\n")
    record = {"verdicts_sha256": _sha(vpath), "n_verdicts": len(verdicts),
              "code_commit": _git_head(where),
              "committed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              **(extra or {})}
    cpath.write_text(json.dumps(record, indent=2, sort_keys=True))
    if archive:
        record["published"] = _archive(batch_dir, Path(archive))
    return record


def _archive(batch_dir, archive):
    archive.mkdir(parents=True, exist_ok=True)
    root = _repo_root(archive.resolve())
    paths = []
    for name in ("verdicts.jsonl", "commit.json"):
        dest = archive / name
        if dest.exists():
            raise CommitError(f"{dest} exists; archive each batch under its own directory")
        dest.write_bytes((batch_dir / name).read_bytes())
        paths.append(str(dest.resolve().relative_to(root)))
    sha = json.loads((batch_dir / "commit.json").read_text())["verdicts_sha256"]
    _git("add", "--", *paths, cwd=root)
    _git("commit", "--quiet", "-m", f"Stage A verdicts for {archive.name}, before the reveal\n\n"
         f"verdicts.jsonl sha256 {sha}", "--", *paths, cwd=root)
    pub = {"commit": _git("rev-parse", "HEAD", cwd=root), "path": paths[0], "repo": str(root)}
    (batch_dir / "published.json").write_text(json.dumps(pub, indent=2, sort_keys=True))
    return pub


def verify_published(batch_dir, remote="origin"):
    """Raise CommitError unless the batch's archive commit is on `remote` with the committed verdict file.
    Returns the remote refs that contain it."""
    batch_dir = Path(batch_dir)
    pub_path = batch_dir / "published.json"
    if not pub_path.exists():
        raise CommitError("batch was not archived (commit --archive); its verdicts cannot be shown to predate "
                          "the reveal")
    pub = json.loads(pub_path.read_text())
    record = json.loads((batch_dir / "commit.json").read_text())
    root = pub["repo"]
    try:
        _git("fetch", "--quiet", remote, cwd=root)
        refs = _git("branch", "-r", "--contains", pub["commit"], cwd=root).split()
    except subprocess.CalledProcessError as exc:
        raise CommitError(f"could not check {remote}: {exc.stderr.strip()}") from None
    refs = [r for r in refs if r.startswith(f"{remote}/")]
    if not refs:
        raise CommitError(f"archive commit {pub['commit'][:12]} is not on {remote}; push it before revealing")
    blob = subprocess.run(["git", "show", f"{pub['commit']}:{pub['path']}"], capture_output=True, check=True,
                          cwd=root).stdout
    if hashlib.sha256(blob).hexdigest() != record["verdicts_sha256"]:
        raise CommitError("the archived verdict file does not match commit.json")
    return refs


def load_verdicts(batch_dir):
    batch_dir = Path(batch_dir)
    record = json.loads((batch_dir / "commit.json").read_text())
    if _sha(batch_dir / "verdicts.jsonl") != record["verdicts_sha256"]:
        raise CommitError("verdicts.jsonl does not match its commit hash")
    return [Verdict.from_dict(json.loads(line)) for line in (batch_dir / "verdicts.jsonl").read_text().splitlines()]


def reveal(batch_dir, key_path, remote="origin"):
    """Return the batch key as {packet_id: key record}, only if the batch is committed and intact and, unless
    remote is None (offline tuning and tests), its archived verdicts are on the remote."""
    batch_dir, key_path = Path(batch_dir), Path(key_path)
    if not (batch_dir / "commit.json").exists():
        raise CommitError("batch not committed; commit verdicts before revealing")
    load_verdicts(batch_dir)  # hash check
    if remote is not None:
        verify_published(batch_dir, remote)
    raw = key_path.read_bytes()
    if raw.startswith(b"age-encryption.org/"):
        from pyrage import decrypt, x25519
        secret = os.environ.get("COSMOS_REVEAL_KEY", "").strip()
        if not secret:
            raise CommitError("sealed key needs COSMOS_REVEAL_KEY")
        raw = decrypt(raw, [x25519.Identity.from_str(secret)])
    keys = json.loads(raw)
    return {k["packet_id"]: k for k in keys} if isinstance(keys, list) else keys
