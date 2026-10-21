"""Commit Stage A verdicts, then reveal the batch's key (docs/blinding.md §3).

A batch's verdicts are written once, hashed, and the hash recorded with the
time and the code commit. Only then does reveal() open the key: the
packet -> candidate map with decoy and control labels and held-out truths. In a
sealed batch the key is age ciphertext made by the truth repo for the reveal
key (env COSMOS_REVEAL_KEY); in a tuning batch it is plaintext JSON. Either
way reveal() refuses to run if the verdict file has changed since its commit.
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


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def commit(verdicts, batch_dir):
    """Write verdicts.jsonl and commit.json in batch_dir. Refuses to overwrite an existing commit."""
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    vpath, cpath = batch_dir / "verdicts.jsonl", batch_dir / "commit.json"
    if cpath.exists():
        raise CommitError(f"{cpath} exists; a committed batch is immutable, start a new batch")
    with vpath.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(v.to_dict() if isinstance(v, Verdict) else v, sort_keys=True) + "\n")
    record = {"verdicts_sha256": _sha(vpath), "n_verdicts": len(verdicts), "code_commit": _git_head(),
              "committed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    cpath.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def load_verdicts(batch_dir):
    batch_dir = Path(batch_dir)
    record = json.loads((batch_dir / "commit.json").read_text())
    if _sha(batch_dir / "verdicts.jsonl") != record["verdicts_sha256"]:
        raise CommitError("verdicts.jsonl does not match its commit hash")
    return [Verdict.from_dict(json.loads(line)) for line in (batch_dir / "verdicts.jsonl").read_text().splitlines()]


def reveal(batch_dir, key_path):
    """Return the batch key as {packet_id: key record}, only if the batch is committed and intact."""
    batch_dir, key_path = Path(batch_dir), Path(key_path)
    if not (batch_dir / "commit.json").exists():
        raise CommitError("batch not committed; commit verdicts before revealing")
    load_verdicts(batch_dir)  # hash check
    raw = key_path.read_bytes()
    if raw.startswith(b"age-encryption.org/"):
        from pyrage import decrypt, x25519
        secret = os.environ.get("COSMOS_REVEAL_KEY", "").strip()
        if not secret:
            raise CommitError("sealed key needs COSMOS_REVEAL_KEY")
        raw = decrypt(raw, [x25519.Identity.from_str(secret)])
    keys = json.loads(raw)
    return {k["packet_id"]: k for k in keys} if isinstance(keys, list) else keys
