"""Write truth tables so that plaintext never touches disk or stdout."""

import hashlib
import json
from pathlib import Path

from pyrage import encrypt, x25519


def load_recipients(paths):
    recipients = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                recipients.append(x25519.Recipient.from_str(line))
    if not recipients:
        raise SystemExit(f"no age recipients found in {paths}")
    return recipients


def write_sealed(obj, out_path, recipient_files):
    """Serialise obj to JSON in memory and write it encrypted to the recipients."""
    plaintext = json.dumps(obj, sort_keys=True).encode()
    ciphertext = encrypt(plaintext, load_recipients(recipient_files))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(ciphertext)
    return sha256(out_path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
