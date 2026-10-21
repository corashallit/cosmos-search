"""Blind-run guard (docs/blinding.md §2).

With COSMOS_BLIND set to anything but "" or "0", every request and every cache read is refused. The cache
holds un-injected archive data, so serving it during a blind run would leak as much as the network.
"""

import os


class BlindModeError(RuntimeError):
    pass


def blind():
    return os.environ.get("COSMOS_BLIND", "") not in ("", "0")


def check_blind():
    if blind():
        raise BlindModeError("COSMOS_BLIND is set: archive access and the response cache are disabled")
