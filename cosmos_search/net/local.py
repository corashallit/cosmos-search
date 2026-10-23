"""Where a machine keeps its local copies of external data: HATS mirrors, the solar-system ephemeris, and a large
persistent response cache (docs/storage.md, "Local data store").

The root is COSMOS_DATA_DIR, else ~/cosmos-data when it exists (Cora's Mac, kore). Cloud containers have neither,
so everything there behaves as before: remote reads and the small cache under ~/.cache.
"""

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "cosmos-data"
PERSISTENT_CACHE_GB = 100.0


def data_dir():
    """The local data root, or None when this machine has none."""
    root = os.environ.get("COSMOS_DATA_DIR")
    if root:
        return Path(root).expanduser()
    return DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.is_dir() else None


def persistent():
    """Whether responses fetched here should be kept for later runs (the machine has a local data store)."""
    return data_dir() is not None
