"""The thresholds step (scripts/slow_release_thresholds.py) on a machine whose worker processes start empty.

On Linux a worker is a fork of the parent and sees what the parent put in the script's module state. On macOS a
worker is a new interpreter that imports the script again, so the state has to be handed to it. The practice on
kore (2026-11-11) stopped here with KeyError: 'drift_share'. No network."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.gaia import epoch

ROOT = Path(__file__).resolve().parents[1]

# Runs a script's main() with the start method macOS uses. The script is imported by name, so a worker imports it
# afresh, as it does on macOS.
DRIVER = """
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.path.insert(0, sys.argv[1])
    sys.exit(__import__(sys.argv[2]).main(sys.argv[3:]))
"""


def lightcurves(rng, ids, n_visits=40, sigma=0.003):
    """Steady stars over a range of brightness, two fields of view a visit, in the kept light curves' layout."""
    tv = np.sort(rng.uniform(1700.0, 2700.0, n_visits))
    t = np.repeat(tv, 2) + np.tile([0.0, 0.074], n_visits)
    parts = []
    for name, rel in (("G", 1.0), ("BP", 0.48), ("RP", 0.62)):
        level = rel * rng.uniform(300.0, 30000.0, len(ids))
        flux = level[:, None] * (1 + rng.normal(0, sigma, (len(ids), t.size)))
        parts.append(pd.DataFrame({
            "source_id": np.repeat(ids, t.size), "band": name, "t": np.tile(t, len(ids)), "flux": flux.ravel(),
            "err": np.repeat(level * sigma, t.size), "flags": 0,
            "transit_id": np.tile(np.arange(t.size) + 1000, len(ids)),
            "fov": np.tile(np.tile([1, 2], n_visits), len(ids)), "ccd_row": 3, "n_obs": 9}))
    df = pd.concat(parts, ignore_index=True).sort_values(["source_id", "band", "t"], kind="stable")
    df["band"] = pd.Categorical(df["band"], categories=["G", "BP", "RP"])
    return df


def test_thresholds_with_workers_that_start_empty(tmp_path):
    rng = np.random.default_rng(5)
    lc = tmp_path / "lightcurves"
    lc.mkdir()
    for k in range(3):
        ids = ((690 + k) << 49) + np.arange(1, 151, dtype=np.int64)
        pq.write_table(pa.Table.from_pandas(lightcurves(rng, ids), schema=epoch.SCHEMA, preserve_index=False),
                       lc / f"f{k}.parquet")
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    run = subprocess.run([sys.executable, str(driver), str(ROOT / "scripts"), "slow_release_thresholds",
                          "--lightcurves", str(lc), "--out", str(tmp_path / "RULE"), "--workers", "2"],
                         capture_output=True, text=True, timeout=240, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads((tmp_path / "RULE" / "thresholds.json").read_text())
    assert pq.read_metadata(tmp_path / "RULE" / "stats.parquet").num_rows == 450
