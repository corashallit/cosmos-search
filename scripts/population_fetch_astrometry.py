"""Fetch positions, parallaxes, Bailer-Jones distances and DPAC classes for all DR3 epoch-photometry sources."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from cosmos_search.gaia import archive

OUT = Path("/home/user/pop/aux")
N = 24
EDGES = [int(i * (12 * 4**12 * 2**35) // N) for i in range(N + 1)]  # equal source_id ranges = equal sky areas
Q = """SELECT v.source_id, g.ra, g.dec, g.l, g.b, g.parallax, g.parallax_error, g.pmra, g.pmdec,
 d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo, c.best_class_name
FROM gaiadr3.vari_summary AS v
JOIN gaiadr3.gaia_source AS g ON g.source_id = v.source_id
LEFT JOIN external.gaiaedr3_distance AS d ON d.source_id = v.source_id
LEFT JOIN gaiadr3.vari_classifier_result AS c ON c.source_id = v.source_id
WHERE v.source_id >= {lo} AND v.source_id < {hi}"""

def one(i):
    p = OUT / f"astrom_{i:02d}.parquet"
    if p.exists():
        return i, "cached"
    t = archive.tap().query(Q.format(lo=EDGES[i], hi=EDGES[i + 1]), cache=False)
    tab = pa.table({c: np.asarray(t[c].filled(np.nan) if hasattr(t[c], "filled") and t[c].dtype.kind == "f" else t[c]) for c in t.colnames if c != "best_class_name"})
    cls = [str(x) if x is not None and str(x) != "--" else "" for x in t["best_class_name"]]
    tab = tab.append_column("best_class_name", pa.array(cls))
    pq.write_table(tab, p, compression="zstd")
    return i, len(t)

with ThreadPoolExecutor(4) as ex:
    for i, r in ex.map(one, range(N)):
        print(i, r, flush=True)
