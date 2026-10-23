"""Column-by-column check of the data.lsdb.io AllWISE HATS copy against IRSA allwise_p3as_psd.

    python scripts/allwise_hats_check.py report.json

For each cone: every column from HATS and from IRSA, joined on cntr. A column agrees when every row matches
(floats within 1e-6 relative, or 1e-9 absolute; nulls equal nulls). For each disagreeing column we also test
whether it matches IRSA's next column, the signature of the shift reported as astronomy-commons/lsdb#1660
(fixed by LSDB's 2026-10-23 re-import; docs/data-sources.md has the result).
"""

import json
import sys

import lsdb
import numpy as np
import pandas as pd
import pyvo

HATS = "https://data.lsdb.io/hats/wise/allwise/allwise"
IRSA = "https://irsa.ipac.caltech.edu/TAP"
# (name, ra, dec, radius_arcsec): the three xmatch validation fields (highlat holds the issue's 60" example),
# and spots spread over the sky (both poles, both hemispheres, ecliptic pole, Galactic centre region) so
# several partitions are touched.
CONES = [
    ("highlat", 180.0, 30.0, 300),
    ("kepler", 301.5644, 44.4568, 90),
    ("plane", 114.066, -24.876, 45),
    ("south_pole", 10.0, -89.5, 120),
    ("north_pole", 200.0, 89.0, 120),
    ("lmc_edge", 75.0, -66.0, 45),
    ("gal_centre", 266.4, -28.9, 60),
    ("ecl_pole", 270.0, 66.56, 45),
    ("ra0", 0.5, 0.5, 120),
    ("ra90_s", 90.0, -45.0, 120),
    ("ra330_n", 330.0, 20.0, 120),
]


def same(a, b):
    """Elementwise equality with nulls equal and a float tolerance. A blank string counts as null: IRSA serves
    blank string fields as '', the HATS copy stores them as null."""
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    blank = lambda s: (s.astype("string").str.strip() == "").fillna(False).to_numpy(dtype=bool)
    na, nb = a.isna().to_numpy() | blank(a), b.isna().to_numpy() | blank(b)
    out = na & nb
    both = ~na & ~nb
    try:
        fa = pd.to_numeric(a[both], errors="raise").astype(float).to_numpy()
        fb = pd.to_numeric(b[both], errors="raise").astype(float).to_numpy()
        ok = np.isclose(fa, fb, rtol=1e-6, atol=1e-9)
    except (ValueError, TypeError):
        ok = (a[both].astype(str).str.strip().to_numpy() == b[both].astype(str).str.strip().to_numpy())
    out[np.flatnonzero(both)] = ok
    return out


def main(out):
    cat = lsdb.open_catalog(HATS, columns="all")
    cols = list(cat.columns)
    tap = pyvo.dal.TAPService(IRSA)
    report = {"hats": HATS, "hats_creation_date": getattr(cat.hc_structure.catalog_info, "hats_creation_date", None),
              "n_columns": len(cols), "cones": {}}
    for name, ra, dec, r in CONES:
        h = cat.cone_search(ra, dec, r).compute()
        h = h.reset_index() if h.index.name == "_healpix_29" else h.reset_index(drop=True)
        q = ("SELECT * FROM allwise_p3as_psd "
             f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r / 3600})) = 1")
        i = tap.run_sync(q, maxrec=100000).to_table().to_pandas()
        i.columns = [c.lower() for c in i.columns]
        m = h.merge(i, on="cntr", suffixes=("_h", "_i"), how="outer", indicator=True)
        both = m[m["_merge"] == "both"].reset_index(drop=True)
        bad, shifted = [], []
        for k, c in enumerate(cols):
            if c == "cntr":
                continue
            ch = f"{c}_h" if f"{c}_h" in both else c
            ci = f"{c}_i"
            if ci not in both:
                bad.append((c, "missing in IRSA"))
                continue
            eq = same(both[ch], both[ci])
            if not eq.all():
                bad.append((c, int((~eq).sum())))
                if k + 1 < len(cols) and f"{cols[k + 1]}_i" in both and same(both[ch], both[f"{cols[k + 1]}_i"]).all():
                    shifted.append(c)
        report["cones"][name] = {
            "ra": ra, "dec": dec, "radius_arcsec": r, "hats_rows": len(h), "irsa_rows": len(i),
            "only_hats": int((m["_merge"] == "left_only").sum()), "only_irsa": int((m["_merge"] == "right_only").sum()),
            "matched": len(both), "columns_compared": len(cols) - 1,
            "columns_disagreeing": bad, "columns_matching_next_irsa_column": shifted,
            "order4_pixels": sorted({int(p) for p in h["_healpix_29"].to_numpy(dtype=np.int64) >> 50})
            if "_healpix_29" in h else [],
        }
        print(name, report["cones"][name]["matched"], "rows;", len(bad), "columns disagree;", len(shifted), "shifted",
              flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
