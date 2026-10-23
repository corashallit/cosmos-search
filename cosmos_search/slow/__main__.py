"""python -m cosmos_search.slow known --out known.parquet
python -m cosmos_search.slow alert-lcs --known known.parquet --out alert_lcs.parquet"""

import argparse

from cosmos_search.gaia.archive import tap

from . import known


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.slow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("known", help="known answers (Boyajian's star, fading alerts, R CrB / UX Ori / DY Per) "
                                     "matched to Gaia DR3")
    k.add_argument("--out", required=True)
    k.add_argument("--alerts", help="local copy of the Gaia Science Alerts index (default: download it)")
    al = sub.add_parser("alert-lcs", help="Gaia Science Alerts light curves of the fading alerts in a known-answers "
                                          "table")
    al.add_argument("--known", required=True)
    al.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "known":
        text = open(a.alerts).read() if a.alerts else None
        df = known.build(tap(), alerts_text=text)
        df.to_parquet(a.out, index=False)
        print(df.groupby("kind").agg(n=("name", "size"), matched=("source_id", lambda s: int(s.notna().sum()))))
    elif a.cmd == "alert-lcs":
        import pandas as pd

        k = pd.read_parquet(a.known)
        lcs = known.alert_lightcurves(k.loc[k["kind"] == "alert_fade", "name"].tolist())
        lcs.to_parquet(a.out, index=False)
        print(f"{lcs['name'].nunique()} alerts, {len(lcs)} points")


if __name__ == "__main__":
    main()
