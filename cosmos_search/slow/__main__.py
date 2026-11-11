"""python -m cosmos_search.slow known --out known.parquet [--mirror aip]
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
    k.add_argument("--mirror", help="match positions on a mirror of gaiadr3.gaia_source (a name in "
                                    "gaia.epoch.MIRRORS, or a TAP URL) when ESA's archive refuses queries")
    k.add_argument("--workers", type=int, default=4, help="cone queries at once")
    al = sub.add_parser("alert-lcs", help="Gaia Science Alerts light curves of the fading alerts in a known-answers "
                                          "table")
    al.add_argument("--known", required=True)
    al.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "known":
        text = open(a.alerts).read() if a.alerts else None
        if a.mirror:
            from cosmos_search.gaia.epoch import MIRRORS
            from cosmos_search.net.tap import Tap

            service = Tap(MIRRORS.get(a.mirror, a.mirror), job_slots=a.workers)
        else:
            service = tap()
        df = known.build(service, alerts_text=text, workers=a.workers, log=lambda s: print(s, flush=True))
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
