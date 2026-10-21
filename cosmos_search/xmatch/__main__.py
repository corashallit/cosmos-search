"""Cross-match a Gaia target table to one or more surveys.

    python -m cosmos_search.xmatch targets.parquet --surveys allwise 2mass --out xm/
    python -m cosmos_search.xmatch targets.parquet --context --out xm/   # every default survey, one wide table

targets.parquet needs gaia_source columns (source_id, ra, dec, ref_epoch, parallax, pmra, pmdec, their
errors and correlations, phot_g_mean_mag). Writes <survey>_matches.parquet, <survey>_best.parquet and
report.json (or context.parquet and report.json with --context).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .context import DEFAULT_SURVEYS, context
from .run import crossmatch


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.xmatch", description=__doc__.split("\n")[0])
    ap.add_argument("targets")
    ap.add_argument("--surveys", nargs="*", default=list(DEFAULT_SURVEYS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", action="store_true", help="one wide table over all surveys, plus blending")
    ap.add_argument("--no-compete", action="store_true", help="skip competition from Gaia neighbours")
    ap.add_argument("--shifts", type=int, default=2)
    args = ap.parse_args(argv)
    gaia = pd.read_parquet(args.targets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = {"compete": not args.no_compete, "n_shifts": args.shifts}
    if args.context:
        wide, results = context(gaia, args.surveys, **kw)
        wide.to_parquet(out / "context.parquet")
        report = {s: r.report for s, r in results.items()}
    else:
        report = {}
        for s in args.surveys:
            res = crossmatch(gaia, s, **kw)
            res.matches.to_parquet(out / f"{s}_matches.parquet")
            res.best.to_parquet(out / f"{s}_best.parquet")
            report[s] = res.report
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
