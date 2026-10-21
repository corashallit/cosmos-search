"""Target queries CLI.

    python -m cosmos_search.targets run --release dr3 --out data/targets/dr3
    python -m cosmos_search.targets freeze --tag dr4-day-one --out configs/targets/dr4-day-one.json
    python -m cosmos_search.targets verify configs/targets/dr4-day-one.json
    python -m cosmos_search.targets check --release dr4
"""

import argparse
import json
import sys

from . import prefilter, releases


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gaia dimmer target queries")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--release", required=True, choices=sorted(releases.RELEASES))
    r.add_argument("--out", required=True)
    r.add_argument("--alpha", type=float, default=prefilter.Rule.alpha)
    f = sub.add_parser("freeze")
    f.add_argument("--tag", required=True)
    f.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    c = sub.add_parser("check")
    c.add_argument("--release", required=True, choices=sorted(releases.RELEASES))
    a = ap.parse_args(argv)

    if a.cmd == "freeze":
        print(json.dumps(prefilter.freeze(a.tag, a.out), indent=2))
        return 0
    if a.cmd == "verify":
        diffs = prefilter.verify(a.manifest)
        print("\n".join(diffs) if diffs else "unchanged")
        return 1 if diffs else 0

    from cosmos_search.gaia import archive

    if a.cmd == "check":
        missing = releases.check_release(a.release, archive.tap())
        print("\n".join(missing) if missing else f"{a.release}: all required columns present")
        return 1 if missing else 0
    rule = prefilter.Rule(alpha=a.alpha)
    _, manifest = prefilter.run(a.release, archive.tap(), a.out, rule)
    print(json.dumps(manifest["selected"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
