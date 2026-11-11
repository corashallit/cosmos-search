"""Release day's review files (scripts/slow_review.py): from a checks directory, a fading list, kept light curves
and a rule to products.json, prompts, the key and, after the verdicts, the reviewed list. No network."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_search.gaia import epoch
from cosmos_search.inject import lightcurve as inj
from cosmos_search.slow import detect, drift, rule_v3, vet
from cosmos_search.vetting import reviewer
from cosmos_search.vetting.commit import CommitError, commit
from cosmos_search.vetting.verdict import failed, parse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slow_review", ROOT / "scripts" / "slow_review.py")
slow_review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slow_review)

# Four stars in one sky pixel: a grey dimming episode, a steady fade, one whose NEOWISE fetch failed in the
# checks, and one whose light curve the run did not keep.
BASE = 690 << 49
DIP, FADE, FAILED, NOT_KEPT = BASE + 11, BASE + 12, BASE + 13, BASE + 14


def lightcurve(rng, sid, sig, n_visits=45, sigma=0.002):
    tv = np.sort(rng.uniform(1700.0, 2700.0, n_visits))
    t = np.repeat(tv, 2) + np.tile([0.0, 0.074], n_visits)
    rows = []
    for name, level in (("G", 1000.0), ("BP", 480.0), ("RP", 620.0)):
        k = sig.factor(t, name) if sig is not None else np.ones_like(t)
        rows.append(pd.DataFrame({
            "source_id": sid, "band": name, "t": t, "flux": level * (k + rng.normal(0, sigma, t.size)),
            "err": level * sigma, "flags": 0, "transit_id": np.arange(t.size) + 1000, "fov": np.tile([1, 2], n_visits),
            "ccd_row": 3, "n_obs": 9}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def run(tmp_path):
    """A release-day run's outputs as the review step finds them."""
    rng = np.random.default_rng(11)
    dip = inj.BandScaled(inj.Dip(t0=2000.0, covering=0.05, ingress=30, egress=30, flat=150), vet.ratios("grey"))
    fade = inj.BandScaled(inj.SecularTrend(rate_per_year=-0.03, t_ref=2200.0), vet.ratios("dust", beta=1.5))
    lcs = {DIP: lightcurve(rng, DIP, dip), FADE: lightcurve(rng, FADE, fade), FAILED: lightcurve(rng, FAILED, dip)}
    lc_dir = tmp_path / "R" / "lightcurves"
    lc_dir.mkdir(parents=True)
    # two files, as the kept light curves of two bulk files
    for name, ids in (("a", (DIP,)), ("b", (FADE, FAILED))):
        df = pd.concat([lcs[i] for i in ids], ignore_index=True)
        df["band"] = pd.Categorical(df["band"], categories=["G", "BP", "RP"])
        pq.write_table(pa.Table.from_pandas(df, schema=epoch.SCHEMA, preserve_index=False), lc_dir / f"{name}.parquet")

    stars = pd.DataFrame({"source_id": [DIP, FADE, FAILED, NOT_KEPT], "phot_g_mean_mag": [14.2, 15.1, 13.0, 16.0],
                          "bp_rp": [0.8, 1.1, 0.6, 0.9]})
    _, _, table = rule_v3.load()
    ep = slow_review.corrected_epochs(lc_dir, stars, table)
    feat = detect.star_features(ep).set_index("source_id")
    rows = []
    for sid, by, strength in ((DIP, "episode", 9.0), (FADE, "trend", 4.0), (FAILED, "episode", 7.0)):
        rows.append({"id": sid, "by": by, "strength": strength, "s_rep": np.nan, "rep_period": np.nan,
                     "depth_bp": 0.04, "depth_bp_err": 0.01, "depth_rp": 0.04, "depth_rp_err": 0.01,
                     "still": False, **feat.loc[sid].to_dict()})
    rows.append({**rows[0], "id": NOT_KEPT, "strength": 3.0})
    fading = pd.DataFrame(rows).merge(stars.rename(columns={"source_id": "id"}), on="id")
    L = tmp_path / "L"
    (L / "checks").mkdir(parents=True)
    fading.to_parquet(L / "fading.parquet", index=False)

    mjd = np.concatenate([m + rng.uniform(0, 2, 12) for m in np.arange(56900.0, 60000.0, 180.0)])
    clean = pd.DataFrame({"mjd": np.tile(mjd, 2), "band": np.repeat(["W1", "W2"], mjd.size),
                          "flux": 1 + rng.normal(0, 0.01, 2 * mjd.size), "err": 0.01})
    pd.concat([clean.assign(source_id=DIP), clean.assign(source_id=FADE)]).to_parquet(L / "checks" / "neowise.parquet")
    pd.DataFrame({"source_id": DIP, "band": "ztf_g", "t": np.linspace(1300, 3000, 200),
                  "flux": 1 + rng.normal(0, 0.01, 200), "err": 0.01}).to_parquet(L / "checks" / "ztf.parquet")
    pd.DataFrame({
        "source_id": [DIP, FADE, FAILED, NOT_KEPT], "strength": [9.0, 4.0, 7.0, 3.0],
        "class": ["RS", None, None, None], "errors": [None, None, "NEOWISE: ConnectionError('reset'); ", None],
        "eb_explained": [True, False, None, None], "eb_period": [31.7, None, None, None],
        "eb_delta_bits": [12.5, 0.4, None, None], "eb_periodic_unphysical": [False, False, None, None],
    }).to_csv(L / "checks" / "checks.csv", index=False)

    gaia = stars.assign(ra=[10.1, 10.2, 10.3, 10.4], dec=20.0, ref_epoch=2017.5, parallax=2.0, parallax_error=0.05,
                        ruwe=1.01, ipd_frac_multi_peak=0)
    gaia.to_parquet(tmp_path / "gaia.parquet", index=False)
    args = ["files", "--checks", str(L / "checks"), "--fading", str(L / "fading.parquet"), "--lightcurves",
            str(lc_dir), "--gaia", str(tmp_path / "gaia.parquet"), "--no-context", "--schema", "gaiadr3",
            "--out", str(tmp_path / "D")]
    return tmp_path, args


def reply(pid, probs):
    return json.dumps({"packet_id": pid, "class_probs": probs, "predictions": {}, "explanation": "x", "evidence": [],
                       "discriminating_test": "y", "recognised_as": None})


def test_review_files_for_the_stars_the_checks_covered(run, capsys):
    tmp, args = run
    d = tmp / "D"
    assert slow_review.main(args) == 0
    packets = json.loads((d / "products.json").read_text())
    keys = json.loads((d / "key.json").read_text())
    assert {k["source_id"] for k in keys} == {DIP, FADE} and [k["packet_id"] for k in keys] == \
        [p["packet_id"] for p in packets] == (d / "packet_ids.txt").read_text().split()
    left = pd.read_csv(d / "left_out.csv").set_index("source_id")["reason"]
    assert "fetch failed" in left[FAILED] and "no light curve" in left[NOT_KEPT]
    assert list((d / "replies").iterdir()) == []                      # the launch script needs it there, empty
    by_star = {k["source_id"]: (k, p) for k, p in zip(keys, packets)}
    key, pk = by_star[DIP]
    # the calibration's four tests and nothing else; the eclipse check waits in the key
    assert set(pk["tests"]) == {"slow_change", "colour_of_change", "neowise_change", "ztf_after"}
    assert key["eclipse"]["eb_explained"] is True and key["eclipse"]["eb_period"] == pytest.approx(31.7)
    assert key["kind"] == "candidate" and key["flagged_by"] == "episode" and key["candidate_id"] == f"gaia_dr3:{DIP}"
    assert pk["bands"] == ["G", "BP", "RP", "ztf_g", "W1", "W2"] and by_star[FADE][1]["bands"] == ["G", "BP", "RP", "W1", "W2"]
    assert abs(pk["tests"]["colour_of_change"]["z_grey"]) < 4          # the grey episode reads as grey
    assert by_star[FADE][1]["tests"]["neowise_change"]["g_slope_per_yr"] == pytest.approx(-0.03, rel=0.3)
    for k, p in zip(keys, packets):
        text = (d / "prompts" / f"{p['packet_id']}.md").read_text()
        assert str(k["source_id"]) not in text and "Reply format" in text and "\n---\n" in text
    assert json.loads((d / "files.json").read_text())["reviewer"] == reviewer.ADOPTED

    # files in place are not replaced unless asked (their reviews would be void)
    with pytest.raises(SystemExit):
        slow_review.main(args)
    capsys.readouterr()
    assert slow_review.main(["launch", "--dir", str(d), "--chunks", "5"]) == 0
    launch = json.loads(capsys.readouterr().out)
    rec = reviewer.record()
    assert launch["pending"] == 2 and len(launch["args"]) == 2
    assert all(x["model"] == rec["model"] and x["effort"] == rec["effort"] and Path(x["dir"]).is_absolute()
               for x in launch["args"])
    assert sorted(i for x in launch["args"] for i in x["ids"]) == sorted(p["packet_id"] for p in packets)

    # with the eclipse check in the files, on request
    assert slow_review.main([*args, "--with-eclipse", "--overwrite"]) == 0
    packets = json.loads((d / "products.json").read_text())
    with_test = [p for p in packets if "eclipsing_binary" in p["tests"]]
    assert len(packets) == 2 and len(with_test) == 2
    assert {p["tests"]["eclipsing_binary"]["explained"] for p in with_test} == {1, 0}
    assert any(p["tests"]["eclipsing_binary"].get("period_d") == pytest.approx(31.7) for p in with_test)


def test_reviewed_list_after_the_verdicts(run, capsys):
    tmp, args = run
    d = tmp / "D"
    slow_review.main(args)
    packets = json.loads((d / "products.json").read_text())
    keys = {k["packet_id"]: k for k in json.loads((d / "key.json").read_text())}
    with pytest.raises(CommitError):                                 # no verdicts committed yet: the key stays shut
        slow_review.main(["table", "--dir", str(d), "--unpublished"])
    verdicts = []
    for p in packets:
        if keys[p["packet_id"]]["source_id"] == DIP:
            verdicts.append(parse(reply(p["packet_id"], {"unexplained": 0.3, "debris_transit": 0.2, "eclipsing_binary": 0.5}),
                                  p, "slow-cal-1"))
        else:
            verdicts.append(failed(p["packet_id"], "slow-cal-1", "no reply"))
    assert verdicts[0].ok or verdicts[1].ok
    commit(verdicts, d / "batch", allow_dirty=True)
    assert slow_review.main(["table", "--dir", str(d), "--unpublished", "--checks", str(tmp / "L" / "checks")]) == 0
    t = pd.read_csv(d / "reviewed.csv").set_index("source_id")
    assert t.loc[DIP, "score"] == pytest.approx(0.5) and bool(t.loc[DIP, "shortlisted"]) and t.loc[DIP, "class"] == "RS"
    assert bool(t.loc[DIP, "eb_explained"]) and t.loc[DIP, "eb_period"] == pytest.approx(31.7)
    # a failed review is not a pass: the star stays on the list to read
    assert not bool(t.loc[FADE, "reviewed"]) and bool(t.loc[FADE, "shortlisted"]) and "no reply" in t.loc[FADE, "review_error"]
    assert "1 of them because the review failed" in capsys.readouterr().out


def test_drift_is_removed_as_the_scoring_does(run):
    tmp, _ = run
    stars = pd.DataFrame({"source_id": [DIP], "phot_g_mean_mag": [14.2], "bp_rp": [0.8]})
    _, _, table = rule_v3.load()
    ep = slow_review.corrected_epochs(tmp / "R" / "lightcurves", stars, table)
    raw = drift.normalise(pq.read_table(tmp / "R" / "lightcurves" / "a.parquet"))
    cells = drift.star_cells(stars)
    assert set(ep["source_id"]) == {DIP} and len(ep) == len(raw)
    assert np.allclose(ep["frac_corr"], raw["frac"] - drift.drift_at(raw, cells, table))
