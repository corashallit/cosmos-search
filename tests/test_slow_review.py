"""Release day's review files (scripts/slow_review.py): from a checks directory, a fading list, kept light curves
and a rule to products.json, prompts, the key and, after the verdicts, the reviewed list. No network."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyrage import decrypt, x25519

from cosmos_search.gaia import epoch
from cosmos_search.inject import lightcurve as inj
from cosmos_search.slow import detect, drift, rule_v3, search, summary, vet
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
HOSTS = [BASE + 100 + i for i in range(60)]     # quiet stars of the same sky pixel, on no list


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
        # what the checks read from other surveys: never in a review file
        "same_years": ["seen", "not_seen", None, None], "skymapper_k": [0.9, 0.1, None, None],
        "tess_amp": [0.004, None, None, None],
    }).to_csv(L / "checks" / "checks.csv", index=False)

    gaia = stars.assign(ra=[10.1, 10.2, 10.3, 10.4], dec=20.0, ref_epoch=2017.5, parallax=2.0, parallax_error=0.05,
                        ruwe=1.01, ipd_frac_multi_peak=0)
    gaia.to_parquet(tmp_path / "gaia.parquet", index=False)
    args = ["files", "--checks", str(L / "checks"), "--fading", str(L / "fading.parquet"), "--lightcurves",
            str(lc_dir), "--gaia", str(tmp_path / "gaia.parquet"), "--no-context", "--schema", "gaiadr3",
            "--out", str(tmp_path / "D"), "--fakes", "0"]
    return tmp_path, args


def write_lightcurves(frames, path):
    df = pd.concat(frames, ignore_index=True)
    df["band"] = pd.Categorical(df["band"], categories=["G", "BP", "RP"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, schema=epoch.SCHEMA, preserve_index=False), path)


@pytest.fixture
def planted(run, monkeypatch):
    """What planted fades need besides the run's review inputs: light curves of every star of a patch (the
    thresholds' sample), the run's summaries and reference, catalogue rows of the hosts, an age key for the test,
    and the hosts' NEOWISE and ZTF fetches answered locally."""
    tmp, args = run
    rng = np.random.default_rng(5)
    write_lightcurves([lightcurve(rng, sid, None) for sid in HOSTS], tmp / "S" / "lightcurves" / "h.parquet")
    thr, thr_p, table = rule_v3.load()
    (tmp / "R" / "summary").mkdir()
    parts = []
    for f in [*sorted((tmp / "R" / "lightcurves").glob("*.parquet")), tmp / "S" / "lightcurves" / "h.parquet"]:
        summ, _, _ = search.score_all(pq.read_table(f), table, thr, thr_p)
        parts.append(summ.assign(ref=True))
        summary.write(parts[-1], tmp / "R" / "summary" / f.name, extra=("full", "ref"))
    summ = pd.concat(parts, ignore_index=True)
    assert summ.set_index("source_id").loc[HOSTS, "quiet"].mean() > 0.8
    summary.reference_table(summ).to_parquet(tmp / "L" / "reference.parquet", index=False)
    pd.DataFrame({"id": [DIP, FADE, FAILED, NOT_KEPT]}).to_parquet(tmp / "L" / "flags.parquet", index=False)
    gaia = pd.read_parquet(tmp / "gaia.parquet")
    hosts = pd.DataFrame({"source_id": HOSTS, "phot_g_mean_mag": 18.2, "bp_rp": 0.87,
                          "ra": 10.5 + 0.01 * np.arange(len(HOSTS)), "dec": 20.0, "ref_epoch": 2017.5,
                          "parallax": 1.0, "parallax_error": 0.1, "ruwe": 1.0, "ipd_frac_multi_peak": 0})
    pd.concat([gaia, hosts], ignore_index=True).to_parquet(tmp / "gaia.parquet", index=False)

    ident = x25519.Identity.generate()
    pub = tmp / "test.age.pub"
    pub.write_text(str(ident.to_public()) + "\n")
    monkeypatch.setattr(slow_review, "RECIPIENTS", [pub])
    monkeypatch.setenv("COSMOS_REVEAL_KEY", str(ident))
    mjd = np.concatenate([m + rng.uniform(0, 2, 12) for m in np.arange(56900.0, 60000.0, 180.0)])
    asked = []

    def neowise(g):
        asked.append(float(g["ra"]))        # (a row of an all-numeric frame is float64: its source_id is not exact)
        return pd.DataFrame({"mjd": np.tile(mjd, 2), "band": np.repeat(["W1", "W2"], mjd.size),
                             "flux": 1 + rng.normal(0, 0.01, 2 * mjd.size), "err": 0.01})

    def ztf(g):
        return pd.DataFrame({"oid": 7, "band": "ztf_g", "t": np.linspace(1300, 3000, 200),
                             "mag": 18.0 + rng.normal(0, 0.01, 200), "magerr": 0.01})

    monkeypatch.setattr(slow_review, "host_neowise", neowise)
    monkeypatch.setattr(slow_review, "host_ztf", ztf)
    args = [*args[:-2], "--hosts", str(tmp / "S" / "lightcurves"), "--fakes", "4", "--practice-seed", "3"]
    return tmp, args, ident, asked


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
    assert "2 stars, 1 with a review that counts: 2 on the shortlist (1 of them" in capsys.readouterr().out


def test_drift_is_removed_as_the_scoring_does(run):
    tmp, _ = run
    stars = pd.DataFrame({"source_id": [DIP], "phot_g_mean_mag": [14.2], "bp_rp": [0.8]})
    _, _, table = rule_v3.load()
    ep = slow_review.corrected_epochs(tmp / "R" / "lightcurves", stars, table)
    raw = drift.normalise(pq.read_table(tmp / "R" / "lightcurves" / "a.parquet"))
    cells = drift.star_cells(stars)
    assert set(ep["source_id"]) == {DIP} and len(ep) == len(raw)
    assert np.allclose(ep["frac_corr"], raw["frac"] - drift.drift_at(raw, cells, table))


def test_planted_fades_among_the_files_and_their_result(planted, capsys):
    tmp, args, ident, asked = planted
    d = tmp / "D"
    assert slow_review.main(args) == 0
    packets = json.loads((d / "products.json").read_text())
    rec = json.loads((d / "files.json").read_text())
    assert len(packets) == 6 and rec["files"] == 6 and rec["stars"] == 2 and rec["planted_fades"] == 4
    assert rec["calibrated_format"] and set(rec["tests_in_the_files"]) == set(vet.CALIBRATED_TESTS)

    # the answers exist only encrypted, and the plan's hash is in the clear for the verdict archive
    assert not (d / "key.json").exists() and (d / "key.json.age").read_bytes().startswith(b"age-encryption.org/")
    plan = json.loads((d / "plan.json").read_text())
    assert plan["planted"] == 4 and plan["by_flavour"] == {"grey": 2, "dust": 1, "reemit": 1}
    assert plan["plan_sha256"] == hashlib.sha256((d / "plan.json.age").read_bytes()).hexdigest()
    keys = json.loads(decrypt((d / "key.json.age").read_bytes(), [ident]))
    assert [k["packet_id"] for k in keys] == [p["packet_id"] for p in packets]
    fakes = [k for k in keys if k["kind"] == "fake"]
    hosts = {k["source_id"] for k in fakes}
    assert len(fakes) == 4 and len(hosts) == 4 and hosts <= set(HOSTS) and len(set(asked)) == 4
    assert {k["source_id"] for k in keys if k["kind"] == "candidate"} == {DIP, FADE}
    for k in fakes:
        t = k["truth"]
        assert t["flavour"] == k["label"] and t["snr"] >= slow_review.FAKE_SNR_MIN and k["flagged"] and k["strength"] > 1
        assert t["shape"] in slow_review.slow_cal.SHAPES and t["signal"]["kind"] == "band_scaled"
    sealed_plan = json.loads(decrypt((d / "plan.json.age").read_bytes(), [ident]))["fakes"]
    assert {x["host"] for x in sealed_plan} == hosts and all({"seed", "shape", "flavour"} <= set(x) for x in sealed_plan)

    # nothing in the clear names a host or says which files are fakes
    assert not (d / "work" / "hosts").exists()
    assert set(pd.read_parquet(d / "work" / "gaia.parquet")["source_id"]) <= {DIP, FADE, FAILED, NOT_KEPT}
    assert set(pd.read_csv(d / "left_out.csv")["source_id"]) == {FAILED, NOT_KEPT}
    clear = [f for f in d.rglob("*") if f.is_file() and f.suffix in (".json", ".md", ".txt", ".csv")]
    assert len(clear) >= 11
    for f in clear:
        text = f.read_text()
        assert not any(str(h) in text for h in hosts), f
        assert "fake" not in text.lower() or f.name == "files.json", f
    out = capsys.readouterr().out
    assert not any(str(h) in out for h in HOSTS)

    # a fake's file is a star's file: the same parts, the calibration's four tests, the same bands
    by_id = {p["packet_id"]: p for p in packets}
    star = by_id[next(k["packet_id"] for k in keys if k["source_id"] == DIP)]
    for k in fakes:
        pk = by_id[k["packet_id"]]
        assert set(pk) == set(star) and set(pk["tests"]) == set(vet.CALIBRATED_TESTS)
        assert pk["bands"] == ["G", "BP", "RP", "ztf_g", "W1", "W2"]
        assert set(pk["tests"]["slow_change"]) == set(star["tests"]["slow_change"])
    # nothing the checks read from other surveys reaches any file
    text = (d / "products.json").read_text()
    assert not any(w in text for w in ("skymapper", "same_years", "tess_amp"))

    assert slow_review.main(["launch", "--dir", str(d), "--chunks", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["pending"] == 6

    # the verdicts: three planted fades shortlisted, the review of the fourth failed
    with pytest.raises(CommitError):
        slow_review.main(["table", "--dir", str(d), "--unpublished"])
    fake_ids = [k["packet_id"] for k in fakes]
    verdicts = []
    for p in packets:
        pid = p["packet_id"]
        if pid == fake_ids[0]:
            verdicts.append(failed(pid, "slow-cal-1", "no reply"))
        elif pid in fake_ids:
            verdicts.append(parse(reply(pid, {"unexplained": 0.6, "eclipsing_binary": 0.4}), p, "slow-cal-1"))
        else:
            verdicts.append(parse(reply(pid, {"unexplained": 0.1, "eclipsing_binary": 0.9}), p, "slow-cal-1"))
    commit(verdicts, d / "batch", allow_dirty=True)
    capsys.readouterr()
    assert slow_review.main(["table", "--dir", str(d), "--unpublished"]) == 0
    out = capsys.readouterr().out
    t = pd.read_csv(d / "reviewed.csv")
    assert set(t["source_id"]) == {DIP, FADE} and not t["shortlisted"].any()      # the stars' list holds no fake
    res = json.loads((d / "fakes.json").read_text())
    assert res["planted"] == 4 and res["shortlisted"] == 3 and res["reviews_failed"] == 1
    assert res["consistent_with_calibration"] and res["upper_95_one_sided"] > 0.9
    assert sum(v["planted"] for v in res["by_flavour"].values()) == 4
    ft = pd.read_csv(d / "fakes_reviewed.csv")
    assert set(ft["source_id"]) == hosts and int(ft["shortlisted"].sum()) == 3
    assert "Planted fades: 3 of 4 shortlisted" in out and "Consistent with the calibration" in out

    # a practice batch without fakes replaces it: plain key, no plan, the old prompts out of the launch's way
    assert slow_review.main([*args[:-6], "--fakes", "0", "--overwrite"]) == 0
    assert (d / "key.json").exists() and not (d / "key.json.age").exists() and not (d / "plan.json.age").exists()
    assert len(list((d / "prompts").glob("*.md"))) == 2 and len(list((d / "replaced-1" / "prompts").glob("*.md"))) == 6
    assert (d / "replaced-1" / "batch" / "commit.json").exists() and (d / "replaced-1" / "fakes.json").exists()
    assert not (d / "batch").exists() and not (d / "reviewed.csv").exists()
    assert "practice only" in capsys.readouterr().out


def test_planted_fades_need_hosts(run):
    _, args = run
    with pytest.raises(SystemExit, match="--hosts"):
        slow_review.main(args[:-2])                                   # no --fakes 0: fakes are the default


def test_no_batch_with_fewer_planted_fades_than_asked(planted, monkeypatch):
    tmp, args, _, _ = planted

    def down(g):
        raise ConnectionError("IRSA is down")

    monkeypatch.setattr(slow_review, "host_neowise", down)
    monkeypatch.setattr(slow_review.time, "sleep", lambda s: None)    # no waiting between the attempts
    with pytest.raises(SystemExit, match="only 0 of 4 planted fades"):
        slow_review.main(args)
    d = tmp / "D"
    assert not (d / "products.json").exists() and not (d / "key.json.age").exists() and not (d / "plan.json.age").exists()
    assert not (d / "work" / "hosts").exists()


def test_the_same_change_goes_into_the_other_surveys():
    # a grey dip of 10%: full depth from day 2030 to 2180, nothing before 2000 or after 2210
    sig = inj.BandScaled(inj.Dip(t0=2000.0, covering=0.10, ingress=30, egress=30, flat=150), vet.ratios("grey"))
    mjd = np.repeat(np.arange(56900.0, 60000.0, 180.0), 6)
    clean = pd.DataFrame({"mjd": np.tile(mjd, 2), "band": np.repeat(["W1", "W2"], mjd.size), "flux": 1.0, "err": 0.01})
    nw = slow_review.neowise_with(clean, sig)
    t = nw["mjd"] - slow_review.checks.BTJD_T0_MJD
    deep, out = (t > 2030) & (t < 2180), (t < 2000) | (t > 2210)
    assert deep.sum() == 12 and np.allclose(nw.loc[deep, "flux"], 0.90) and np.allclose(nw.loc[out, "flux"], 1.0)
    assert np.allclose(nw["err"] / nw["flux"], 0.01)
    raw = pd.DataFrame({"oid": np.repeat([7, 8], 300), "band": np.repeat(["ztf_g", "ztf_r"], 300),
                        "t": np.tile(np.linspace(1300, 3000, 300), 2), "mag": np.repeat([18.0, 17.5], 300), "magerr": 0.01})
    zt = slow_review.ztf_with(raw, sig)
    assert list(zt.columns) == ["band", "t", "flux", "err", "group"] and set(zt["group"]) == {"7", "8"}
    deep, out = (zt["t"] > 2030) & (zt["t"] < 2180), (zt["t"] < 2000) | (zt["t"] > 2210)
    assert deep.sum() > 40 and np.allclose(zt.loc[deep, "flux"], 0.90) and np.allclose(zt.loc[out, "flux"], 1.0)
    assert len(slow_review.ztf_with(None, sig)) == 0                  # south of the checks' ZTF limit


def test_number_of_planted_fades_and_the_check_against_the_calibration():
    assert slow_review.n_fakes(300) == 30 and slow_review.n_fakes(1000) == 50
    assert slow_review.fake_slots(30) == {"grey": 15, "dust": 9, "reemit": 6}
    assert slow_review.fake_slots(50) == {"grey": 25, "dust": 15, "reemit": 10}     # the calibration's own counts
    assert sum(slow_review.fake_slots(7).values()) == 7

    def result(n, k):
        ft = pd.DataFrame({"shortlisted": [True] * k + [False] * (n - k), "reviewed": True, "eb_explained": None,
                           "flavour": "grey"})
        return slow_review.fakes_result(ft)["consistent_with_calibration"]

    # docs/vetting.md §16: 20 of 30 or more, 34 of 50 or more
    assert result(30, 20) and not result(30, 19) and result(50, 34) and not result(50, 33) and result(20, 12)
    assert not result(20, 11)
