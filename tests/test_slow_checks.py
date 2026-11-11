"""scripts/slow_checks.py and ATLAS. ATLAS answers about one star an hour, so the checks ask it last and only about
the strongest stars: every file is written before the first ATLAS request and again after each answer, the stars
go in order of strength, and --ids keeps a run to listed stars (the review's shortlist). No network."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cosmos_search import surveys

SLOPE = -0.03                       # the fade Gaia measured, as a fraction per year
T_FIRST, T_LAST = 0.0, 1800.0       # Gaia's first and last visit, days


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sc():
    return _script("slow_checks")


def fading_epochs(rng, source_id, survey, band, group, lo, hi, err=0.005):
    """Epochs of one survey every four days that follow Gaia's fade."""
    t = np.arange(lo, hi, 4.0)
    flux = 1 + SLOPE * (t - T_FIRST) / 365.25 + rng.normal(0, err, len(t))
    return pd.DataFrame({"source_id": source_id, "survey": survey, "band": band, "group": group, "t": t,
                         "flux": flux, "err": err, "mag": np.nan})[surveys.COLUMNS]


def star(source_id, strength, g=15.0, dec=-40.0):
    return SimpleNamespace(source_id=source_id, strength=strength, ra=120.0, dec=dec, phot_g_mean_mag=g, by="trend",
                           slope=SLOPE, t_first=T_FIRST, t_last=T_LAST)


def test_atlas_epochs_join_the_surveys_read_earlier(sc, monkeypatch):
    rng = np.random.default_rng(3)
    r = star(7, 5.0)
    have = fading_epochs(rng, 7, "skymapper", "sm_r", "g1", -100.0, 1900.0)
    asked = []

    def fetch_all(s, names, **options):
        asked.append((list(names), options))
        return fading_epochs(rng, 7, "atlas", "atlas_o", "01a_o", 400.0, 3700.0), {}

    monkeypatch.setattr(sc.surveys, "fetch_all", fetch_all)
    monkeypatch.setattr(sc, "gaia_query", lambda adql: pd.DataFrame(
        {"source_id": [7], "ra": [120.0], "dec": [-40.0], "phot_g_mean_mag": [15.0]}))
    cols, lc, sectors, err = sc.other_surveys(r, None, ["atlas"], "gaiadr4", False, have=have, atlas={"cache": True})
    assert asked == [(["atlas"], {"atlas": {"cache": True}})] and err == "" and sectors is None
    assert set(lc["survey"]) == {"skymapper", "atlas"} and len(lc) == len(have) + len(np.arange(400.0, 3700.0, 4.0))
    # both surveys saw Gaia's fade over Gaia's years; only ATLAS has the years since, where the fade went on
    assert cols["skymapper_same"] == "seen" and cols["atlas_same"] == "seen" and cols["same_years"] == "seen"
    assert cols["atlas_after"] == "went_on" == cols["after"]
    assert cols["atlas_dilution"] == 0.0 and cols["skymapper_dilution"] == 0.0
    # with nothing read earlier, ATLAS is compared alone
    alone, lc, _, _ = sc.other_surveys(r, None, ["atlas"], "gaiadr4", False)
    assert set(lc["survey"]) == {"atlas"} and "skymapper_same" not in alone and alone["atlas_same"] == "seen"


def test_atlas_goes_strongest_first_and_a_failure_keeps_the_row(sc, monkeypatch):
    stars = [star(i, s) for i, s in ((11, 9.0), (12, 7.0), (13, 4.0))]
    row_of = {r.source_id: {"source_id": r.source_id, "strength": r.strength, "errors": ""} for r in stars}
    row_of[13]["errors"] = "ZTF: timeout; "
    lc_of, order, saved, lines = {11: "earlier"}, [], [], []

    def other_surveys(r, z, names, schema, with_tess, have=None, **options):
        order.append((r.source_id, have))
        if r.source_id == 12:
            raise RuntimeError("ATLAS task failed: disk full")
        return {"atlas_same": "seen", "atlas_after": "stopped"}, f"epochs of {r.source_id}", None, \
            "neighbours: timeout; " if r.source_id == 13 else ""

    monkeypatch.setattr(sc, "other_surveys", other_surveys)
    sc.atlas_step(stars, row_of, lc_of, "gaiadr4", lambda: saved.append(sorted(k for k, v in row_of.items()
                                                                                if "atlas_s" in v)),
                  at_once=1, log=lambda text, flush=False: lines.append(text))
    assert order == [(11, "earlier"), (12, None), (13, None)]
    assert saved == [[11], [11, 12], [11, 12, 13]]                       # written again after every answer
    assert row_of[11]["atlas_same"] == "seen" and row_of[11]["atlas_error"] == "" == row_of[11]["errors"]
    assert "atlas_same" not in row_of[12] and "disk full" in row_of[12]["atlas_error"] and row_of[12]["errors"] == ""
    assert row_of[13]["errors"] == "ZTF: timeout; " and row_of[13]["atlas_error"] == "neighbours: timeout; "
    # a later run does the two with an error again (for 13 ATLAS's answer then comes from the cache)
    assert [k for k, v in row_of.items() if sc.unanswered(v)] == [12, 13]
    assert lc_of == {11: "epochs of 11", 13: "epochs of 13"}
    assert lines[0].startswith("ATLAS 1 of 3: 11 ") and "since stopped" in lines[0] and "no_data" in lines[1]


def test_a_stop_ends_the_waiting_requests_and_keeps_the_rows(sc, monkeypatch):
    stars = [star(i, s) for i, s in ((21, 9.0), (22, 7.0), (23, 4.0))]
    row_of = {r.source_id: {"source_id": r.source_id, "strength": r.strength, "errors": ""} for r in stars}
    asked, ended, lines = [], [], []

    def other_surveys(r, z, names, schema, with_tess, have=None, **options):
        asked.append(r.source_id)
        if r.source_id == 21:
            return {"atlas_same": "seen"}, None, None, ""
        try:
            while True:                        # the reader looking at ATLAS's queue until the task has run
                options["atlas"]["sleep"](0.01)
        except sc.Stopped:
            ended.append(r.source_id)          # where the reader takes its task off the queue
            raise

    def save():
        raise KeyboardInterrupt                # the stop arrives once the first answer is in

    monkeypatch.setattr(sc, "other_surveys", other_surveys)
    with pytest.raises(KeyboardInterrupt):
        sc.atlas_step(stars, row_of, {}, "gaiadr4", save, at_once=1, log=lambda text, flush=False: lines.append(text))
    assert asked == [21, 22] and ended == [22]             # the third star was never sent
    assert row_of[21]["atlas_same"] == "seen" and row_of[22] == {"source_id": 22, "strength": 7.0, "errors": ""}
    assert lines[-1].startswith("Stopped")


def test_a_stop_takes_the_waiting_request_off_the_atlas_queue(sc, tmp_path, monkeypatch):
    """The same with the real ATLAS reader against a local server whose task never runs."""
    import _thread
    import json
    import time

    from test_net import FakeArchive, make_client

    from cosmos_search.surveys import atlas

    monkeypatch.setenv(atlas.TOKEN_ENV, "abc123")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    archive, client, real, stopped = FakeArchive(), make_client(tmp_path), surveys.fetch_all, []
    js = {"Content-Type": "application/json"}
    archive.script["/queue/"] = [(201, js, json.dumps({"url": f"{archive.url}/queue/1/"}).encode())]
    archive.script["/queue/1/"] = [(200, js, json.dumps({"finishtimestamp": None}).encode())]

    def fetch_all(s, names, **options):
        pause = options["atlas"]["sleep"]

        def sleep(seconds):
            if not stopped:                    # the reader is waiting on the queue: now the run is stopped
                stopped.append(True)
                time.sleep(0.05)
                _thread.interrupt_main()
            pause(seconds)

        return real(s, names, atlas={**options["atlas"], "sleep": sleep, "client": client, "base_url": archive.url,
                                     "poll": 0.01})

    monkeypatch.setattr(sc.surveys, "fetch_all", fetch_all)
    stars = [star(31, 9.0), star(32, 7.0)]
    row_of = {r.source_id: {"source_id": r.source_id, "strength": r.strength, "errors": ""} for r in stars}
    try:
        with pytest.raises(KeyboardInterrupt):
            sc.atlas_step(stars, row_of, {}, "gaiadr4", lambda: None, at_once=1, log=lambda text, flush=False: None)
    finally:
        archive.close()
    # one task was sent and then deleted; the second star was never sent
    assert [m for m in archive.methods if m[0] != "GET"] == [("POST", "/queue/"), ("DELETE", "/queue/1/")]
    assert row_of[31] == {"source_id": 31, "strength": 9.0, "errors": ""}


@pytest.fixture
def run(sc, tmp_path, monkeypatch):
    """A fading list of five stars and the checks' fetches replaced: every star gets SkyMapper epochs, and ATLAS
    answers with epochs after looking at what is on disk at that moment."""
    rng = np.random.default_rng(4)
    # the strongest star is too bright for ATLAS (G 11), the faintest too faint (G 19.5)
    g_of = {1: 11.0, 2: 15.0, 3: 16.0, 4: 17.0, 5: 19.5}
    pd.DataFrame({"id": list(g_of), "strength": [9.0, 8.0, 7.0, 6.0, 5.0], "by": "trend",
                  "phot_g_mean_mag": list(g_of.values()), "bp_rp": 0.9}).to_parquet(tmp_path / "fading.parquet")
    out, calls = tmp_path / "checks", []

    def gaia_rows(ids, schema):
        t = pd.DataFrame({c: np.nan for c in sc.GAIA_COLS}, index=range(len(ids))).assign(source_id=np.asarray(ids))
        return t.assign(ra=120.0, dec=-40.0, phot_g_mean_mag=[g_of[int(i)] for i in ids], best_class_name=None)

    def check_one(r, irsa, g=None, names=(), schema="gaiadr4", with_tess=False):
        calls.append(("checks", r.source_id, tuple(names)))
        row = {"source_id": r.source_id, "by": r.by, "strength": r.strength, "class": None, "errors": "",
               "neowise_s": 1.0, "same_years": "seen"}
        return (row, pd.DataFrame({"source_id": [r.source_id], "mjd": [57000.0]}), None,
                fading_epochs(rng, r.source_id, "skymapper", "sm_r", "g1", -100.0, 1900.0), None)

    def other_surveys(r, z, names, schema, with_tess, have=None, **options):
        on_disk = pd.read_csv(out / "checks.csv")
        calls.append(("atlas", r.source_id, tuple(names), options, len(on_disk),
                      set(pd.read_parquet(out / "surveys.parquet")["source_id"]) == set(g_of), have is not None))
        lc = pd.concat([have, fading_epochs(rng, r.source_id, "atlas", "atlas_o", "01a_o", 400.0, 3700.0)])
        return {"atlas_same": "seen", "atlas_after": "went_on", "after": "went_on"}, lc, None, ""

    monkeypatch.setattr(sc, "gaia_rows", gaia_rows)
    monkeypatch.setattr(sc, "check_one", check_one)
    monkeypatch.setattr(sc, "other_surveys", other_surveys)
    monkeypatch.setattr(sc, "Tap", lambda *a, **k: None)
    monkeypatch.setattr(sc.surveys, "usable", lambda names, **o: (list(names), {}))
    return SimpleNamespace(sc=sc, dir=tmp_path, out=out, calls=calls,
                           args=["--fading", str(tmp_path / "fading.parquet"), "--out", str(out), "--no-tess"])


def test_checks_are_written_before_atlas_is_asked_and_after_each_answer(run, capsys):
    run.sc.main(run.args + ["--surveys", "skymapper", "--atlas-n", "2"])
    first = [c for c in run.calls if c[0] == "checks"]
    assert sorted(c[1] for c in first) == [1, 2, 3, 4, 5] and {c[2] for c in first} == {("skymapper",)}
    atlas = [c for c in run.calls if c[0] == "atlas"]
    # the two strongest stars ATLAS covers (star 1 is too bright for it), asked for ATLAS alone with its answers
    # kept, each with all five stars' rows and the other surveys' epochs already on disk
    assert sorted(c[1] for c in atlas) == [2, 3]
    assert all(c[2] == ("atlas",) and c[3]["atlas"]["cache"] is True and callable(c[3]["atlas"]["sleep"])
               and c[4:] == (5, True, True) for c in atlas)
    res = pd.read_csv(run.out / "checks.csv")
    assert res["source_id"].tolist() == [1, 2, 3, 4, 5]
    assert res.loc[res["atlas_same"] == "seen", "source_id"].tolist() == [2, 3] and res["atlas_s"].notna().sum() == 2
    assert res["same_years"].tolist() == ["seen"] * 5 and res["after"].notna().sum() == 2
    lc = pd.read_parquet(run.out / "surveys.parquet")
    assert set(lc.loc[lc["survey"] == "atlas", "source_id"]) == {2, 3}
    assert set(lc.loc[lc["survey"] == "skymapper", "source_id"]) == {1, 2, 3, 4, 5}
    assert not list(run.out.glob("*.tmp"))
    text = capsys.readouterr().out
    assert text.index("the steps that read them can start") < text.index("ATLAS: 2 stars") < text.index("ATLAS 2 of 2")


def test_atlas_is_not_asked_unless_a_number_of_stars_is_given(run):
    run.sc.main(run.args)
    assert [c for c in run.calls if c[0] == "atlas"] == []
    assert {c[2] for c in run.calls} == {tuple(s for s in surveys.DEFAULT if s != "atlas")}
    assert "atlas_same" not in pd.read_csv(run.out / "checks.csv")
    with pytest.raises(SystemExit):
        run.sc.main(run.args + ["--surveys", "ztf,atlas"])


def test_atlas_carries_on_from_the_checks_already_written(run):
    args = run.args + ["--surveys", "skymapper"]
    run.sc.main(args + ["--atlas-n", "1"])                     # star 2 answered
    before = pd.read_csv(run.out / "checks.csv")
    neowise = (run.out / "neowise.parquet").stat().st_mtime_ns
    run.calls.clear()
    # the shortlist of a review: stars 4, 3 and 1 (too bright for ATLAS); nothing but ATLAS is fetched
    pd.DataFrame({"source_id": [4, 3, 1, 2], "shortlisted": [True, True, True, False]}).to_csv(
        run.dir / "reviewed.csv", index=False)
    assert run.sc.main(run.args + ["--atlas-only", "--atlas-n", "100", "--ids", str(run.dir / "reviewed.csv"),
                                   "--ids-where", "shortlisted"]) == 0
    assert sorted(c[1] for c in run.calls) == [3, 4] and {c[0] for c in run.calls} == {"atlas"}
    assert all(c[4:] == (5, True, True) for c in run.calls)
    res = pd.read_csv(run.out / "checks.csv")
    assert res["source_id"].tolist() == [1, 2, 3, 4, 5]
    assert res.loc[res["atlas_same"] == "seen", "source_id"].tolist() == [2, 3, 4]
    kept = [c for c in before.columns if c not in ("atlas_same", "atlas_after", "after", "atlas_s", "atlas_error")]
    pd.testing.assert_frame_equal(res[kept], before[kept])       # the other columns come back as they were
    lc = pd.read_parquet(run.out / "surveys.parquet")
    assert set(lc.loc[lc["survey"] == "atlas", "source_id"]) == {2, 3, 4}
    assert set(lc.loc[lc["survey"] == "skymapper", "source_id"]) == {1, 2, 3, 4, 5}
    assert lc.groupby(["source_id", "survey"]).size().max() == len(np.arange(400.0, 3700.0, 4.0))   # none twice
    assert (run.out / "neowise.parquet").stat().st_mtime_ns == neowise
    # a second run has nothing left to ask among those stars; without a list, the one star left that ATLAS covers
    run.calls.clear()
    run.sc.main(run.args + ["--atlas-only", "--atlas-n", "100", "--ids", str(run.dir / "reviewed.csv")])
    assert run.calls == []
    with pytest.raises(SystemExit):
        run.sc.main(run.args + ["--atlas-only"])


def test_a_failed_atlas_request_is_asked_again(run, monkeypatch):
    run.sc.main(run.args + ["--surveys", "skymapper"])
    answer = run.sc.other_surveys

    def fails(r, z, names, schema, with_tess, have=None, **options):
        return {}, have, None, "atlas: TimeoutError: not finished after 21600 s; "

    monkeypatch.setattr(run.sc, "other_surveys", fails)
    run.sc.main(run.args + ["--atlas-only", "--atlas-n", "1"])
    res = pd.read_csv(run.out / "checks.csv")
    assert res.loc[res["atlas_error"].notna(), "source_id"].tolist() == [2] and res["errors"].isna().all()
    monkeypatch.setattr(run.sc, "other_surveys", answer)
    run.calls.clear()
    run.sc.main(run.args + ["--atlas-only", "--atlas-n", "1"])
    assert [c[1] for c in run.calls] == [2]
    res = pd.read_csv(run.out / "checks.csv")
    assert res["atlas_error"].isna().all() and res.loc[res["atlas_same"] == "seen", "source_id"].tolist() == [2]


def test_a_run_can_be_kept_to_the_stars_a_review_shortlisted(run):
    pd.DataFrame({"source_id": [4, 2, 5, 77], "shortlisted": [True, True, False, True]}).to_csv(
        run.dir / "reviewed.csv", index=False)
    run.sc.main(run.args + ["--ids", str(run.dir / "reviewed.csv"), "--ids-where", "shortlisted", "--atlas-n", "5"])
    assert pd.read_csv(run.out / "checks.csv")["source_id"].tolist() == [2, 4]
    assert sorted(c[1] for c in run.calls if c[0] == "atlas") == [2, 4]
    run.calls.clear()
    run.sc.main(run.args + ["--ids", str(run.dir / "reviewed.csv"), "--n", "2"])        # every listed star, cut by --n
    assert pd.read_csv(run.out / "checks.csv")["source_id"].tolist() == [2, 4]
    with pytest.raises(SystemExit):
        run.sc.main(run.args + ["--ids-where", "shortlisted"])
