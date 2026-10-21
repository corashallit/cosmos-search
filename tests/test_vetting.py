import json
import re
import subprocess

import numpy as np
import pandas as pd
import pytest
from pyrage import encrypt, x25519

from cosmos_search.inject.classes.vetting_decoys import DECOY_SPECS, make_decoy, vetting_batch
from cosmos_search.vetting import coincidence, design, quantify, scoring, taxonomy
from cosmos_search.vetting.anonymise import ecliptic_latitude, galactic_latitude, leak_scan, scrub_source
from cosmos_search.vetting.batch import build_batch, harmonise, simple_dips
from cosmos_search.vetting.candidate import Candidate
from cosmos_search.vetting.commit import CommitError, commit, load_verdicts, reveal, verify_published
from cosmos_search.vetting.packet import LeakError, build
from cosmos_search.vetting.report import score
from cosmos_search.vetting.verdict import parse


def gaia_like(cid="gaia_dr3:4567890123456789", dip_at=None, seed=0, n=120):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2457000.0, 2458000.0, n))
    lcs = []
    for band, noise in (("G", 0.002), ("BP", 0.004), ("RP", 0.003)):
        f = 1 + rng.normal(0, noise, n)
        if dip_at is not None:
            f[np.argmin(np.abs(t - dip_at))] *= 0.9
        lcs.append({"band": band, "time": t, "flux": f, "flux_err": np.full(n, noise),
                    "meta": {"fov": rng.integers(1, 3, n)}})
    return Candidate.from_dict({
        "candidate_id": cid, "arm": "gaia", "lightcurves": lcs,
        "source": {"source_id": 4567890123456789, "ra": 301.56, "dec": 44.46, "phot_g_mean_mag": 11.73,
                   "bp_rp": 0.55, "parallax": 2.2, "parallax_error": 0.02, "ruwe": 1.01,
                   "phot_bp_n_obs": n, "phot_bp_n_blended_transits": 0, "phot_rp_n_obs": n,
                   "phot_rp_n_blended_transits": 0},
        "scores": {"surprisal": 1e-6}, "tests": {"fov_invariance": {"statistic": 0.3, "p": 0.7}},
        "context": {"ir_excess_sigma": 0.4},
    })


# --- anonymisation ----------------------------------------------------------


def test_sky_frames_match_known_values():
    assert galactic_latitude(266.405, -28.936) == pytest.approx(0, abs=0.1)        # Sgr A*
    assert galactic_latitude(192.859, 27.128) == pytest.approx(90, abs=0.1)        # north galactic pole
    assert ecliptic_latitude(270.0, 66.56) == pytest.approx(90, abs=0.1)           # north ecliptic pole


def test_scrub_removes_identifiers_and_position():
    s = scrub_source(gaia_like().source)
    assert "source_id" not in s and "ra" not in s and "dec" not in s and "parallax" not in s
    assert s["phot_g_mean_mag"] == 11.7
    assert s["distance_pc"] == 450
    assert "abs_galactic_latitude_deg" in s


@pytest.mark.parametrize("text", ["KIC 8462852", "TIC 123456", "Gaia DR3 2072351234567890", "Gaia17bpp",
                                  "J203724.5+442012", "t=2457123.5", "ra = 301.5", "2016MNRAS.457.3988B",
                                  "Boyajian's star"])
def test_leak_scan_catches(text):
    assert leak_scan(text)


def test_leak_scan_passes_ordinary_packet_text():
    assert leak_scan("G band, 120 epochs, median flux 1.0003, depth 0.104, |b| 5 to 10 deg") == []


# --- packets ------------------------------------------------------------------


def test_packet_is_rezeroed_and_clean():
    packet, key = build(gaia_like(dip_at=2457500.0))
    assert key["candidate_id"].startswith("gaia_dr3")
    # Every rendered time is on the packet clock (the light curve spans ~1000 d), never a JD.
    assert not re.search(r"(?<![\d.])24[5-6]\d{4}\.\d", packet["text"])
    t_col = [float(line.split()[0]) for line in packet["text"].splitlines() if re.match(r"^\d+\.\d{4} ", line)]
    assert t_col and max(t_col) < 1001
    assert packet["summaries"]["G"]["n_epochs"] == 120
    assert leak_scan(packet["text"].replace(packet["packet_id"], "")) == []


def test_packet_refuses_leaky_context():
    cand = gaia_like()
    cand.context["note"] = "near KIC 8462852"
    with pytest.raises(LeakError):
        build(cand)


def test_holdouts_withhold_data_and_record_truth():
    cand = gaia_like(dip_at=2457100.0)
    packet, key = build(cand, holdout=("tail", "colour", "ir"))
    assert packet["bands"] == ["G"]
    assert {t["id"] for t in packet["targets"]} == {"tail_dip", "tail_min_flux", "dip_colour_ratio", "ir_excess"}
    assert "ir_excess_sigma" not in packet["context"]
    assert packet["scores"] == {} and packet["tests"] == {}
    assert key["truths"]["ir_excess"] is False
    assert key["truths"]["dip_colour_ratio"] == pytest.approx(1, abs=0.2)  # injected grey dip
    assert packet["summaries"]["G"]["span_days"] < 0.85 * 1000
    for t in packet["targets"]:
        assert all(float(x) < 1001 for x in re.findall(r"t = (-?[\d.]+) d", t["question"]))


def test_batch_harmonises_fields_and_shuffles():
    real = gaia_like(dip_at=2457300.0)
    host = gaia_like(cid="gaia_dr3:1", seed=3)
    host.scores = {}
    entries = [(real, {"kind": "real"}), (host, {"kind": "decoy", "label": "unexplained"})]
    packets, keys = build_batch(entries, np.random.default_rng(0))
    assert all(p["scores"] == {} for p in packets)  # surprisal missing from one, so dropped from all
    assert {k["kind"] for k in keys} == {"real", "decoy"}
    assert all("kind" not in p for p in packets)


# --- decoys ---------------------------------------------------------------------


@pytest.mark.parametrize("spec", sorted(DECOY_SPECS))
def test_every_decoy_spec_injects(spec):
    host = gaia_like(seed=5)
    detect = None if spec in ("secular_fade", "step") else simple_dips  # not what a dip detector is for
    cand, extra, truth = make_decoy(host, spec, np.random.default_rng(1), max_tries=3000, detect=detect)
    assert extra["kind"] == "decoy" and extra["label"] in taxonomy.LEAVES
    assert truth["host"] == host.candidate_id
    assert truth["visibility"]["max_dev"] >= 4
    assert any(not np.allclose(a.flux, b.flux) for a, b in zip(cand.lightcurves, host.lightcurves))


def test_vetting_batch_generator_keeps_decoy_identity_out_of_packets(tmp_path):
    hosts = tmp_path / "hosts.jsonl"
    hosts.write_text("\n".join(json.dumps(gaia_like(cid=f"gaia_dr3:{i}", seed=i).to_dict()) for i in range(5)))
    ctrl = gaia_like(cid="kic:8462852", seed=9, dip_at=2457600.0).to_dict()
    ctrl.update(label="debris_transit", aliases=["KIC 8462852", "Boyajian's star"])
    controls = tmp_path / "controls.jsonl"
    controls.write_text(json.dumps(ctrl))
    cfg = {"hosts": str(hosts), "controls": str(controls), "holdout": ["tail"],
           "decoys": {"grey_primes": 2, "chromatic_flip": 2}}
    packets, truth, keys = vetting_batch(cfg, np.random.default_rng(2))
    assert len(packets) == len(keys) == 5 and len(truth) == 4
    blob = json.dumps(packets)
    assert "decoy" not in blob and "gaia_dr3" not in blob and "8462852" not in blob
    assert sum(k["candidate_id"] == "decoy" for k in keys) == 4


# --- verdicts, commit, reveal, report -------------------------------------------


def reply(pid, probs, preds=None, rec=None):
    return json.dumps({"packet_id": pid, "class_probs": probs, "predictions": preds or {},
                       "explanation": "x", "recognised_as": rec})


def test_parse_normalises_and_rejects():
    packet, _ = build(gaia_like(), holdout=("ir",))
    v = parse("```json\n" + reply(packet["packet_id"], {"eclipsing_binary": 0.6, "unexplained": 0.42},
                                  {"ir_excess": 1.4}) + "\n```", packet)
    assert v.ok and sum(v.class_probs.values()) == pytest.approx(1) and v.predictions["ir_excess"] == 1.0
    assert not parse(reply(packet["packet_id"], {"aliens": 1.0}), packet).ok
    assert not parse("I think it is an eclipsing binary.", packet).ok
    assert not parse(reply("other", {"unexplained": 1.0}), packet).ok


def test_commit_then_reveal_sealed(tmp_path, monkeypatch):
    packet, key = build(gaia_like())
    key.update(kind="control", label="eclipsing_binary")
    ident = x25519.Identity.generate()
    sealed = tmp_path / "reveal.json.age"
    sealed.write_bytes(encrypt(json.dumps([key]).encode(), [ident.to_public()]))
    batch = tmp_path / "batch"
    v = parse(reply(packet["packet_id"], {"eclipsing_binary": 1.0}), packet)
    with pytest.raises(CommitError):
        reveal(batch, sealed, remote=None)
    commit([v], batch, allow_dirty=True)
    with pytest.raises(CommitError):
        commit([v], batch, allow_dirty=True)
    monkeypatch.setenv("COSMOS_REVEAL_KEY", str(ident))
    with pytest.raises(CommitError):  # not archived, so not provably fixed before the reveal
        reveal(batch, sealed)
    keys = reveal(batch, sealed, remote=None)
    assert keys[packet["packet_id"]]["label"] == "eclipsing_binary"
    (batch / "verdicts.jsonl").write_text((batch / "verdicts.jsonl").read_text().replace("1.0", "0.9"))
    with pytest.raises(CommitError):
        load_verdicts(batch)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_reveal_needs_archived_verdicts_on_the_remote(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "--quiet", "--bare", str(remote))
    _git(tmp_path, "clone", "--quiet", str(remote), str(work))
    for k, val in (("user.email", "t@example.org"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(work, "config", k, val)
    (work / "README").write_text("x\n")
    _git(work, "add", "README")
    _git(work, "commit", "--quiet", "-m", "init")
    _git(work, "push", "--quiet", "origin", "HEAD:main")
    packet, key = build(gaia_like())
    key.update(kind="control", label="eclipsing_binary")
    plain = tmp_path / "key.json"
    plain.write_text(json.dumps([key]))
    v = parse(reply(packet["packet_id"], {"eclipsing_binary": 1.0}), packet)
    (work / "README").write_text("changed\n")
    with pytest.raises(CommitError, match="uncommitted"):
        commit([v], tmp_path / "b0", archive=work / "prereg" / "b0")
    _git(work, "checkout", "--", "README")
    batch = tmp_path / "batch"
    rec = commit([v], batch, archive=work / "prereg" / "batch")
    assert rec["published"]["path"] == "prereg/batch/verdicts.jsonl"
    with pytest.raises(CommitError, match="not on origin"):
        reveal(batch, plain)
    _git(work, "push", "--quiet", "origin", "HEAD:main")
    assert verify_published(batch) == ["origin/main"]
    assert packet["packet_id"] in reveal(batch, plain)


def test_report_measures_confabulation_and_recognition():
    keys, verdicts = {}, []
    for i in range(20):
        packet, key = build(gaia_like(seed=i))
        pid = packet["packet_id"]
        if i < 10:
            key.update(kind="decoy", role="incoherent", label="unexplained", acceptable=["unexplained", "instrumental"])
            probs = {"yso_dipper": 0.9, "unexplained": 0.1} if i < 4 else {"processing": 0.7, "unexplained": 0.3}
            rec = None
        else:
            key.update(kind="control", label="eclipsing_binary", aliases=["V1234 Cyg"])
            probs, rec = {"eclipsing_binary": 0.8, "unexplained": 0.2}, ("V1234 Cyg" if i == 10 else None)
        keys[pid] = key
        verdicts.append(parse(reply(pid, probs, rec=rec), packet))
    rep = score(verdicts, keys)
    assert rep["confabulation_rate"]["k"] == 4 and rep["confabulation_rate"]["n"] == 10
    assert rep["recognition"]["correct"]["k"] == 1
    assert rep["control_false_unexplained"]["k"] == 0
    assert rep["labelled"]["accuracy"] == pytest.approx(16 / 20)


# --- statistics -------------------------------------------------------------------


def test_rate_bounds():
    assert scoring.rate_upper(0, 300) == pytest.approx(1 - 0.05 ** (1 / 300), rel=1e-6)
    lo, hi = scoring.rate_interval(5, 100)
    assert lo < 0.05 < hi
    assert design.n_zero_failure(0.01) == 299


def test_betainc_matches_closed_forms():
    assert scoring.betainc(1, 1, 0.3) == pytest.approx(0.3)
    assert scoring.betainc(2, 1, 0.3) == pytest.approx(0.09)
    assert scoring.betainc(0.5, 0.5, 0.5) == pytest.approx(0.5)


def test_isotonic_is_monotone_and_recalibrates_overconfidence():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0, 1, 2000)
    y = rng.random(2000) < p_true
    stated = p_true ** 0.3                     # overconfident toward 1
    iso = scoring.Isotonic(stated, y)
    grid = np.linspace(0, 1, 50)
    assert np.all(np.diff(iso(grid)) >= 0)
    assert scoring.ece(iso(stated), y) < scoring.ece(stated, y) / 2


def test_auc_and_permutation():
    rng = np.random.default_rng(0)
    lab = rng.random(400) < 0.5
    assert scoring.auc(lab + rng.normal(0, 0.5, 400), lab) > 0.8
    assert scoring.auc_permutation_p(rng.normal(size=400), lab, n_perm=300) > 0.01


def test_label_shift_recovers_mix():
    rng = np.random.default_rng(1)
    C = np.array([[0.9, 0.07, 0.03], [0.1, 0.85, 0.05], [0.1, 0.2, 0.7]])
    q = np.array([0.7, 0.25, 0.05])
    true = rng.choice(3, 20000, p=q)
    dec = np.array([rng.choice(3, p=C[c]) for c in true])
    assert quantify.em_label_shift(C, dec) == pytest.approx(q, abs=0.02)


def test_shortlist_false_interval_contains_truth():
    rng = np.random.default_rng(2)
    C = np.array([[0.9, 0.07, 0.03], [0.08, 0.9, 0.02], [0.1, 0.2, 0.7]])
    q_true = rng.choice(3, 1000, p=[0.6, 0.39, 0.01])
    d = np.array([rng.choice(3, p=C[c]) for c in q_true])
    ctrl = np.repeat([0, 1, 2], [200, 150, 50])
    cd = np.array([rng.choice(3, p=C[c]) for c in ctrl])
    est = quantify.shortlist_false(ctrl, cd, d, 3, 2, n_boot=300, rng=rng)
    truth = int(np.sum((d == 2) & (q_true != 2)))
    assert est["interval"][0] <= truth <= est["interval"][1]


def test_allocation_and_sequential_check():
    n = design.neyman_allocation(300, [0.6, 0.39, 0.01], [0.03, 0.02, 0.3])
    assert n.sum() == 300 and n[0] > n[1] and min(n) >= 10
    assert design.sequential_check(0, 600, 0.01)[0] == "accept"
    assert design.sequential_check(10, 100, 0.01)[0] == "reject"
    assert design.sequential_check(1, 50, 0.01)[0] == "continue"


# --- stage 0 coincidence ---------------------------------------------------------


def test_coincidence_flags_shared_dips():
    rng = np.random.default_rng(0)
    dips = [{"candidate_id": f"c{i}", "t_min": float(rng.uniform(0, 1000))} for i in range(300)]
    dips += [{"candidate_id": f"s{i}", "t_min": 500.0 + 0.01 * i} for i in range(8)]   # shared event
    res = coincidence.coincidence(dips, tau=0.5, window=50)
    flagged = coincidence.flag_candidates(res, 0.01)
    assert {f"s{i}" for i in range(8)} <= set(flagged)
    assert sum(c.startswith("c") for c in flagged) <= 2


def test_agent_definitions_match_prompts():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for agent, src in (("expert-blind", "stage_a.md"), ("expert-reconcile", "stage_b.md")):
        body = (root / ".claude/agents" / f"{agent}.md").read_text().split("---\n", 2)[2].strip()
        assert body == (root / "cosmos_search/vetting" / src).read_text().strip()


def test_cli_prompts_commit_score(tmp_path):
    from cosmos_search.vetting.__main__ import main as cli
    packet, key = build(gaia_like())
    key.update(kind="control", label="eclipsing_binary")
    (tmp_path / "products.json").write_text(json.dumps([packet]))
    (tmp_path / "key.json").write_text(json.dumps([key]))
    cli(["prompts", "--packets", str(tmp_path / "products.json"), "--out", str(tmp_path / "prompts")])
    assert "Reply format" in (tmp_path / "prompts" / f"{packet['packet_id']}.md").read_text()
    (tmp_path / "replies").mkdir()
    (tmp_path / "replies" / f"{packet['packet_id']}.txt").write_text(reply(packet["packet_id"], {"eclipsing_binary": 1}))
    cli(["commit", "--packets", str(tmp_path / "products.json"), "--replies", str(tmp_path / "replies"),
         "--vetter", "test", "--batch", str(tmp_path / "batch"), "--allow-dirty"])
    cli(["score", "--batch", str(tmp_path / "batch"), "--key", str(tmp_path / "key.json"), "--out", str(tmp_path / "r.json"),
         "--unpublished"])
    assert json.loads((tmp_path / "r.json").read_text())["labelled"]["accuracy"] == 1.0


def test_audit_catches_a_field_only_decoys_lack():
    from cosmos_search.vetting.distinguish import audit
    packets, planted = [], []
    for i in range(60):
        c = gaia_like(seed=i)
        if i % 2:
            c.scores = {}          # decoys built without the detector: a giveaway
        packets.append(build(c)[0])
        planted.append(bool(i % 2))
    assert audit(packets, planted, n_perm=200)["auc"] > 0.9
    fair = [build(gaia_like(seed=100 + i))[0] for i in range(60)]
    assert audit(fair, planted, n_perm=200)["p"] > 0.01


def test_test_descriptions_are_rendered():
    cand = gaia_like()
    cand.tests["z_bkg"] = {"statistic": 4.2, "description": "in-dip background excess in its own scatter"}
    packet, _ = build(cand)
    assert packet["tests"]["z_bkg"]["description"].startswith("in-dip")
    assert "in-dip background excess" in packet["text"]


def test_tail_dip_target_skipped_for_large_amplitude_stars():
    cand = gaia_like()
    for lc in cand.lightcurves:
        lc.flux = 1 + 0.4 * np.sin(lc.time / 30.0)
    packet, key = build(cand, holdout=("tail",))
    assert [t["id"] for t in packet["targets"]] == ["tail_min_flux"]
    assert "withheld" in packet["targets"][0]["question"]


def test_decoys_are_visible_before_the_tail_cut():
    from cosmos_search.inject.classes.vetting_decoys import visibility
    from cosmos_search.inject import lightcurve as lcinj
    host = gaia_like(seed=7)
    t0, t1 = host.time_span()
    cut = t0 + 0.8 * (t1 - t0)
    for spec in ("grey_dip", "step", "chromatic_flip", "single_band_dip"):
        detect = None if spec == "step" else simple_dips
        cand, extra, truth = make_decoy(host, spec, np.random.default_rng(3), t_cut=cut, max_tries=3000, detect=detect)
        assert truth["visibility"]["max_dev"] >= 4 and truth["visibility"]["n_dev3"] >= 2
    # A dip placed entirely in the withheld tail is not visible.
    hidden = lcinj.Dip(t0=t1 - 5, covering=0.5, ingress=1, egress=1, flat=3)
    assert not visibility(host, hidden, None, cut, "rare_natural")[0]


def test_anomaly_probes_must_defeat_every_single_period():
    from cosmos_search.inject.classes.vetting_decoys import single_period_fits
    t = np.sort(np.random.default_rng(0).uniform(0, 1000, 150))
    per = 37.3
    in_dip = t[np.abs(((t - t[5]) / per + 0.5) % 1 - 0.5) * per < 0.3]
    out = np.setdiff1d(t, in_dip)
    assert single_period_fits(in_dip, out, 0.3, 1000)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    ev = 10 + 4.0 * np.cumsum(primes)
    t2 = np.sort(np.concatenate([t, ev]))
    in2 = ev[ev < 1000]
    assert not single_period_fits(in2, np.setdiff1d(t2, in2), 0.3, 1000)


def test_anomaly_probe_decoys_pass_the_identifiability_check():
    host = gaia_like(seed=11, n=150)
    cand, extra, truth = make_decoy(host, "grey_primes", np.random.default_rng(5), max_tries=3000)
    assert extra["role"] == "anomaly_probe" and truth["visibility"]["n_events"] >= 4


def test_random_packet_ids_never_trip_the_leak_scan():
    cand = gaia_like()
    for _ in range(300):
        packet, _ = build(cand)
        assert leak_scan(packet["text"]) == []
    assert leak_scan("source 4567890123456789") == ["long_integer"]


def test_injection_reaches_every_flux_channel_with_photon_noise():
    from cosmos_search.inject.classes.vetting_decoys import inject_candidate
    from cosmos_search.inject import lightcurve as lcinj
    host = gaia_like(seed=2, n=4000)
    for lc in host.lightcurves:
        lc.meta["flux_small_aperture"] = lc.flux.copy()
        lc.meta["flux_small_aperture_err"] = lc.flux_err.copy()
    t0, t1 = host.time_span()
    dip = lcinj.Dip(t0=0.5 * (t0 + t1), covering=0.75, ingress=1, egress=1, flat=0.6 * (t1 - t0))
    cand = inject_candidate(host, dip, rng=np.random.default_rng(0))
    g, g0 = cand.lightcurves[0], host.lightcurves[0]
    k = dip.factor(g.time, "G")
    inside = k < 0.3
    assert inside.sum() > 1000
    # Every channel dips; errors follow sqrt(k), not k.
    assert np.allclose(g.meta["flux_small_aperture"][inside].mean(), g.flux[inside].mean(), rtol=0.01)
    assert np.allclose(g.flux_err[inside], g0.flux_err[inside] * np.sqrt(k[inside]))
    assert np.allclose(g.meta["flux_small_aperture_err"][inside], g0.flux_err[inside] * np.sqrt(k[inside]))
    # In-dip scatter matches the quoted errors (photon-limited), as for a real dimming.
    pull = (g.flux[inside] - k[inside]) / g.flux_err[inside]
    assert 0.9 < np.std(pull) < 1.1


def test_single_period_test_allows_an_eccentric_secondary():
    from cosmos_search.inject.classes.vetting_decoys import single_period_fits
    t = np.sort(np.random.default_rng(1).uniform(0, 1000, 400))
    per, w = 41.7, 0.3
    ph = ((t - 3.0) / per) % 1 * per
    d = np.minimum(np.minimum(ph, per - ph), np.abs(ph - 0.31 * per))
    # w is the full duration; epochs on ingress/egress are neither clearly in nor clearly out, as in visibility().
    assert single_period_fits(t[d < 0.35 * w], t[d > 0.5 * w], w, 1000)


def test_decoy_signals_can_continue_into_the_withheld_tail():
    host = gaia_like(seed=7, n=150)
    t0, t1 = host.time_span()
    cut = t0 + 0.8 * (t1 - t0)
    tail = host.lightcurves[0].time[host.lightcurves[0].time >= cut]
    in_tail = 0
    for seed in range(8):
        cand, extra, truth = make_decoy(host, "periodic_grey", np.random.default_rng(seed), t_cut=cut)
        in_tail += np.any(cand.lightcurves[0].flux[-len(tail):] < 0.97)
    assert in_tail >= 1


def test_chromatic_flip_decoys_show_both_halves():
    host = gaia_like(seed=4, n=150)
    cand, extra, truth = make_decoy(host, "chromatic_flip", np.random.default_rng(2), max_tries=3000)
    bp, rp = (next(lc for lc in cand.lightcurves if lc.band == b) for b in ("BP", "RP"))
    i = np.argmin(bp.flux)
    assert bp.flux[i] < 1 - 3 * 0.004 and rp.flux[i] > 1 + 3 * 0.003


def test_decoys_trigger_the_candidate_detector():
    host = gaia_like(seed=9, n=150)
    for spec in ("grey_dip", "dust_dip", "single_band_dip"):
        cand, extra, truth = make_decoy(host, spec, np.random.default_rng(4))
        assert simple_dips(cand)
        if spec == "single_band_dip":
            assert truth["bands"] == ["G"]


def test_period_floor_from_duration_matches_the_sun():
    from cosmos_search.vetting import physics
    r, m = physics.stellar_params({"phot_g_mean_mag": 4.67, "bp_rp": 0.82, "parallax": 100.0, "parallax_error": 0.1})
    assert 0.9 < r < 1.1 and 0.85 < m < 1.15
    assert 6.0 < physics.min_period_days(1.0, 1.0, 1.0, e_max=0.0) / 365.25 < 6.6
    assert 0.8 < physics.min_period_days(106.5 / 1440, 1.0, 1.0, e_max=0.0) < 1.0
    assert physics.stellar_params({"phot_g_mean_mag": 15.0, "bp_rp": 1.0, "parallax": 0.1, "parallax_error": 0.1}) is None


def test_duration_tension_probes_recur_inside_their_period_floor():
    host = gaia_like(seed=12, n=400)
    cand, extra, truth = make_decoy(host, "duration_tension", np.random.default_rng(6), max_tries=5000)
    v = truth["visibility"]
    assert extra["role"] == "anomaly_probe" and v["n_events"] >= 2 and v["d_obs"] >= 0.25
    assert 3 * v["spacing"] < v["period_floor"]


def test_decoys_must_pass_the_queue_veto():
    host = gaia_like(seed=9, n=150)
    seen = []

    def veto(cand):  # rejects the first two otherwise acceptable draws
        seen.append(1)
        return ["systematic"] if len(seen) <= 2 else []

    make_decoy(host, "grey_dip", np.random.default_rng(4), veto=veto)
    assert len(seen) == 3
    with pytest.raises(RuntimeError):
        make_decoy(host, "grey_dip", np.random.default_rng(4), max_tries=50, veto=lambda c: ["always"])


# --- Stage 0 ----------------------------------------------------------------


def _one_epoch(cand, i, factors):
    """Multiply epoch i of each band by its factor (G, BP and RP share epochs in gaia_like)."""
    for lc in cand.lightcurves:
        lc.flux[i] *= factors.get(lc.band, 1.0)
    return cand


def test_stage0_flags_a_g_only_dip_and_passes_coherent_ones():
    from cosmos_search.vetting.stage0 import band_consistency, clean, veto
    i = 40
    cases = {
        "band_inconsistent": {"G": 0.9},  # G dips, BP and RP flat
        "consistent": {"G": 0.9, "BP": 0.9, "RP": 0.9},  # grey
        "colour_fault": {"BP": 3.0},  # BP brightens threefold at an otherwise flat transit
    }
    for want, f in cases.items():
        cand = _one_epoch(gaia_like(seed=1), i, f)
        t = cand.lightcurves[0].time[i]
        (row,) = band_consistency(cand, [{"t_min": t, "depth": 0.1}])
        assert row["verdict"] == want, (want, row)
    # A chromatic sign flip, BP down and RP up with G between, is no fault.
    flip = _one_epoch(gaia_like(seed=1), i, {"BP": 0.8, "G": 0.95, "RP": 1.1})
    (row,) = band_consistency(flip, [{"t_min": flip.lightcurves[0].time[i]}])
    assert row["verdict"] == "consistent"
    # G-only dips are vetoed and cleaned from G alone; BP/RP faults leave G alone.
    g_only = _one_epoch(gaia_like(seed=1), i, {"G": 0.9})
    assert veto(g_only) == ["band_inconsistent"]
    cleaned, dropped = clean(g_only)
    assert len(dropped) >= 1 and len(cleaned.lightcurves[0].time) < 120 and len(cleaned.lightcurves[1].time) == 120
    assert not simple_dips(cleaned) or min(d["depth"] for d in simple_dips(cleaned)) < 0.05
    bp_fault = _one_epoch(gaia_like(seed=1), i, {"BP": 3.0})
    cleaned, dropped = clean(bp_fault)
    assert len(cleaned.lightcurves[0].time) == 120 and len(cleaned.lightcurves[1].time) < 120


def test_stage0_flags_inflated_errors_and_leaves_noisy_colours_untested():
    from cosmos_search.vetting.stage0 import band_consistency
    cand = gaia_like(seed=1)
    i = 40
    cand.lightcurves[0].flux[i] *= 0.9
    cand.lightcurves[0].flux_err[i] *= 20
    (row,) = band_consistency(cand, [{"t_min": cand.lightcurves[0].time[i]}])
    assert row["verdict"] == "inflated_error"
    noisy = gaia_like(seed=1)
    for lc in noisy.lightcurves[1:]:
        lc.flux_err[:] = 0.2  # BP/RP too poor to have shown a 10% dip
    noisy.lightcurves[0].flux[i] *= 0.9
    (row,) = band_consistency(noisy, [{"t_min": noisy.lightcurves[0].time[i]}])
    assert row["verdict"] == "untested"


def test_stage0_passes_injected_decoys():
    from cosmos_search.vetting.stage0 import epoch_verdicts
    host = gaia_like(seed=9, n=150)
    for spec in ("grey_primes", "chromatic_flip", "dust_dip"):
        cand, _, _ = make_decoy(host, spec, np.random.default_rng(3))
        g0 = epoch_verdicts(host)["verdict"]
        g1 = epoch_verdicts(cand)["verdict"]
        new = (g1 == "band_inconsistent") & (g0 != "band_inconsistent")
        assert not new.any(), spec


def test_conformal_widening_restores_coverage():
    rng = np.random.default_rng(0)
    n = 400
    mid = rng.normal(0, 1, n)
    y = mid + rng.normal(0, 1, n)
    half = np.full(n, 0.6)  # a nominal 80% interval would need 1.28: these are overconfident
    s = scoring.conformal_widening(mid - half, mid + half, y[:n])
    assert 1.8 < s < 2.5
    lo, hi = scoring.widen(mid - half, mid + half, s)
    assert 0.78 < scoring.interval_coverage(lo, hi, y) < 0.84
    assert scoring.conformal_widening([0], [1], [0.5]) == np.inf


def test_stage0_long_table_matches_the_candidate_path():
    import pyarrow as pa
    from cosmos_search.gaia.epoch import SCHEMA
    from cosmos_search.vetting.stage0 import clean, clean_long, epoch_verdicts, verdict_counts
    cand = _one_epoch(gaia_like(seed=1), 40, {"G": 0.9})
    _one_epoch(cand, 70, {"RP": 4.0})
    cols = {f.name: [] for f in SCHEMA}
    for lc in cand.lightcurves:
        n = len(lc.time)
        for k, v in (("source_id", [7] * n), ("band", [lc.band] * n), ("t", lc.time), ("flux", lc.flux * 1e4),
                     ("err", lc.flux_err * 1e4), ("flags", [0] * n), ("transit_id", np.arange(n) * 1000),
                     ("fov", [1] * n), ("ccd_row", [1] * n), ("n_obs", [9] * n)):
            cols[k] += list(v)
    table = pa.table({k: pa.array(v).cast(SCHEMA.field(k).type) if k != "band" else pa.array(v).dictionary_encode()
                      for k, v in cols.items()})
    out, counts = clean_long(table, blend={7: 0.0})
    want = verdict_counts(epoch_verdicts(cand)["verdict"])
    assert counts.iloc[0].drop("source_id").to_dict() == want
    assert want["stage0_n_band_inconsistent"] >= 1 and want["stage0_n_colour_fault"] >= 1
    cleaned, _ = clean(cand)
    bands = out.column("band").to_pylist()
    assert [bands.count(b) for b in ("G", "BP", "RP")] == [len(lc.time) for lc in cleaned.lightcurves]


def test_stage0_keeps_corroborated_and_diluted_changes():
    from cosmos_search.vetting.stage0 import band_consistency, dilution_floor
    i = 40
    # A real 10% dip seen in all three bands, on a transit whose G error is inflated, is kept.
    cand = _one_epoch(gaia_like(seed=1), i, {"G": 0.9, "BP": 0.9, "RP": 0.9})
    cand.lightcurves[0].flux_err[i] *= 20
    (row,) = band_consistency(cand, [{"t_min": cand.lightcurves[0].time[i]}])
    assert row["verdict"] == "consistent"
    # An eclipsing binary blended in BP/RP: BP and RP see a third of each eclipse. The eclipses are not faults.
    eb = gaia_like(seed=2, n=150)
    for lc in eb.lightcurves:
        ecl = (np.arange(len(lc.time)) % 15) == 0
        lc.flux[ecl] *= 1 - (0.3 if lc.band == "G" else 0.1)
    eb.source.update(phot_bp_n_blended_transits=150, phot_rp_n_blended_transits=150)
    g = eb.lightcurves[0]
    rows = band_consistency(eb, [{"t_min": t} for t in g.time[::15]])
    assert {r["verdict"] for r in rows} == {"consistent"} and 0.1 < rows[0]["r_lo"] < 0.35
    # A quiet star that is often blended cannot have its G-only dips told from diluted real ones.
    quiet = _one_epoch(gaia_like(seed=1), i, {"G": 0.9})
    quiet.source.update(phot_bp_n_blended_transits=60, phot_rp_n_blended_transits=60)
    (row,) = band_consistency(quiet, [{"t_min": quiet.lightcurves[0].time[i]}])
    assert row["verdict"] == "untested"
    assert dilution_floor(np.zeros(10), np.ones(10), np.zeros(10), np.zeros(10), None) == 0.0


def test_stage0_long_table_survives_a_source_without_bp():
    import pyarrow as pa
    from cosmos_search.gaia.epoch import SCHEMA
    from cosmos_search.vetting.stage0 import clean_long
    n = 30
    rows = {"source_id": [1] * n, "band": ["G"] * n, "t": np.arange(n, dtype=float), "flux": np.ones(n),
            "err": np.full(n, 0.01), "flags": [0] * n, "transit_id": np.arange(n), "fov": [1] * n,
            "ccd_row": [1] * n, "n_obs": [9] * n}
    table = pa.table({k: pa.array(v).cast(SCHEMA.field(k).type) if k != "band" else pa.array(v).dictionary_encode()
                      for k, v in rows.items()})
    out, counts = clean_long(table, blend={1: 0.0})
    assert out.num_rows == n and counts.iloc[0]["stage0_n_untested"] == n


SKYBOT_TEXT = """# Flag: 1
# Ticket: 1
# Num | Name | RA(h) | DE(deg) | Class | Mv | Err(arcsec) | d(arcsec) | dRA(arcsec/h) | dDEC(arcsec/h) | Dg(ua) | Dh(ua)
 1009 | Sirene | 15 55 45.3229 | -21 23 17.828 | Mars-Crosser | 20.7 | 0.067 | 543.824 | 14.4594 | 0.615 | 3.63335939559 | 3.77265346836
"""
MIRIADE = {
    "@gaia": "# header\n  2015-02-20T21:28:46.524  +15 56 22.96766  -21 25 34.8964      3.635510581     20.69\n",
    "500": "# header\n  2015-02-20T21:28:46.524  +15 55 45.29456  -21 23 17.8251      3.633355470     20.69\n",
}


class _FakeIMCCE:
    """SkyBoT and Miriade answers for Gaia DR3's own detection of (1009) Sirene, 2015-02-20 (sso_observation)."""

    def get(self, url, params=None, cache=False):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass
        r = R()
        r.text = SKYBOT_TEXT if "skybot" in url else MIRIADE[params["-observer"]]
        return r


def test_sso_search_corrects_for_gaia_parallax():
    from cosmos_search.vetting import sso
    ra, dec = 239.09585284122534, -21.426396751900427  # where Gaia saw Sirene
    bodies = sso.nearby(2457074.3949829205, ra, dec, 15.0, "gaia", client=_FakeIMCCE())
    # Geocentric, SkyBoT puts it 544 arcsec away; seen from Gaia it is where Gaia measured it.
    assert len(bodies) == 1 and bodies[0]["geo_sep_arcsec"] > 500 and bodies[0]["sep_arcsec"] < 1.0
    assert sso.flux_ratio(20.7, 15.0, "gaia") < 0.01
    cand = gaia_like()
    cand.source.update(ra=ra, dec=dec)
    cand.dips = [{"t_min": 57074.0 + 2400000.0 - 2457000.0, "depth": 0.002}]
    (row,) = sso.check(cand, client=_FakeIMCCE())
    assert row["verdict"] in ("sso_bright", "sso_faint") and row["bodies"]


def _clean_long_loop(table, blend=None, nsigma=4.0, err_ratio=5.0, w_range=(0.1, 0.9)):
    """The per-source clean_long as of #30, kept as the reference the vectorised version must match."""
    import pandas as pd
    import pyarrow as pa
    from cosmos_search.vetting.stage0 import ARTEFACTS, _assess, _partner, dilution_floor, verdict_counts

    df = table.to_pandas()
    band = df["band"].astype(str).to_numpy()
    flux, err, tid = (df[c].to_numpy() for c in ("flux", "err", "transit_id"))
    drop = np.zeros(len(df), bool)
    counts = []
    for sid, idx in df.groupby("source_id", sort=False).indices.items():
        rows = {b: idx[(band[idx] == b) & np.isfinite(flux[idx])] for b in ("G", "BP", "RP")}
        g = rows["G"]
        if len(g) == 0:
            counts.append({"source_id": sid, **verdict_counts([])})
            continue

        def changes(r):
            f, e = flux[r], err[r]
            med = float(np.median(f)) if len(f) else np.nan
            return 1 - f / med, e / med

        d_g, e_g = changes(g)
        cols = []
        for b in ("BP", "RP"):
            d, e = changes(rows[b])
            if len(d) == 0:
                cols += [np.full(len(g), np.nan)] * 2
                continue
            at = {t: i for i, t in enumerate(tid[rows[b]])}
            j = np.array([at.get(t, -1) for t in tid[g]])
            cols += [np.where(j >= 0, d[np.clip(j, 0, None)], np.nan), np.where(j >= 0, e[np.clip(j, 0, None)], np.nan)]
        r_lo = dilution_floor(d_g, e_g, cols[0], cols[2], (blend or {}).get(sid))
        v = _assess(d_g, e_g, *cols, nsigma, err_ratio, w_range, r_lo, _partner(df["t"].to_numpy()[g]))["verdict"]
        drop[g[np.isin(v, ARTEFACTS)]] = True
        faulty = set(tid[g][v == "colour_fault"])
        for b in ("BP", "RP"):
            drop[rows[b][np.isin(tid[rows[b]], list(faulty))]] = True
        counts.append({"source_id": sid, **verdict_counts(v)})
    return table.filter(pa.array(~drop)), pd.DataFrame(counts)


def _random_long_table(seed, n_sources=60):
    import pyarrow as pa
    from cosmos_search.gaia.epoch import SCHEMA

    rng = np.random.default_rng(seed)
    cols = {f.name: [] for f in SCHEMA}
    for s in range(n_sources):
        n = int(rng.choice([0, 3, 4, 6, 25, 80]))
        # Transits in FoV pairs (0.074 d apart), so partner transits exist.
        tt = np.sort(rng.uniform(0, 900, (n + 1) // 2)).repeat(2)[:n] + np.tile([0, 0.074], (n + 1) // 2)[:n]
        tids = np.arange(n) * 1000 + s
        base = rng.uniform(1e3, 1e5)
        variable = rng.random() < 0.4  # real, grey changes in some sources, so dilution floors are measured
        signal = np.where(rng.random(n) < 0.3, rng.uniform(-0.4, 0.4, n), 0.0) if variable else np.zeros(n)
        for b, share, dil in (("G", 1.0, 1.0), ("BP", 0.5, rng.choice([1.0, 0.4])), ("RP", 0.8, 1.0)):
            keep = rng.random(n) > (0.0 if b == "G" else 0.2)  # BP/RP missing at some transits
            t, tid_b = tt[keep], tids[keep]
            f = base * share * (1 + dil * signal[keep] + rng.normal(0, rng.uniform(0.001, 0.05), len(t)))
            if b == "G" and len(t) > 2:
                f[rng.integers(len(t))] *= rng.choice([0.5, 1.0, 1.6])  # G-only faults
            if b != "G" and len(t) > 2:
                f[rng.integers(len(t))] *= rng.choice([1.0, 30.0])  # colour faults
            f[rng.random(len(t)) < 0.03] = np.nan
            e = np.abs(f) * 0.01
            if len(t) > 2:
                e[rng.integers(len(t))] *= rng.choice([1.0, 20.0])  # inflated errors
            if b == "BP" and len(t) > 3 and rng.random() < 0.3:  # a repeated transit
                t, tid_b = np.append(t, t[0]), np.append(tid_b, tid_b[0])
                f, e = np.append(f, f[0] * 1.1), np.append(e, e[0])
            m = len(t)
            for k, val in (("source_id", [10_000 + s] * m), ("band", [b] * m), ("t", t), ("flux", f),
                           ("err", e), ("flags", [0] * m), ("transit_id", tid_b), ("fov", [1] * m),
                           ("ccd_row", [1] * m), ("n_obs", [9] * m)):
                cols[k] += list(val)
    table = pa.table({k: pa.array(v).cast(SCHEMA.field(k).type) if k != "band" else pa.array(v).dictionary_encode()
                      for k, v in cols.items()})
    blend = {10_000 + s: float(rng.choice([0.0, 0.02, 0.5, 1.0])) for s in range(n_sources) if rng.random() < 0.7}
    return table, blend


@pytest.mark.parametrize("seed", range(8))
def test_vectorised_clean_long_matches_the_per_source_loop(seed):
    from cosmos_search.vetting.stage0 import clean_long

    table, blend = _random_long_table(seed)
    for bl in (blend, None):
        out, counts = clean_long(table, blend=bl)
        ref_out, ref_counts = _clean_long_loop(table, blend=bl)
        assert out.num_rows == ref_out.num_rows and out.to_pandas().equals(ref_out.to_pandas())
        a = counts.sort_values("source_id").reset_index(drop=True).astype("int64")
        b = ref_counts.sort_values("source_id").reset_index(drop=True).astype("int64")
        assert a.equals(b)
    exercised = b.drop(columns=["source_id", "stage0_n_untested"]).sum()
    assert (exercised > 0).sum() >= 2  # several verdicts occur, so the comparison means something


def test_external_bands_keep_the_gaia_clock_and_holdout_window():
    from cosmos_search.vetting.candidate import LightCurve
    cand = gaia_like(dip_at=2457300.0)
    rng = np.random.default_rng(3)
    t = np.sort(np.concatenate([rng.uniform(2457850.0, 2457990.0, 50), rng.uniform(2458100.0, 2459000.0, 950)]))
    f = 1 + rng.normal(0, 0.003, len(t))
    f[-5] = 0.8                                            # a deep external epoch after Gaia's span
    cand.lightcurves.append(LightCurve("ztf_r", t, f, np.full(len(t), 0.003), {}))
    g = cand.band("G")
    assert cand.time_span() == (g.time.min(), g.time.max())
    packet, key = build(cand, holdout=("tail",))
    t0, t1 = g.time.min(), g.time.max()
    cut = t0 + 0.8 * (t1 - t0)
    w = packet["withheld"]
    assert w["after_t"] == pytest.approx(cut - t0, rel=1e-4) and w["until_t"] == pytest.approx(t1 - t0, rel=1e-4)
    shown = packet["summaries"]["ztf_r"]["n_epochs"]
    assert shown == int(np.sum((t < cut) | (t > t1)))      # external epochs inside the withheld window are hidden
    assert "Faintest epochs" in packet["text"] and f"{t[-5] - t0:.4f}" in packet["text"]


def test_external_apply_dilutes_tess_and_skips_g_only():
    from cosmos_search.dayone.external import apply
    from cosmos_search.inject import lightcurve as lcinj
    dip = lcinj.truth(lcinj.Dip(t0=10.0, covering=0.2, ingress=0.1, egress=0.1, flat=1.0))
    ext = pd.DataFrame({"source_id": [1, 1, 2], "band": ["TESS", "ztf_r", "TESS"], "t": [10.5, 10.5, 10.5],
                        "flux": 1.0, "err": 0.01})
    plan = {1: {"signal": dip, "bands": None}, 2: {"signal": dip, "bands": ["G"]}}
    out = apply(ext, plan, {1: 0.5})
    assert out.flux.tolist() == pytest.approx([0.9, 0.8, 1.0])


def test_dip_trains_continue_past_the_span():
    from cosmos_search.inject import lightcurve as lcinj
    for fam in ("dip_train_periodic", "dip_train_poisson", "dip_train_primes"):
        late = 0
        for seed in range(20):
            sig = lcinj.sample(np.random.default_rng(seed), fam, 0.0, 1000.0, t_ext=4000.0)
            late += sum(p.t0 > 1000.0 for p in sig.parts)
        assert late > 0, fam
