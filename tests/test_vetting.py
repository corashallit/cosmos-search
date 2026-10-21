import json
import re

import numpy as np
import pytest
from pyrage import encrypt, x25519

from cosmos_search.inject.classes.vetting_decoys import DECOY_SPECS, make_decoy, vetting_batch
from cosmos_search.vetting import coincidence, design, quantify, scoring, taxonomy
from cosmos_search.vetting.anonymise import ecliptic_latitude, galactic_latitude, leak_scan, scrub_source
from cosmos_search.vetting.batch import build_batch, harmonise
from cosmos_search.vetting.candidate import Candidate
from cosmos_search.vetting.commit import CommitError, commit, load_verdicts, reveal
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
                   "bp_rp": 0.55, "parallax": 2.2, "parallax_error": 0.02, "ruwe": 1.01},
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
    cand, extra, truth = make_decoy(host, spec, np.random.default_rng(1))
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
        reveal(batch, sealed)
    commit([v], batch)
    with pytest.raises(CommitError):
        commit([v], batch)
    monkeypatch.setenv("COSMOS_REVEAL_KEY", str(ident))
    keys = reveal(batch, sealed)
    assert keys[packet["packet_id"]]["label"] == "eclipsing_binary"
    (batch / "verdicts.jsonl").write_text((batch / "verdicts.jsonl").read_text().replace("1.0", "0.9"))
    with pytest.raises(CommitError):
        load_verdicts(batch)


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
         "--vetter", "test", "--batch", str(tmp_path / "batch")])
    cli(["score", "--batch", str(tmp_path / "batch"), "--key", str(tmp_path / "key.json"), "--out", str(tmp_path / "r.json")])
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
        cand, extra, truth = make_decoy(host, spec, np.random.default_rng(3), t_cut=cut)
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
    cand, extra, truth = make_decoy(host, "grey_primes", np.random.default_rng(5))
    assert extra["role"] == "anomaly_probe" and truth["visibility"]["n_events"] >= 3


def test_random_packet_ids_never_trip_the_leak_scan():
    cand = gaia_like()
    for _ in range(300):
        packet, _ = build(cand)
        assert leak_scan(packet["text"]) == []
    assert leak_scan("source 4567890123456789") == ["long_integer"]
