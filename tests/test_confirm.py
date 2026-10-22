import pandas as pd

from cosmos_search.dayone import confirm


def _table(n_dec_hit, n_dec, ctl_hits, n_queue_hit, n_queue=100):
    rows = []
    fams = list(confirm.HELD_OUT_FAMILIES)
    for i in range(n_dec):
        rows.append({"candidate_id": f"gaia_dr3:{i}", "kind": "decoy", "injected_family": fams[i % 4],
                     "role_in_queue": "decoy_host", "ok": True, "s_interesting": 0.9 if i < n_dec_hit else 0.1})
    strata = {}
    for c, hits in enumerate(ctl_hits):
        for j in range(15):
            cid = f"gaia_dr3:{1000 + 15 * c + j}"
            strata[cid] = f"class{c}"
            rows.append({"candidate_id": cid, "kind": "control", "injected_family": None, "role_in_queue": "control",
                         "ok": True, "s_interesting": 0.9 if j < hits else 0.0})
    for i in range(n_queue):
        rows.append({"candidate_id": f"gaia_dr3:{5000 + i}", "kind": "real", "injected_family": None,
                     "role_in_queue": "queue_sample", "ok": True, "s_interesting": 0.9 if i < n_queue_hit else 0.0})
    return pd.DataFrame(rows), strata


def test_good_rule_adopted_and_each_criterion_can_fail():
    t, st = _table(28, 48, [1] * 9, 2)
    assert confirm.evaluate(t, st)["adopted"]
    t, st = _table(15, 48, [1] * 9, 2)               # recall 0.31: lower bound under 0.30
    assert not confirm.evaluate(t, st)["criteria"]["held_out_recall"]["pass"]
    t, st = _table(28, 48, [1] * 8 + [4], 2)         # one class over 3 of 15
    r = confirm.evaluate(t, st)
    assert r["criteria"]["controls_pooled"]["pass"] and not r["criteria"]["controls_each_class"]["pass"]
    t, st = _table(28, 48, [1] * 9, 6)               # 6 of 100: upper bound over 0.10
    assert not confirm.evaluate(t, st)["adopted"]


def test_failed_verdict_never_helps_the_rule():
    t, st = _table(28, 48, [0] * 9, 0)
    t.loc[t.kind == "control", "ok"] = False
    assert confirm.evaluate(t, st)["criteria"]["controls_pooled"]["k"] == 135


TIER1 = ["ACV|CP|MCP|ROAM|ROAP|SXARI", "ECL", "LPV", "DSCT|GDOR|SXPHE", "YSO", "BE|GCAS|SDOR|WR"]
ACTIVE = ["SOLAR_LIKE", "RS", "AGN"]


def _table_v6(n_dec_hit, tier1_hits, active_hits, n_queue_hit, n_dec=48, s_hit=0.25):
    rows = []
    fams = list(confirm.HELD_OUT_FAMILIES_V2)
    for i in range(n_dec):
        rows.append({"candidate_id": f"gaia_dr3:{i}", "kind": "decoy", "injected_family": fams[i % 4],
                     "role_in_queue": "decoy_host", "ok": True, "s_interesting": s_hit if i < n_dec_hit else 0.1})
    strata = {}
    for c, (name, hits) in enumerate(zip(TIER1 + ACTIVE, list(tier1_hits) + list(active_hits))):
        for j in range(15):
            cid = f"gaia_dr3:{1000 + 15 * c + j}"
            strata[cid] = name
            rows.append({"candidate_id": cid, "kind": "control", "injected_family": None, "role_in_queue": "control",
                         "ok": True, "s_interesting": s_hit if j < hits else 0.0})
    for i in range(100):
        rows.append({"candidate_id": f"gaia_dr3:{5000 + i}", "kind": "real", "injected_family": None,
                     "role_in_queue": "queue_sample", "ok": True, "s_interesting": s_hit if i < n_queue_hit else 0.0})
    return pd.DataFrame(rows), strata


def test_v6_tier_aware_criteria_each_can_fail():
    t, st = _table_v6(30, [3, 1, 0, 0, 2, 0], [5, 1, 0], 2)    # batch 6's rates at S >= 0.2
    r = confirm.evaluate_v6(t, st)
    assert r["adopted"]
    assert r["v5_cut_same_verdicts"]["held_out_recall"]["k"] == 0     # hits sit at S = 0.25, under v5's cut
    t, st = _table_v6(25, [3, 1, 0, 0, 2, 0], [5, 1, 0], 2)    # recall lower bound under 0.40
    assert not confirm.evaluate_v6(t, st)["criteria"]["held_out_recall"]["pass"]
    t, st = _table_v6(30, [5, 1, 0, 0, 0, 0], [5, 1, 0], 2)    # one tier-1 class at 5 of 15
    r = confirm.evaluate_v6(t, st)["criteria"]
    assert r["controls_tier1_pooled"]["pass"] and not r["controls_tier1_each_class"]["pass"]
    t, st = _table_v6(30, [3, 3, 3, 3, 0, 0], [0, 0, 0], 2)    # 12 of 90 pooled: upper bound over 0.20
    r = confirm.evaluate_v6(t, st)["criteria"]
    assert r["controls_tier1_each_class"]["pass"] and not r["controls_tier1_pooled"]["pass"]
    t, st = _table_v6(30, [0] * 6, [12, 0, 0], 2)               # active classes are not held to 4 of 15 each
    r = confirm.evaluate_v6(t, st)["criteria"]
    assert r["controls_tier1_each_class"]["pass"] and r["controls_active_pooled"]["pass"]
    t, st = _table_v6(30, [0] * 6, [12, 5, 0], 2)               # but pooled: 17 of 45
    assert not confirm.evaluate_v6(t, st)["criteria"]["controls_active_pooled"]["pass"]
    t, st = _table_v6(30, [0] * 6, [0, 0, 0], 6)
    assert not confirm.evaluate_v6(t, st)["adopted"]
