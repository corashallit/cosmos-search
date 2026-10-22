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
