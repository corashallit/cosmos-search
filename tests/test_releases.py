import numpy as np
import pandas as pd
import pytest

from cosmos_search.targets import releases


def test_every_dr4_query_column_is_in_the_draft_data_model():
    assert releases.missing_columns("dr4", releases.draft_columns()) == []


def test_missing_columns_names_absent_tables_and_columns():
    have = releases.draft_columns()
    have = {k: [c for c in v if c != "parallax_pmra_corr"] for k, v in have.items() if k != "dr3_neighbourhood"}
    assert releases.missing_columns("dr4", have) == [
        "gaiadr4.dr3_neighbourhood (table)", "gaiadr4.all_source_astrometry.parallax_pmra_corr"]


def test_dr3_names_that_moved_in_dr4_are_not_asked_of_gaia_source():
    draft = releases.draft_columns()
    for gone in ("astrometric_params_solved", "phot_bp_n_blended_transits", "has_epoch_photometry",
                 "phot_bp_rp_excess_factor"):
        assert gone not in draft["gaia_source"]  # why the lookups below go to the all_source tables
    q = dict((names[0], adql) for adql, names in releases.star_queries(
        "dr4", ("astrometric_params_solved", "phot_bp_n_blended_transits", "best_class_name")))
    assert "s.astrometric_params AS astrometric_params_solved" in q["astrometric_params_solved"]
    assert "gaiadr4.all_source_photometry" in q["phot_bp_n_blended_transits"]
    assert "s.phot_variable_best_class AS best_class_name" in q["best_class_name"]
    assert len(releases.star_queries("dr3", ("parallax", "bp_rp", "has_epoch_photometry"))) == 1
    with pytest.raises(ValueError):
        releases.star_queries("dr4", ("no_such_column",))


class FakeTap:
    """Answers each per-table query for the uploaded ids it knows (id 3 is in no table)."""

    def __init__(self):
        self.calls = []

    def query(self, adql, uploads=None, cache=False, **kw):
        from astropy.table import Table

        ids = [int(i) for i in uploads["ids"]["source_id"] if int(i) != 3]
        self.calls.append((adql, len(uploads["ids"])))
        if "all_source_flags" in adql:
            ids = [i for i in ids if i == 1]
            return Table({"SOURCE_ID": ids, "BEST_CLASS_NAME": ["ECL"] * len(ids)})
        return Table({"source_id": ids, "parallax": [float(i) for i in ids]})


def test_fetch_star_rows_merges_tables_and_keeps_every_id():
    tap = FakeTap()
    d = releases.fetch_star_rows(tap, [5, 3, 1, 5], "dr4", ("parallax", "best_class_name", "best_class_score"),
                                 chunk=2, workers=1)
    assert d["source_id"].tolist() == [1, 3, 5]
    assert d["parallax"].tolist()[0] == 1.0 and np.isnan(d["parallax"].iloc[1])
    assert d["best_class_name"].tolist()[0] == "ECL" and pd.isna(d["best_class_name"].iloc[2])
    assert d["best_class_score"].isna().all()  # DR4 publishes no single score
    assert len(tap.calls) == 4  # two tables x two chunks
