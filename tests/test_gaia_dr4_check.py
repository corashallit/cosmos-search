"""First-hour checks on fake DR4 files: a file's layout, and DR4 against DR3 for the same stars."""

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from test_gaia_dr4_fake import (  # noqa: F401 (served is a fixture)
    build,
    read_all,
    served,
)

from cosmos_search.gaia import dr4_check, dr4_fake


def fake_file(root, pixel=7):
    return next((root / dr4_fake.PRODUCT).rglob(f"Npix={pixel}.parquet"))


def test_file_report_passes_a_file_in_the_draft_layout(tmp_path):
    root, _ = build(tmp_path)
    rep = dr4_check.file_report(fake_file(root))
    assert rep["problems"] == [] and rep["one_row_per"] == "transit" and rep["stars_contiguous"]
    assert rep["stars"] == 2 and rep["rows"] == 74 and rep["duplicate_transits"] == 0
    assert rep["draft_columns_absent"] == [] and rep["columns_not_in_draft"] == []
    assert abs(rep["bands"]["G"]["zero_point"]["median"] - dr4_fake.ZERO_POINT["g"]) < 1e-4
    assert rep["reader"]["lightcurve_rows"]["G"] == 74
    assert any("variability" in n for n in rep["notes"])  # the draft table marks no rejected epochs


def test_file_report_names_what_is_wrong(tmp_path):
    root, _ = build(tmp_path)
    t = pq.read_table(fake_file(root))

    def report(table, name):
        pq.write_table(table, tmp_path / name, row_group_size=20)
        return dr4_check.file_report(tmp_path / name)

    # Times in Julian days instead of days since 2010-01-01.
    i = t.column_names.index("g_obs_time")
    jd = report(t.set_column(i, "g_obs_time", pc.add(t["g_obs_time"], 2455197.5)), "jd.parquet")
    assert any("time zero point" in p for p in jd["problems"])
    # A required column under a name the reader does not know.
    renamed = report(t.rename_columns(["g_brightness" if c == "g_flux" else c for c in t.column_names]), "r.parquet")
    assert any("g_flux" in p for p in renamed["problems"])
    # DR3's spelling is accepted and reported; an extra column is noted against the draft data model.
    dr3_names = report(t.rename_columns(["g_transit_time" if c == "g_obs_time" else c for c in t.column_names])
                       .append_column("new_flag", pa.array([0] * t.num_rows)), "a.parquet")
    assert dr3_names["problems"] == [] and dr3_names["columns_not_in_draft"] == ["g_transit_time", "new_flag"]
    assert any("another spelling" in n for n in dr3_names["notes"])
    # A star's rows in two places.
    shuffled = report(pa.concat_tables([t.slice(0, 10), t.slice(50), t.slice(10, 40)]), "s.parquet")
    assert not shuffled["stars_contiguous"] and any("not contiguous" in n for n in shuffled["notes"])
    # The same transit twice.
    twice = report(pa.concat_tables([t.slice(0, 5), t]), "d.parquet")
    assert twice["duplicate_transits"] == 5 and any("repeated" in p for p in twice["problems"])


def test_compare_releases(tmp_path):
    root, dr3 = build(tmp_path)
    lc = read_all(root)
    same = dr4_check.compare_releases(lc, dr3)
    assert same["problems"] == [] and same["stars_in_both"] == dr3["source_id"].nunique()
    assert same["matched_by_transit_id"] == 1.0 and same["dr3_epochs_found_in_dr4"] == 1.0
    assert abs(same["bands"]["G"]["flux_scale"]["median"] - 1) < 1e-6 and same["time_offset_days"]["median"] == 0

    # Another flux unit, and times an hour off.
    off = lc.assign(flux=lc["flux"] * 1.5, t=lc["t"] + 1 / 24)
    rep = dr4_check.compare_releases(off, dr3, time_tolerance=0.1)
    assert any("1.500 times" in p for p in rep["problems"]) and any("offset" in p for p in rep["problems"])

    # Transit ids renumbered: epochs are matched by time instead, with a note.
    renum = dr4_check.compare_releases(lc.assign(transit_id=lc["transit_id"] + 1), dr3)
    assert renum["matched_by_transit_id"] == 0 and renum["dr3_epochs_found_in_dr4"] > 0.99
    assert renum["problems"] == [] and any("matched by time" in n for n in renum["notes"])

    # Source ids changed between releases: nothing lines up until the map from dr3_neighbourhood is given.
    moved = lc.assign(source_id=lc["source_id"] + 1)
    assert dr4_check.compare_releases(moved, dr3)["problems"]
    id_map = {s + 1: s for s in dr3["source_id"].unique()}
    assert dr4_check.compare_releases(moved, dr3, id_map)["problems"] == []


def test_routes_see_the_stand_in_files(served):  # noqa: F811
    from cosmos_search.gaia import dr4_routes

    cdn = dr4_routes.check_cdn()
    assert cdn["epoch_photometry_files"] == 3 and cdn["with_md5"] == 3
    assert dr4_routes.check_cdn("Gaia/gdr5/")["top_level"] == []
