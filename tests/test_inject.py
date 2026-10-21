import json

from pyrage import decrypt, x25519

from cosmos_search.inject.__main__ import main


def test_smoke_run_seals_truth(tmp_path, capsys):
    ident = x25519.Identity.generate()
    pub = tmp_path / "test.age.pub"
    pub.write_text(str(ident.to_public()) + "\n")
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text("class: smoke\nn_packets: 50\ninjected_fraction: 0.5\namplitude: [3, 6]\n")
    out = tmp_path / "out"

    main([
        "--config", str(cfg), "--products-dir", str(out / "products"),
        "--truth-out", str(out / "truth.json.age"), "--recipient-file", str(pub),
        "--reveal-out", str(out / "reveal.json.age"),
    ])

    printed = capsys.readouterr().out
    truth = json.loads(decrypt((out / "truth.json.age").read_bytes(), [ident]))
    products = json.loads((out / "products" / "products.json").read_text())
    assert [t["packet_id"] for t in truth] == [p["packet_id"] for p in products]
    assert not any(t["packet_id"] in printed for t in truth)
    assert b"injected" not in (out / "truth.json.age").read_bytes()
    manifest = json.loads((out / "products" / "manifest.json").read_text())
    assert set(manifest["sha256"]) == {"products.json", "truth.json.age", "reveal.json.age"}
