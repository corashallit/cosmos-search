"""Generate an injection run.

    python -m cosmos_search.inject --config configs/inject/smoke.yaml \
        --products-dir out/products --truth-out out/truth.json.age \
        --recipient-file keys/cora.age.pub \
        [--reveal-out out/reveal.json.age --reveal-recipient-file keys/reveal.age.pub]

Only the smoke class exists so far; it exercises the sealing path end to end
without touching real data.
"""

import argparse
import json
import os
import secrets
from pathlib import Path

import numpy as np
import yaml

from .sealed import sha256, write_sealed


def smoke(cfg, rng):
    """Synthetic packets, a fraction of them carrying an injected offset."""
    n = int(cfg["n_packets"])
    injected = rng.random(n) < float(cfg["injected_fraction"])
    lo, hi = cfg["amplitude"]
    amplitude = np.where(injected, rng.uniform(lo, hi, n), 0.0)
    packet_ids = [secrets.token_hex(8) for _ in range(n)]
    products = [
        {"packet_id": pid, "value": float(rng.normal() + a)}
        for pid, a in zip(packet_ids, amplitude)
    ]
    truth = [
        {"packet_id": pid, "injected": bool(f), "amplitude": float(a)}
        for pid, f, a in zip(packet_ids, injected, amplitude)
    ]
    reveal = {pid: ("decoy" if f else None) for pid, f in zip(packet_ids, injected)}
    return products, truth, reveal


GENERATORS = {"smoke": smoke}


def main(argv=None):
    p = argparse.ArgumentParser(prog="cosmos_search.inject")
    p.add_argument("--config", required=True)
    p.add_argument("--products-dir", required=True)
    p.add_argument("--truth-out", required=True)
    p.add_argument("--recipient-file", action="append", required=True)
    p.add_argument("--reveal-out")
    p.add_argument("--reveal-recipient-file", action="append")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(int.from_bytes(os.urandom(32), "little"))
    products, truth, reveal = GENERATORS[cfg["class"]](cfg, rng)

    products_dir = Path(args.products_dir)
    products_dir.mkdir(parents=True, exist_ok=True)
    products_path = products_dir / "products.json"
    products_path.write_text(json.dumps(products, sort_keys=True))

    manifest = {
        "config": args.config,
        "class": cfg["class"],
        "n_packets": len(products),
        "sha256": {"products.json": sha256(products_path)},
    }
    manifest["sha256"][Path(args.truth_out).name] = write_sealed(
        truth, args.truth_out, args.recipient_file
    )
    if args.reveal_out:
        manifest["sha256"][Path(args.reveal_out).name] = write_sealed(
            reveal, args.reveal_out, args.reveal_recipient_file or args.recipient_file
        )
    (products_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Counts only. Never print anything derived from truth.
    print(f"wrote {len(products)} packets; manifest at {products_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
