# Blinding, injections and decoys

## Short answer

Full blinding is **not achievable** in one repo and one container. Every agent working here (orchestrator,
subagents, workflow agents) shares a filesystem and environment, and can read any file, any git history and
any environment variable. "Don't look at `truth/`" in a prompt is an honour system, not blinding.

It **is** achievable to make the analysis blind *by construction* against every realistic leak except one,
with the design below. The remaining leak (comparing injected data against the live archive) is closed by
running blind stages on a frozen snapshot. If that is not strong enough, the upgrade path is at the end.

## Where truth can leak

| Leak | Example | Closed by |
|---|---|---|
| Truth on disk or in git | `truth.csv` committed once, then deleted; history keeps it | Truth only ever exists encrypted (§1) |
| Truth in an agent's context | The orchestrator ran the injector and saw its output | Injector draws from unlogged randomness and prints nothing (§1) |
| Data diff against the archive | An expert pulls the real TESS light curve and sees the injected transit is missing | Blind stages read a frozen snapshot, no network (§2, §3) |
| Identifiers | The expert looks up the object in SIMBAD/ADS and finds it is a boring K dwarf | Packets are anonymised (§3) |
| Statistical fingerprints | Injections are too clean, too periodic, or cluster on the sky | Inject into real data; draw hosts from the real sample; audit with a classifier (§4) |
| Thresholds tuned on the test set | Cut chosen because it recovers the sealed injections | Tuning/sealed split; sealed truth decrypted only after a frozen tag (§5) |

## 1. Sealed truth

- Cora generates an `age` key pair on her own machine. Only the public key (`keys/cosmos-search.age.pub`)
  goes in the repo. The private key never enters the container.
- `cosmos_search/inject/` holds the **distributions** in code, in public: which classes, what parameter
  ranges, what rate. Knowing the distribution is fine (LIGO's blind injections work the same way).
- The injector takes its seed from `os.urandom`, writes the injected data products, and writes the truth
  table straight to `truth/<run>.truth.parquet.age` encrypted to the public key. It prints only counts.
  Plaintext never touches disk.
- The orchestrator runs the injector but never sees the draws, so it is as blind as any subagent.
- `.gitignore` excludes plaintext truth patterns, and a pre-commit hook refuses any file under `truth/`
  that is not `age`-encrypted.
- Commit the SHA-256 of each sealed file in the run log, so the truth cannot be quietly regenerated later.

## 2. Frozen snapshot for blind runs

The classical tier and the blind expert stage read only from a snapshot of the data store taken *after*
injection. During a blind run, the archive clients are disabled (the shared HTTP layer refuses requests
when `COSMOS_BLIND=1`). This is what stops an agent from noticing that an injected light curve differs from
the MAST copy.

## 3. Two-stage experts

**Stage A, blind physical interpretation.** Each expert subagent receives one packet: anonymised ID,
coordinates rotated or withheld, distance, photometry, SED, light-curve features and plots, spectra,
classical-tier residuals. Decoys and controls are mixed in at a hidden ~10% rate (statistics.md §6).
The expert agent is defined in `.claude/agents/expert-blind.md` with **no tools**; the packet is inlined
in its prompt. With no tools it cannot read files, run code or reach the network. Its output is a
structured verdict: explanation class, probability, and the evidence it used.

**Commit, then reveal.** Stage A verdicts for a batch are committed (file plus SHA-256 in the run log)
before anything about the batch is revealed. The packet → real-ID map, in which decoys map to "decoy", is
encrypted to a second, per-project *reveal key* held as an environment secret. Only after the commit does a
script decrypt the map, drop decoys and score them. Agents could read that secret in principle, but Stage A
agents have no tools, and verdicts are fixed before the reveal, so reading it early gains nothing.

**Stage B, unblinded reconciliation.** For real objects only, a second agent gets the real identifiers and
full tool access: SIMBAD, ADS, VizieR, archive re-queries. Its job is to reconcile the Stage A verdict with
the literature and flag disagreements.

Two keys, then: the reveal key (per-batch decoy scoring, routine) and Cora's key (the sealed
injection-recovery set, decrypted only after a frozen tag).

Stage A cannot use literature, which is a real cost: some objects are only explicable with a paper in hand.
Those land as "unexplained in Stage A, explained in Stage B", which is itself a useful measurement.

## 4. Making injections indistinguishable

- Inject into real data at the level the pipeline reads (light curves, photometry tables), so noise and
  systematics are real.
- Draw injection hosts from the real sample with the same sky, magnitude and colour distribution.
- Before each sealed run, train a quick classifier to separate injected from real objects using pipeline
  features that should carry no signal (sky position, cadence, error columns). If it beats chance, fix the
  injector before sealing.

## 5. Tuning versus sealed sets

Two injection sets per run: a visible tuning set (truth in plaintext, used to set thresholds) and a sealed
set (truth encrypted). Final completeness and expert-calibration numbers come only from the sealed set,
decrypted after the `dr3-frozen` tag and again after the DR4 freeze.

## Upgrade path if this is not enough

Move injection out of this container entirely:

- A second private repo, `cosmos-search-truth`, never added to any analysis session's repository scope.
- A GitHub Actions workflow there holds the injection seed as an Actions secret, generates the injected
  data products, publishes them to the durable store, and publishes truth encrypted to Cora's key.
- Analysis sessions only ever see the injected products.

This removes the remaining honour-system element (an agent in the analysis container could, in principle,
read the injector's code path and reason about which objects it touched). It costs a second repo and some
Actions minutes, and it slows iteration on the injector. I'd start with the single-repo design and move to
this before the DR4 sealed run if the DR3 classifier audit (§4) or the decoy scores look suspicious.
