# Blinding, injections and decoys

## Short answer

Inside one repo and one container, blinding is an honour system: every agent working here (orchestrator,
subagents, workflow agents) shares a filesystem and environment, and can read any file, git history or
environment variable. So the randomness and the truth live somewhere no Claude session can reach.

**Design:** a second private repo, `cosmos-search-truth`, on which the Claude GitHub App is **not**
installed. That is an access-control boundary, not an instruction: Claude sessions reach GitHub only
through that App, so no agent in this project can clone, read or attach the truth repo, whatever it tries.
A second repo with the App installed would not be enough, because any session can attach a repo the App
can see.

## Leaks and what closes them

| Leak | Example | Closed by |
|---|---|---|
| Truth on disk or in git | `truth.csv` committed once, then deleted; history keeps it | Truth is created only in the truth repo's Actions runner and leaves it encrypted (§1) |
| Truth in an agent's context | The orchestrator ran the injector and saw its output | No agent runs the injector; Actions does (§1) |
| Another repo in the project | An agent attaches `cosmos-search-truth` mid-session | Claude GitHub App not installed on it (§1) |
| Data diff against the archive | An expert pulls the real TESS light curve and sees the injected transit is missing | Blind stages read a frozen snapshot, no network (§2, §3) |
| Identifiers | The expert looks up the object in SIMBAD/ADS and finds a boring K dwarf | Packets are anonymised (§3) |
| Statistical fingerprints | Injections are too clean, too periodic, or cluster on the sky | Inject into real data; draw hosts from the real sample; audit with a classifier (§4) |
| Thresholds tuned on the test set | Cut chosen because it recovers the sealed injections | Tuning/sealed split; sealed truth decrypted only after a frozen tag (§5) |

## 1. Truth repo

Setup (Cora, once):

1. Create private repo `cosmos-search-truth`.
2. In GitHub → Settings → Applications → Claude → Repository access, use **Only select repositories** and
   leave `cosmos-search-truth` out. Do not add it under the project's repositories either.
3. Generate an `age` key pair locally. Commit the public key to `keys/cosmos-search.age.pub` in *both*
   repos. The private key never enters a container.
4. Give the truth repo read access to `cosmos-search` (a fine-grained token with read-only contents
   permission on that one repo, stored as an Actions secret), and write access to the durable store once
   it exists.

What runs there:

- The injector **code and distributions** stay public, in `cosmos_search/inject/` in this repo. Knowing the
  distribution is fine; LIGO's blind injections work the same way. The truth repo holds only a short
  workflow that checks out a tagged version of this repo and runs it.
- The workflow draws its seed from the runner's `os.urandom`, writes injected data products and decoy
  packets to the durable store, and writes truth tables encrypted to the public key. Logs print counts only.
- Each run records the SHA-256 of its encrypted outputs and the `cosmos-search` commit it ran, so truth
  cannot be quietly regenerated.
- Sealed truth is decrypted only by Cora, after a frozen tag.

In this repo, `.gitignore` excludes plaintext truth patterns, and a pre-commit hook refuses anything under
`truth/` that is not `age`-encrypted, as a backstop.

Until the truth repo and durable store exist, the DR3 tuning set (visible truth, §5) is all we need, so
none of this blocks early work.

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
produced by the truth repo's workflow alongside the packets, encrypted to a second, per-project *reveal
key* held as an environment secret here. Only after the commit does a
script decrypt the map, drop decoys and score them. Agents could read that secret in principle, but Stage A
agents have no tools, and verdicts are fixed before the reveal, so reading it early gains nothing.

**Stage B, unblinded reconciliation.** For real objects only, a second agent gets the real identifiers and
full tool access: SIMBAD, ADS, VizieR, archive re-queries. Its job is to reconcile the Stage A verdict with
the literature and flag disagreements.

Two keys, then: the reveal key (per-batch decoy scoring, routine) and Cora's `age` key (the sealed
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

## What remains honour-system

Agents here can read the injector code and reason about how it chooses hosts. §4's classifier audit is the
check on that: if injected objects are distinguishable from pipeline features alone, the injector gets
fixed before sealing. Agents could also read the reveal key (§3) early, but Stage A verdicts are
committed before the reveal and Stage A agents have no tools, so it gains them nothing.
