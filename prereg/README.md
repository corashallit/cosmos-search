# Pre-registration record

This directory is where verdicts are committed before a reveal, so that "rule written, then verdicts fixed, then
results seen" can be checked from the repository instead of taken on trust.

## How a blind batch is recorded (from 2026-10-21, review finding R2)

1. **Rule.** The rule, its pass criteria and the batch design go into `docs/` in a commit pushed before the decoy
   plan is drawn. The sealed plan's sha256 goes into `commit.json` (`vetting commit --plan`).
2. **Verdicts.** `python -m cosmos_search.vetting commit ... --archive prereg/verdicts/<batch>` writes
   `verdicts.jsonl` and `commit.json` into the batch directory and also into `prereg/verdicts/<batch>/`, and
   makes a git commit of only those two files. It refuses to run with uncommitted changes to tracked files, so
   `code_commit` names the code that built the packets.
3. **Push.** The archive commit is pushed to the PR branch. GitHub keeps every pushed commit of a pull request
   under `refs/pull/<n>/head` even after a squash merge, and the PR's timeline shows when each push happened.
4. **Reveal.** `reveal()` (and `dayone.vet score`) fetch from `origin` and refuse unless a remote branch contains
   the archive commit and the archived `verdicts.jsonl` hashes to the value in `commit.json`. Only tuning batches
   may skip this (`--unpublished`).
5. **Result.** The scored result is a later commit on the same PR.

To check a batch: `git fetch origin '+refs/pull/*/head:refs/remotes/pr/*'`, find the archive commit, compare its
`verdicts.jsonl` sha256 with `commit.json`, and confirm on the PR's timeline that it was pushed before the commit
that reports the result.

## Day-one rehearsal batches 1–4: archived after the fact

These four batches ran before the procedure above existed. Their verdict files lived only in the agent's
container and were copied here on 2026-10-21 at about 23:00 UTC, after every reveal. Each `verdicts.jsonl` still
hashes to the `verdicts_sha256` in its `commit.json`, but nothing outside the container shows when the files were
fixed. Treat their "committed before the reveal" as the agent's statement, not as verified.

The code and rule commits they cite were squashed out of `main` but survive in the pull request refs:

| batch | rule pre-registered | code at verdict commit | verdicts committed (local) | result pushed | PR |
|---|---|---|---|---|---|
| 1 (rehearsal) | P(u) ≥ 0.25, in `docs/dayone.md` | 4cea668 (on `main`) | 14:54:30 | in 506bd83 | #40 |
| 2 | v1: ff66b65, 15:38:30 | 830f1aa | 16:17:14 | 50be0e7, 16:20:11 | #41 (`refs/pull/41/head`) |
| 3 | v2: 0f57fcb, 16:50:48 | 0a3882c | 17:27:03 | 8dcdd9c, 17:29:31 | #43 (`refs/pull/43/head`) |
| 4 | v3: 1eed896, 18:16:06 | 2d15944 | 18:28:57 | 4fe376f, 18:33 | #44 (`refs/pull/44/head`) |

Times are UTC on 2026-10-21, from commit metadata (author-set, so not independent evidence). The rule commits
precede the verdicts in each PR's history. Batch 1's rule was set in the pilots before the rehearsal.
