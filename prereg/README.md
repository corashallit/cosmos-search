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

## Release-day review batches (from 2026-11-11)

The rule for these batches is `docs/vetting.md` §16, pushed before any of them is built: how many planted fades
a batch holds, of what kinds, and how many must be shortlisted for the batch to count as consistent with the
calibration. `scripts/slow_review.py files` draws the plan and builds the files in one step and writes the plan
and the key only encrypted (`D/plan.json.age`, `D/key.json.age`); steps 2 to 4 above apply unchanged, with
`vetting commit --plan D/plan.json.age`, and `scripts/slow_review.py table` is the reveal.

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

## Batches recorded under the procedure

| batch | rule pre-registered | verdicts archive | result |
|---|---|---|---|
| 5 | v4, in `docs/dayone.md` | `verdicts/dayone-batch5` cf60d94 | v4 not adopted |
| confirm-main, confirm-hidden | v5: d167af0 (PR #53) | `verdicts/confirm-main` e3ea437, `verdicts/confirm-hidden` 7d23f83, pushed 2026-10-22 ~03:07 UTC | v5 adopted (`docs/dayone.md`, "Confirmatory result") |
| slow-cal-1 | `docs/vetting.md` §14: 9b88faf (PR #77) | `verdicts/slow-cal-1` 83b887f | recall passes, false shortlist fails (§14, "Result") |
| slow-cal-2, slow-cal-2-repeat | `docs/vetting.md` §15: f505da4, amended 0ec1481 before the reveal (PR #85) | `verdicts/slow-cal-2` dc92bef (116 of 432 files), `verdicts/slow-cal-2-repeat` e3a9fcb (14) | stopped part way; criteria reported, not applied; reviewer stays that of slow-cal-1 (§15, "Result") |
| planted-trial-dr3-1 | `docs/vetting.md` §16 (PR #94, fb68acd), before the batch was built | `verdicts/planted-trial-dr3-1` cd637dc (31 files: 30 planted fades, one star) | 24 of 30 planted fades shortlisted: consistent with the calibration (§16, "Trial on DR3") |


## Which reviewer made the verdicts (from 2026-11-10)

A reviewer is a model, a reasoning effort and an agent type; a subagent takes the first two from the thread that
starts it unless the launch names them. From batch `slow-cal-2` on, `scripts/stage_a_audit.py` reads them from
every review's transcript, a review served by anything but the named model does not count, and `vetting commit
--audit` writes them into `commit.json` beside the verdicts' hash. `reviewers/<batch>.json` is the record of a
calibration batch's reviewer; the audit of a later batch takes it with `--reviewer` (`docs/vetting.md` §15).
Earlier batches carry no such record: by their transcripts all ran on `claude-opus-5-5`, the confirmatory
batches at effort `high` and `slow-cal-1` at `xhigh`.
