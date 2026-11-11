"""Which reviewer blind reviews run on (docs/vetting.md §7 and §15).

A reviewer is a model, a reasoning effort and an agent type. A subagent takes the first two from the thread that
starts it unless the launch names them, so nothing fixed them until 2026-11-10. Now:

- a calibration batch's reviewer is read from its transcripts and written to `prereg/reviewers/<batch>.json`
  (`scripts/stage_a_audit.py --record`), next to the recall and false-shortlist rate measured for it;
- ADOPTED names the batch whose reviewer release-day reviews use;
- launches name that model and effort (`scripts/stage_a_reviews.js`, `launch_args`), the reviewing agents' files
  in `.claude/agents/` carry them too (a test keeps them equal to the record), and the audit of every later batch
  takes the record with `--reviewer` and refuses reviews served by anything else.

No model is named in code: the record names it, and changing reviewer means a new calibration batch and a new
record, not an edit here.
"""

import json
from pathlib import Path

ADOPTED = "slow-cal-1"
ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "prereg" / "reviewers"
AGENT_FILES = (ROOT / ".claude" / "agents" / "expert-blind.md", ROOT / ".claude" / "agents" / "expert-reconcile.md")


def record(batch=ADOPTED):
    """The reviewer record of a calibration batch: model, effort, agent_type, launch, harness, packets."""
    return json.loads((RECORDS / f"{batch}.json").read_text())


def launch_args(directory, ids, batch=ADOPTED):
    """The `args` for scripts/stage_a_reviews.js: this batch's reviewer on the given prompt files."""
    r = record(batch)
    return {"dir": str(Path(directory).resolve()), "ids": list(ids), "model": r["model"], "effort": r["effort"]}


def frontmatter(path):
    """The `key: value` lines between the two `---` markers of an agent file."""
    head = Path(path).read_text().split("---\n", 2)[1]
    return {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in head.splitlines() if ":" in line)}
