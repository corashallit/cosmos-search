"""Audit Stage A subagent transcripts: what each blind reviewer did, and which reviewer it was.

    python scripts/stage_a_audit.py --transcripts ~/.claude/projects/<project>/<session>/subagents \\
        --prompts D/prompts --replies D/replies --out D/audit.json [--copy D/transcripts] \\
        [--model <model id> --effort <level> | --reviewer prereg/reviewers/<batch>.json] [--record <file> --batch <name>]

Two things are checked for every prompt file.

*What the reviewer did.* Its tool calls must be exactly one Read of prompts/<id>.md and one Write of
replies/<id>.txt (plus the harness's hand-back), in that order. Prompt and reply paths are compared as absolute
paths, so pass absolute directories.

*Which reviewer it was.* The transcript records, for every model turn, the model that served it, the model that was
asked for, the reasoning effort and the agent type. A review counts only if one model served every turn, that model
is the one asked for (the harness can fall back to another model when the first is unavailable), no turn is an API
error, and nothing from the launching thread's conversation was copied into the reviewer's context. With --model
(and optionally --effort, --agent-type), or --reviewer pointing at the record of a calibration batch, the model,
effort and agent type must also equal the expected ones; without them, every review must share one configuration,
which is reported. docs/vetting.md §7: any change to the reviewer's configuration is a new reviewer, whose recall
and false-shortlist rate have to be measured again.

Prompt files without a counted review are listed for a re-run. --record writes the batch's reviewer record (the
configuration every counted review shares), which later audits take with --reviewer. Transcript directories are
searched recursively, so the `subagents` directory of a session covers Agent-tool launches and workflow runs.
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

HANDBACK = {"SubagentHandback"}
ID = re.compile(r"prompts/([a-z]{16})\.md")
SYNTHETIC = "<synthetic>"          # the harness's placeholder model on an API-error turn
FORWARDED = "queued_command"       # a message from the launching thread's conversation, copied into the subagent
WORKFLOW_MARK = "[Workflow harness"
KEYS = ("model", "effort", "agent_type")


def read(path):
    """First user text, tool calls and reviewer record of one transcript."""
    first, tools = None, []
    served, asked, effort, agent, harness, errors, forwarded = set(), set(), set(), set(), set(), 0, 0
    usage, stamps = {}, []
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("timestamp") and d.get("type") in ("user", "assistant"):
            stamps.append(d["timestamp"])    # an attachment carries the time of what it copies, not of the review
        if d.get("type") == "attachment" and (d.get("attachment") or {}).get("type") == FORWARDED:
            forwarded += 1
        m = d.get("message") or {}
        c = m.get("content")
        if first is None and m.get("role") == "user" and isinstance(c, str):
            first = c
        if d.get("type") == "assistant":
            if m.get("model") == SYNTHETIC or d.get("isApiErrorMessage"):
                errors += 1
            else:
                served.add(m.get("model"))
                asked.add(d.get("requestedModel"))
                effort.add(d.get("effort"))
                agent.add(d.get("attributionAgent"))
                harness.add(d.get("version"))
                if d.get("requestId") and isinstance(m.get("usage"), dict):
                    usage[d["requestId"]] = m["usage"]      # one request spans several transcript lines
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools.append((b.get("name"), b.get("input") or {}))
    first = first or ""
    tok = {k: sum(int(u.get(k) or 0) for u in usage.values())
           for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    rec = {"served": _sorted(served), "asked": _sorted(asked), "effort": _sorted(effort), "agent_type": _sorted(agent),
           "harness": _sorted(harness), "api_errors": errors, "forwarded_messages": forwarded,
           "launch": "workflow" if first.lstrip().startswith(WORKFLOW_MARK) else "agent tool",
           "tokens": sum(tok.values()), "output_tokens": tok["output_tokens"],
           "start": min(stamps) if stamps else None, "end": max(stamps) if stamps else None}
    return first, tools, rec


def _sorted(values):
    return sorted(str(v) for v in values if v is not None)


def config(rec):
    """(model, effort, agent type) when one of each served the whole review, else None."""
    if len(rec["served"]) == 1 and len(rec["effort"]) <= 1 and len(rec["agent_type"]) <= 1:
        return rec["served"][0], (rec["effort"] or [None])[0], (rec["agent_type"] or [None])[0]
    return None


def reviewer_problems(rec, expected=None):
    """Why this transcript's review does not count as the expected reviewer's ([] when it does)."""
    out = []
    if not rec["served"]:
        out.append("no model turn")
    if len(rec["served"]) > 1:
        out.append(f"served by more than one model: {', '.join(rec['served'])}")
    if rec["asked"] and rec["asked"] != rec["served"]:
        out.append(f"asked for {', '.join(rec['asked'])}, served by {', '.join(rec['served']) or 'nothing'}")
    if len(rec["effort"]) > 1:
        out.append(f"more than one effort level: {', '.join(rec['effort'])}")
    if rec["api_errors"]:
        out.append(f"{rec['api_errors']} API-error turn(s)")
    if rec["forwarded_messages"]:
        out.append("a message from the launching thread was copied into the reviewer's context")
    for key, got in (("model", rec["served"]), ("effort", rec["effort"]), ("agent_type", rec["agent_type"])):
        want = (expected or {}).get(key)
        if want and got != [want]:
            out.append(f"{key} {', '.join(got) or 'not recorded'}, expected {want}")
    return out


def audit(transcripts, prompts, replies, expected=None):
    """{packet id: [one entry per transcript]}. An entry is `clean` when the tool calls are the one Read and the one
    Write, and `counted` when it is also the expected reviewer (reviewer_problems is empty)."""
    prompts, replies = Path(prompts).resolve(), Path(replies).resolve()
    dirs = [transcripts] if isinstance(transcripts, (str, Path)) else list(transcripts)
    out = {}
    for t in sorted({p for d in dirs for p in Path(d).rglob("*.jsonl")}):
        first, tools, rec = read(t)
        m = ID.search(first)
        if not m or str(prompts) not in first:
            continue
        pid = m.group(1)
        work = [(n, i) for n, i in tools if n not in HANDBACK]
        clean = (len(work) == 2 and work[0][0] == "Read" and work[0][1].get("file_path") == str(prompts / f"{pid}.md")
                 and work[1][0] == "Write" and work[1][1].get("file_path") == str(replies / f"{pid}.txt"))
        problems = reviewer_problems(rec, expected)
        out.setdefault(pid, []).append({"transcript": str(t), "clean": bool(clean), "calls": [n for n, _ in tools],
                                        "reviewer": rec, "reviewer_problems": problems,
                                        "counted": bool(clean and not problems)})
    return out


def counted_entry(entries):
    """The transcript that stands for a packet: its last counted one (a re-run supersedes), else None."""
    good = [e for e in entries if e["counted"]]
    return max(good, key=lambda e: e["reviewer"]["end"] or "") if good else None


def summarise(res, ids, expected=None):
    counted = {i: counted_entry(res.get(i, [])) for i in ids}
    ok = [i for i in ids if counted[i]]
    configs = Counter(config(counted[i]["reviewer"]) for i in ok)
    mixed = len(configs) > 1
    # Without an expected reviewer every counted review must still be the same reviewer: the commonest
    # configuration stands and the rest are sent back for a re-run.
    if mixed:
        keep = configs.most_common(1)[0][0]
        for i in list(ok):
            if config(counted[i]["reviewer"]) != keep:
                counted[i] = None
        ok = [i for i in ids if counted[i]]
    recs = [counted[i]["reviewer"] for i in ok]
    one = config(recs[0]) if recs else None
    tokens = sorted(r["tokens"] for r in recs)
    return {
        "packets": len(ids), "clean": len(ok),
        "unclean": [i for i in ids if i in res and not counted[i]],
        "no_transcript": [i for i in ids if i not in res],
        "wrong_reviewer": sorted(i for i in ids if i in res and not counted[i]
                                 and any(e["clean"] for e in res[i])),
        "reviewer": {
            "expected": expected or None,
            "model": one[0] if one else None, "effort": one[1] if one else None,
            "agent_type": one[2] if one else None,
            "configurations_seen": [{"model": k[0], "effort": k[1], "agent_type": k[2], "packets": n}
                                    for k, n in configs.most_common() if k],
            "launch": dict(Counter(r["launch"] for r in recs)),
            "harness": dict(Counter(v for r in recs for v in r["harness"])),
            "first_review": min((r["start"] for r in recs if r["start"]), default=None),
            "last_review": max((r["end"] for r in recs if r["end"]), default=None),
            "tokens_per_review_median": tokens[len(tokens) // 2] if tokens else None,
            "output_tokens_total": sum(r["output_tokens"] for r in recs),
        },
        "counted_transcripts": {i: counted[i]["transcript"] for i in ok},
    }


def record(summary, batch):
    """The reviewer record of a batch whose every packet has a counted review: what later audits check against."""
    if summary["clean"] != summary["packets"]:
        raise SystemExit(f"{summary['packets'] - summary['clean']} packets have no counted review; no record written")
    r = summary["reviewer"]
    return {"batch": batch, "model": r["model"], "effort": r["effort"], "agent_type": r["agent_type"],
            "launch": r["launch"], "harness": r["harness"], "packets": summary["packets"],
            "first_review": r["first_review"], "last_review": r["last_review"]}


def load_expected(args):
    expected = {}
    if args.reviewer:
        rec = json.loads(Path(args.reviewer).read_text())
        expected = {k: rec[k] for k in KEYS if rec.get(k)}
    for key, value in (("model", args.model), ("effort", args.effort), ("agent_type", args.agent_type)):
        if value:
            expected[key] = value
    return expected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--transcripts", required=True, nargs="+", help="directories, searched recursively")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", help="copy each packet's transcripts here")
    ap.add_argument("--model", help="the model id every review must have been served by")
    ap.add_argument("--effort", help="the reasoning effort every review must have run at")
    ap.add_argument("--agent-type", help="the agent type every review must have run as")
    ap.add_argument("--reviewer", help="reviewer record of a calibration batch (prereg/reviewers/<batch>.json)")
    ap.add_argument("--record", help="write this batch's reviewer record here (needs every packet counted)")
    ap.add_argument("--batch", help="batch name for --record")
    a = ap.parse_args(argv)
    expected = load_expected(a)
    res = audit(a.transcripts, a.prompts, a.replies, expected)
    ids = sorted(p.stem for p in Path(a.prompts).glob("*.md"))
    summary = summarise(res, ids, expected)
    Path(a.out).write_text(json.dumps({**summary, "transcripts": res}, indent=1))
    if a.copy:
        dest = Path(a.copy)
        dest.mkdir(parents=True, exist_ok=True)
        for pid, rs in res.items():
            for k, r in enumerate(rs):
                target = dest / (f"{pid}.jsonl" if k == 0 else f"{pid}.{k}.jsonl")
                if Path(r["transcript"]).resolve() != target.resolve():
                    shutil.copy(r["transcript"], target)
    r = summary["reviewer"]
    print(f"{len(ids)} packets: {summary['clean']} counted, {len(summary['unclean'])} not counted "
          f"({len(summary['wrong_reviewer'])} of them for the reviewer), {len(summary['no_transcript'])} without a "
          f"transcript")
    for c in r["configurations_seen"]:
        print(f"  {c['packets']} by {c['model']}, effort {c['effort']}, agent type {c['agent_type']}")
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        Path(a.record).write_text(json.dumps(record(summary, a.batch or Path(a.record).stem), indent=1) + "\n")
        print(f"reviewer record at {a.record}")
    return summary


if __name__ == "__main__":
    main()
