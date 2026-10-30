"""Audit Stage A subagent transcripts: each blind vetter may Read its own prompt once and Write its own reply once.

    python scripts/stage_a_audit.py --transcripts ~/.claude/projects/<project>/<session>/subagents \\
        --prompts D/prompts --replies D/replies --out D/audit.json [--copy D/transcripts]

A transcript is clean when its tool calls are exactly one Read of prompts/<id>.md and one Write of replies/<id>.txt
(plus the harness's hand-back), in that order. Prompt and reply paths are compared as absolute paths, so pass
absolute directories. Packets with no clean transcript are listed for a re-run.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

HANDBACK = {"SubagentHandback"}
ID = re.compile(r"prompts/([a-z]{16})\.md")


def calls(path):
    first, tools = None, []
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = d.get("message") or {}
        c = m.get("content")
        if first is None and m.get("role") == "user" and isinstance(c, str):
            first = c
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools.append((b.get("name"), b.get("input") or {}))
    return first or "", tools


def audit(transcripts, prompts, replies):
    prompts, replies = Path(prompts).resolve(), Path(replies).resolve()
    out = {}
    for t in sorted(Path(transcripts).glob("*.jsonl")):
        first, tools = calls(t)
        m = ID.search(first)
        if not m or str(prompts) not in first:
            continue
        pid = m.group(1)
        work = [(n, i) for n, i in tools if n not in HANDBACK]
        clean = (len(work) == 2 and work[0][0] == "Read" and work[0][1].get("file_path") == str(prompts / f"{pid}.md")
                 and work[1][0] == "Write" and work[1][1].get("file_path") == str(replies / f"{pid}.txt"))
        out.setdefault(pid, []).append({"transcript": str(t), "clean": bool(clean),
                                        "calls": [n for n, _ in tools]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy")
    a = ap.parse_args(argv)
    res = audit(a.transcripts, a.prompts, a.replies)
    ids = sorted(p.stem for p in Path(a.prompts).glob("*.md"))
    clean = [i for i in ids if any(r["clean"] for r in res.get(i, []))]
    unclean = [i for i in ids if i in res and i not in clean]
    missing = [i for i in ids if i not in res]
    Path(a.out).write_text(json.dumps({"packets": len(ids), "clean": len(clean), "unclean": unclean,
                                       "no_transcript": missing, "transcripts": res}, indent=1))
    if a.copy:
        dest = Path(a.copy)
        dest.mkdir(parents=True, exist_ok=True)
        for pid, rs in res.items():
            for k, r in enumerate(rs):
                shutil.copy(r["transcript"], dest / (f"{pid}.jsonl" if k == 0 else f"{pid}.{k}.jsonl"))
    print(f"{len(ids)} packets: {len(clean)} clean, {len(unclean)} unclean, {len(missing)} without a transcript")


if __name__ == "__main__":
    main()
