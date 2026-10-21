"""Run Stage A over a batch of packets, commit, reveal and score.

    # 1. Prompts for every packet (one file each), for whichever backend runs the experts
    python -m cosmos_search.vetting prompts --packets products.json --out run/prompts

    # 2a. Run them through the Messages API with no tools (needs ANTHROPIC_API_KEY)
    python -m cosmos_search.vetting api --packets products.json --replies run/replies --model <model id>
    # 2b. ...or have an orchestrating session's no-tool subagents write run/replies/<packet_id>.txt

    # 3. Parse and commit (verdicts become immutable)
    python -m cosmos_search.vetting commit --packets products.json --replies run/replies --vetter <label> --batch run/batch

    # 4. Only now open the key and score
    python -m cosmos_search.vetting score --batch run/batch --key reveal.json.age --out run/report.json

Steps 1-3 never read the key, so they can run in the same session that will later reveal it.
"""

import argparse
import json
from pathlib import Path

from .commit import commit, load_verdicts, reveal
from .report import score
from .verdict import SCHEMA_TEXT, failed, parse

STAGE_A = (Path(__file__).with_name("stage_a.md")).read_text()


def prompt(packet):
    """System text and user text for one packet. The same text goes to every backend."""
    return STAGE_A, packet["text"] + "\n\n## Reply format\n" + SCHEMA_TEXT


def _packets(path):
    return json.loads(Path(path).read_text())


def cmd_prompts(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in _packets(args.packets):
        system, user = prompt(p)
        (out / f"{p['packet_id']}.md").write_text(f"{system}\n\n---\n\n{user}\n")
    print(f"wrote prompts to {out}")


def cmd_api(args):
    import anthropic  # optional dependency, only for this backend

    client = anthropic.Anthropic()
    out = Path(args.replies)
    out.mkdir(parents=True, exist_ok=True)
    for p in _packets(args.packets):
        dest = out / f"{p['packet_id']}.txt"
        if dest.exists():
            continue
        system, user = prompt(p)
        # No tools parameter: the expert can only read the packet and answer.
        msg = client.messages.create(model=args.model, max_tokens=args.max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}])
        dest.write_text("".join(b.text for b in msg.content if b.type == "text"))
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        (out / f"{p['packet_id']}.usage.json").write_text(json.dumps(usage))
    print(f"replies in {out}")


def cmd_commit(args):
    verdicts = []
    for p in _packets(args.packets):
        f = Path(args.replies) / f"{p['packet_id']}.txt"
        verdicts.append(parse(f.read_text(), p, args.vetter) if f.exists() else failed(p["packet_id"], args.vetter, "no reply"))
    rec = commit(verdicts, args.batch)
    print(json.dumps(rec, indent=2))


def cmd_score(args):
    keys = reveal(args.batch, args.key)
    rep = score(load_verdicts(args.batch), keys)
    Path(args.out).write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(f"report at {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cosmos_search.vetting")
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("prompts")
    s.add_argument("--packets", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_prompts)
    s = sub.add_parser("api")
    s.add_argument("--packets", required=True)
    s.add_argument("--replies", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--max-tokens", type=int, default=4000)
    s.set_defaults(fn=cmd_api)
    s = sub.add_parser("commit")
    s.add_argument("--packets", required=True)
    s.add_argument("--replies", required=True)
    s.add_argument("--vetter", required=True)
    s.add_argument("--batch", required=True)
    s.set_defaults(fn=cmd_commit)
    s = sub.add_parser("score")
    s.add_argument("--batch", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
