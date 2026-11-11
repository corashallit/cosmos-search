"""The review audit (scripts/stage_a_audit.py): what each reviewer did and which reviewer it was, the commit that
takes the audit, and the paired comparison of two reviewers on the same packets (scripts/slow_cal.py compare)."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit_mod = _load("stage_a_audit")

A, B, C = "aaaabbbbccccdddd", "eeeeffffgggghhhh", "iiiijjjjkkkkllll"
MODEL, OTHER = "model-new-1", "model-old-1"


def transcript(path, pid, d, model=MODEL, asked=None, effort="xhigh", agent="general-purpose", workflow=True,
               forwarded=False, api_error=False, extra_call=None, second_model=None, user_request=None):
    """A subagent transcript in the harness's shape: the launch text, then a Read turn and a Write turn.
    user_request: the launching turn's user message, which a newer harness hands to the subagent first."""
    launch = f"Read the file {d}/prompts/{pid}.md and write to {d}/replies/{pid}.txt"
    if workflow:
        launch = "[Workflow harness — computed task] ...\n  " + launch
    lines = []
    if user_request:
        lines.append({"type": "user", "timestamp": "2026-11-10T22:00:00Z", "message": {
            "role": "user", "content": "[Workflow harness — user request] The harness relays, verbatim ...:\n  "
                                       + user_request}})
    lines.append({"type": "user", "timestamp": "2026-11-10T22:00:00Z", "message": {"role": "user", "content": launch}})
    if forwarded:
        lines.append({"type": "attachment", "timestamp": "2026-10-21T01:00:00Z",
                      "attachment": {"type": "queued_command", "prompt": "a message from the thread"}})

    def turn(name, inp, req, m, stamp):
        return {"type": "assistant", "timestamp": stamp, "requestId": req, "requestedModel": asked or model,
                "effort": effort, "attributionAgent": agent, "version": "9.9.9",
                "message": {"role": "assistant", "model": m,
                            "usage": {"input_tokens": 2, "cache_read_input_tokens": 1000, "output_tokens": 50},
                            "content": [{"type": "tool_use", "name": name, "input": inp}]}}

    lines.append(turn("Read", {"file_path": f"{d}/prompts/{pid}.md"}, "r1", model, "2026-11-10T22:00:05Z"))
    if extra_call:
        lines.append(turn(extra_call, {"command": "ls"}, "r1b", model, "2026-11-10T22:00:06Z"))
    if api_error:
        lines.append({"type": "assistant", "timestamp": "2026-11-10T22:00:07Z", "isApiErrorMessage": True,
                      "message": {"role": "assistant", "model": "<synthetic>", "content": []}})
    lines.append(turn("Write", {"file_path": f"{d}/replies/{pid}.txt"}, "r2", second_model or model,
                      "2026-11-10T22:01:00Z"))
    if not workflow:
        lines.append(turn("SubagentHandback", {"message": "written"}, "r3", second_model or model,
                          "2026-11-10T22:01:02Z"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(x) for x in lines) + "\n")


@pytest.fixture
def batch(tmp_path):
    d = tmp_path.resolve()
    for sub in ("prompts", "replies", "t/workflows/wf_1"):
        (d / sub).mkdir(parents=True)
    for pid in (A, B, C):
        (d / "prompts" / f"{pid}.md").write_text("prompt")
    return d


def run(d, *extra):
    return audit_mod.main(["--transcripts", str(d / "t"), "--prompts", str(d / "prompts"), "--replies",
                           str(d / "replies"), "--out", str(d / "audit.json"), *extra])


def test_audit_counts_reviews_by_the_named_model(batch):
    for pid in (A, B):
        transcript(batch / "t/workflows/wf_1" / f"agent-{pid}.jsonl", pid, batch)
    transcript(batch / "t" / f"agent-{C}.jsonl", C, batch, workflow=False)     # Agent tool, with its hand-back
    s = run(batch, "--model", MODEL, "--effort", "xhigh", "--agent-type", "general-purpose")
    assert s["clean"] == 3 and not s["unclean"] and not s["no_transcript"]
    r = s["reviewer"]
    assert (r["model"], r["effort"], r["agent_type"]) == (MODEL, "xhigh", "general-purpose")
    assert r["launch"] == {"workflow": 2, "agent tool": 1}
    assert r["tokens_per_review_median"] == 2 * 1052          # per request, not per transcript line


def test_audit_rejects_another_model_a_fallback_and_a_mixed_review(batch):
    transcript(batch / "t" / "a.jsonl", A, batch, model=OTHER)                       # launched on another model
    transcript(batch / "t" / "b.jsonl", B, batch, model=OTHER, asked=MODEL)          # asked for one, served another
    transcript(batch / "t" / "c.jsonl", C, batch, second_model=OTHER)                # changed part way
    s = run(batch, "--model", MODEL)
    assert s["clean"] == 0 and s["wrong_reviewer"] == [A, B, C]
    res = json.loads((batch / "audit.json").read_text())["transcripts"]
    assert any("expected" in p for p in res[A][0]["reviewer_problems"])
    assert any("asked for" in p for p in res[B][0]["reviewer_problems"])
    assert any("more than one model" in p for p in res[C][0]["reviewer_problems"])
    assert all(e["clean"] and not e["counted"] for i in (A, B, C) for e in res[i])


def test_audit_rejects_effort_copied_messages_api_errors_and_extra_calls(batch):
    transcript(batch / "t" / "a.jsonl", A, batch, effort="high")
    transcript(batch / "t" / "b.jsonl", B, batch, forwarded=True)
    transcript(batch / "t" / "c.jsonl", C, batch, api_error=True)
    s = run(batch, "--model", MODEL, "--effort", "xhigh")
    assert s["clean"] == 0 and len(s["wrong_reviewer"]) == 3
    transcript(batch / "t" / "c.jsonl", C, batch, extra_call="Bash")
    s = run(batch, "--model", MODEL, "--effort", "xhigh")
    assert C in s["unclean"] and C not in s["wrong_reviewer"]                         # the tool calls, not the model


def test_audit_without_an_expected_model_wants_one_reviewer_for_the_batch(batch):
    transcript(batch / "t" / "a.jsonl", A, batch)
    transcript(batch / "t" / "b.jsonl", B, batch)
    transcript(batch / "t" / "c.jsonl", C, batch, model=OTHER)
    s = run(batch)
    assert s["clean"] == 2 and s["wrong_reviewer"] == [C] and s["reviewer"]["model"] == MODEL
    assert len(s["reviewer"]["configurations_seen"]) == 2


def test_rerun_supersedes_and_record_round_trips(batch):
    transcript(batch / "t" / "a1.jsonl", A, batch, model=OTHER, asked=MODEL)         # fallback, then a re-run
    transcript(batch / "t" / "a2.jsonl", A, batch)
    transcript(batch / "t" / "b.jsonl", B, batch)
    transcript(batch / "t" / "c.jsonl", C, batch)
    rec = batch / "reviewers" / "cal-x.json"
    s = run(batch, "--model", MODEL, "--record", str(rec), "--copy", str(batch / "kept"))
    assert s["clean"] == 3 and s["counted_transcripts"][A].endswith("a2.jsonl")
    assert sorted(p.name for p in (batch / "kept").iterdir()) == [f"{A}.1.jsonl", f"{A}.jsonl", f"{B}.jsonl",
                                                                   f"{C}.jsonl"]
    r = json.loads(rec.read_text())
    assert (r["batch"], r["model"], r["effort"], r["packets"]) == ("cal-x", MODEL, "xhigh", 3)
    assert run(batch, "--reviewer", str(rec))["clean"] == 3                           # a later batch, same reviewer
    transcript(batch / "t" / "c.jsonl", C, batch, model=OTHER)
    assert run(batch, "--reviewer", str(rec))["wrong_reviewer"] == [C]
    with pytest.raises(SystemExit):                                                    # no record for a partial batch
        run(batch, "--reviewer", str(rec), "--record", str(batch / "reviewers" / "y.json"))


def test_commit_with_audit_fails_uncounted_reviews_and_records_the_reviewer(batch):
    from cosmos_search.vetting.__main__ import main as cli

    packets = [{"packet_id": pid, "text": "t", "targets": []} for pid in (A, B, C)]
    (batch / "products.json").write_text(json.dumps(packets))
    for pid in (A, B, C):
        (batch / "replies" / f"{pid}.txt").write_text(json.dumps(
            {"packet_id": pid, "class_probs": {"eclipsing_binary": 1.0}}))
    transcript(batch / "t" / "a.jsonl", A, batch)
    transcript(batch / "t" / "b.jsonl", B, batch)
    transcript(batch / "t" / "c.jsonl", C, batch, model=OTHER, asked=MODEL)           # a reply, by the wrong model
    run(batch, "--model", MODEL)
    cli(["commit", "--packets", str(batch / "products.json"), "--replies", str(batch / "replies"), "--vetter", "t",
         "--batch", str(batch / "batch"), "--audit", str(batch / "audit.json"), "--allow-dirty"])
    rec = json.loads((batch / "batch" / "commit.json").read_text())
    assert rec["reviewer"]["model"] == MODEL and rec["reviews_not_counted"] == 1
    v = {x["packet_id"]: x for x in map(json.loads, (batch / "batch" / "verdicts.jsonl").read_text().splitlines())}
    assert v[A]["ok"] and v[B]["ok"] and not v[C]["ok"] and "asked for" in v[C]["error"]


def test_pending_lists_prompts_without_a_reply(batch, capsys):
    from cosmos_search.vetting.__main__ import main as cli

    (batch / "replies" / f"{B}.txt").write_text("{}")
    cli(["pending", "--prompts", str(batch / "prompts"), "--replies", str(batch / "replies"), "--chunks", "2"])
    assert json.loads(capsys.readouterr().out) == [[A], [C]]


def test_review_launch_script_names_no_model():
    """The launch script takes the model and effort as arguments; the reviewer's name lives in prereg/reviewers."""
    js = (ROOT / "scripts" / "stage_a_reviews.js").read_text()
    assert "agentType: 'general-purpose', model, effort" in js and "claude-" not in js


def launch_script(check_answer, extra_args=""):
    """Run scripts/stage_a_reviews.js as the Workflow tool would, with its hooks replaced: the check subagent
    answers `check_answer` and every review returns "written". Returns the script's result and the agent calls."""
    js = (ROOT / "scripts" / "stage_a_reviews.js").read_text().replace("export const meta", "const meta")
    harness = f"""
const calls = [];
const args = {{dir: '/b', ids: ['aaaa', 'bbbb'], model: 'some-model', effort: 'xhigh'{extra_args}}};
const phase = () => {{}}, log = () => {{}};
const parallel = thunks => Promise.all(thunks.map(t => t()));
const agent = async (prompt, opts) => {{
  calls.push({{label: opts.label, model: opts.model, agentType: opts.agentType || null, schema: !!opts.schema}});
  return opts.schema ? {check_answer} : 'written';
}};
(async () => {{ {js} }})().then(out => console.log(JSON.stringify({{out, calls}})));
"""
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the launch script")
def test_launch_script_stops_before_any_review_when_a_thread_message_reaches_its_subagents():
    clean = "{messages_before_task: 0, user_request_before_task: false, first_words: ''}"
    r = launch_script(clean)
    assert r["out"] == {"launched": 2, "returned": 2, "lost": []}
    assert [c["label"] for c in r["calls"]] == ["launch check", "aaaa", "bbbb"]
    # the check is not a reviewer: no general-purpose agent, not the reviewer's model; the reviews are
    assert r["calls"][0]["agentType"] is None and r["calls"][0]["model"] != "some-model" and r["calls"][0]["schema"]
    assert all(c["agentType"] == "general-purpose" and c["model"] == "some-model" for c in r["calls"][1:])

    for seen in ("{messages_before_task: 1, user_request_before_task: true, first_words: '[Workflow harness'}",
                 "{messages_before_task: 1, user_request_before_task: false, first_words: ''}",   # inconsistent
                 "null"):                                                                         # check died
        r = launch_script(seen)
        assert r["out"]["launched"] == 0 and "stopped" in r["out"] and len(r["calls"]) == 1

    r = launch_script("null", ", check: false")                       # asked to skip the check
    assert r["out"]["launched"] == 2 and [c["label"] for c in r["calls"]] == ["aaaa", "bbbb"]


def test_audit_can_record_a_batch_reviewed_before_the_check(batch):
    transcript(batch / "t" / "a.jsonl", A, batch, workflow=False, forwarded=True)
    transcript(batch / "t" / "b.jsonl", B, batch, workflow=False)
    transcript(batch / "t" / "c.jsonl", C, batch, workflow=False)
    rec = batch / "reviewers" / "old.json"
    s = run(batch, "--allow-thread-messages", "--record", str(rec))
    assert s["clean"] == 3 and json.loads(rec.read_text())["thread_messages_in_context"] == 1
    assert run(batch, "--reviewer", str(rec))["wrong_reviewer"] == [A]            # a new batch is held to the rule


def test_agent_files_and_launch_name_the_adopted_reviewer():
    """The model and effort in .claude/agents/ are those of the calibration batch whose reviewer is adopted, and
    that batch's record was made from its transcripts."""
    from cosmos_search.vetting import reviewer

    rec = reviewer.record()
    assert rec["batch"] == reviewer.ADOPTED and rec["model"] and rec["effort"] and rec["packets"] > 0
    for path in reviewer.AGENT_FILES:
        fm = reviewer.frontmatter(path)
        assert (fm["model"], fm["effort"]) == (rec["model"], rec["effort"]), path.name
    args = reviewer.launch_args("/tmp/x", [A, B])
    assert (args["model"], args["effort"], args["ids"]) == (rec["model"], rec["effort"], [A, B])
    for name in ("verdicts.jsonl", "commit.json"):                                  # the batch it names is archived
        assert (ROOT / "prereg" / "verdicts" / reviewer.ADOPTED / name).exists()


# --- two reviewers on the same packets ---------------------------------------------------------------------------

slow_cal = _load("slow_cal")


def _table(rng, s_fake, s_nat, n_fake=50, n_nat=322):
    rows = []
    for i in range(n_fake):
        rows.append({"packet_id": f"f{i}", "kind": "fake", "label": "fake", "flavour": "grey", "snr": 30.0,
                     "s": float(np.clip(rng.normal(s_fake, 0.2), 0, 1))})
    for i in range(n_nat):
        rows.append({"packet_id": f"n{i}", "kind": "natural" if i % 8 else "chance", "label": "RS" if i % 2 else "LPV",
                     "flavour": None, "snr": np.nan, "s": float(np.clip(rng.normal(s_nat, 0.1), 0, 1))})
    df = pd.DataFrame(rows)
    return df.assign(ok=True, shortlisted=df["s"] >= 0.35, top="x", recognised_as=None)


def test_compare_same_reviewer_twice_chooses_the_new_model():
    rng = np.random.default_rng(5)     # each one-sided test trips for 1 seed in 20, as it should; this one does not
    old, new = _table(rng, 0.7, 0.2), _table(rng, 0.7, 0.2)
    out = slow_cal.compare(old, new, None)
    assert out["outcome"]["code"] == "new_model" and out["new"]["R1_recall"]["pass"]
    p = out["paired"]["natural_and_chance"]["shortlist"]
    assert p["n"] == 322 and p["both"] + p["only_old"] + p["only_new"] + p["neither"] == 322
    assert p["old"] == p["both"] + p["only_old"] and p["p_new_more"] > 0.05


def test_compare_flags_a_reviewer_that_shortlists_more_natural_stars_or_misses_fakes():
    rng = np.random.default_rng(2)
    old = _table(rng, 0.7, 0.2)
    worse = slow_cal.compare(old, _table(rng, 0.7, 0.3), None)
    assert worse["outcome"]["code"] == "old_model_rerun" and worse["outcome"]["more_natural_shortlisted"]
    blind = slow_cal.compare(old, _table(rng, 0.3, 0.2), None)
    assert blind["outcome"]["code"] == "old_model_rerun" and not blind["outcome"]["new_passes_R1"]
    better = slow_cal.compare(old, _table(rng, 0.7, 0.1), None)
    assert better["outcome"]["code"] == "new_model" and better["paired"]["natural_and_chance"]["shortlist"][
        "p_new_fewer"] < 0.05


def test_compare_refuses_different_packets_and_reports_a_sensitivity_copy():
    rng = np.random.default_rng(3)
    old, new = _table(rng, 0.7, 0.2), _table(rng, 0.7, 0.2)
    with pytest.raises(ValueError):
        slow_cal.compare(old, new.iloc[1:], None)
    out = slow_cal.compare(old, new, None, without=["n0", "n1", "f0"])
    assert out["without"]["left_out"] == 3
    assert out["without"]["paired"]["natural_and_chance"]["shortlist"]["n"] == 320
    assert out["without"]["old"]["R1_recall"]["n"] == 49


def test_repeat_subset_is_reproducible_and_scatter_is_measured():
    ids = [f"p{i:03d}" for i in range(432)]
    a = slow_cal.repeat_ids(ids)
    assert len(a) == 87 and a == slow_cal.repeat_ids(list(reversed(ids))) and set(a) < set(ids)
    rng = np.random.default_rng(4)
    first = _table(rng, 0.7, 0.2)
    second = first.assign(s=np.clip(first["s"] + rng.normal(0, 0.05, len(first)), 0, 1))
    second = second.assign(shortlisted=second["s"] >= 0.35).iloc[::5]
    out = slow_cal.repeat_scatter(first, second, other=_table(rng, 0.7, 0.2))
    assert out["packets"] == len(second) and 0.03 < out["s_scatter_per_review"] < 0.07
    assert out["other_reviewer_same_packets"]["shortlist"]["n"] == len(second)


def test_failed_review_counts_against_the_rule():
    from cosmos_search.vetting.verdict import failed

    keys = {p: {"kind": k, "label": "x", "g": 15.0, "candidate_id": 1, "truth": {},
                "measured": {"colour": {}, "neowise": {}, "ztf": {}}} for p, k in (("a", "fake"), ("b", "natural"))}
    df = slow_cal.table([failed("a", "t", "no reply"), failed("b", "t", "no reply")], keys).set_index("packet_id")
    assert not df.loc["a", "shortlisted"] and df.loc["b", "shortlisted"]


def test_audit_finds_a_review_that_was_handed_the_launching_turns_user_message(batch):
    """Harness 2.1.322: the subagent's first message is the user message that started the launching turn. The
    review is found (by the launch text that follows) and does not count as blind."""
    for pid in (A, B):
        transcript(batch / "t/workflows/wf_1" / f"agent-{pid}.jsonl", pid, batch)
    transcript(batch / "t/workflows/wf_1" / f"agent-{C}.jsonl", C, batch,
               user_request=f"go ahead; and look at {batch}/prompts/{A}.md first")     # names another file
    s = run(batch, "--model", MODEL)
    assert s["clean"] == 2 and s["wrong_reviewer"] == [C] and not s["no_transcript"]
    res = json.loads((batch / "audit.json").read_text())["transcripts"]
    assert len(res[A]) == 1 and res[C][0]["clean"] and res[C][0]["reviewer"]["relayed_user_request"] == 1
    assert "user message of the turn that launched" in res[C][0]["reviewer_problems"][0]
    assert res[C][0]["reviewer"]["launch"] == "workflow"
    s = run(batch, "--model", MODEL, "--allow-thread-messages")
    assert s["clean"] == 3 and s["reviewer"]["thread_messages_in_context"] == 1
