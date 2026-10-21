"""The structured verdict a Stage A expert returns, and its validation.

Experts reply with one JSON object (schema in SCHEMA_TEXT, which goes into the
prompt). parse() accepts the reply text, pulls out the JSON, and normalises the
class probabilities. A reply that cannot be parsed is recorded as a failed
verdict rather than retried silently: format failures are part of the measured
cost and reliability of a vetter.
"""

import json
import math
import re
from dataclasses import asdict, dataclass, field

from . import taxonomy

SCHEMA_TEXT = """Reply with exactly one JSON object and nothing else:
{
  "packet_id": "<as given>",
  "predictions": {"<target id>": <probability 0-1 for binary targets, or [low, high] for interval targets>, ...},
  "class_probs": {"<class>": <probability>, ...},
  "explanation": "<at most 120 words: the single most probable explanation and why>",
  "evidence": [{"feature": "<what in the packet>", "favours": "<class>", "against": "<class or null>"}],
  "discriminating_test": "<one cheap observation or archival check that would most change your probabilities>",
  "recognised_as": "<a specific known object you think this is, or null>"
}
class_probs uses only the class names listed in the packet, may omit classes you give zero, and must sum to 1.
Give "unexplained" the probability that none of the listed explanations fits; do not reserve it for exotic ideas."""


@dataclass
class Verdict:
    packet_id: str
    class_probs: dict
    predictions: dict = field(default_factory=dict)
    explanation: str = ""
    evidence: list = field(default_factory=list)
    discriminating_test: str = ""
    recognised_as: str | None = None
    vetter: str = ""            # vetter configuration label (model, prompt version, lens)
    ok: bool = True
    error: str = ""

    @property
    def groups(self):
        return taxonomy.group_probs(self.class_probs)

    @property
    def p_unexplained(self):
        return float(self.class_probs.get("unexplained", 0.0))

    def top(self):
        return max(self.class_probs, key=self.class_probs.get) if self.class_probs else None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def failed(packet_id, vetter, error):
    return Verdict(packet_id=packet_id, class_probs={}, vetter=vetter, ok=False, error=error)


def _extract_json(text):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _prob(x):
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"bad probability {x}")
    return x


def parse(text, packet, vetter=""):
    """Parse an expert reply against its packet. Never raises: failures come back as ok=False."""
    pid = packet["packet_id"]
    try:
        d = _extract_json(text)
        if str(d.get("packet_id", pid)) != pid:
            raise ValueError(f"reply is for packet {d.get('packet_id')!r}")
        raw = {str(k): _prob(v) for k, v in (d.get("class_probs") or {}).items()}
        unknown = sorted(set(raw) - set(taxonomy.LEAVES))
        if unknown:
            raise ValueError(f"unknown classes {unknown}")
        total = sum(raw.values())
        if not total > 0:
            raise ValueError("class probabilities sum to zero")
        if abs(total - 1) > 0.05:
            raise ValueError(f"class probabilities sum to {total:.3f}")
        probs = {k: v / total for k, v in raw.items() if v > 0}

        preds = {}
        kinds = {t["id"]: t["kind"] for t in packet.get("targets", [])}
        for tid, v in (d.get("predictions") or {}).items():
            if tid not in kinds:
                continue
            if kinds[tid] == "binary":
                preds[tid] = min(max(_prob(v), 0.0), 1.0)
            else:
                lo, hi = sorted(float(x) for x in v)
                preds[tid] = [lo, hi]
        rec = d.get("recognised_as")
        return Verdict(packet_id=pid, class_probs=probs, predictions=preds,
                       explanation=str(d.get("explanation", ""))[:2000], evidence=list(d.get("evidence") or [])[:20],
                       discriminating_test=str(d.get("discriminating_test", ""))[:500],
                       recognised_as=None if rec in (None, "", "null", "none") else str(rec), vetter=vetter)
    except (ValueError, TypeError, KeyError) as e:
        return failed(pid, vetter, f"{type(e).__name__}: {e}")
