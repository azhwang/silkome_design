"""
Grounding Checker for Tool-Augmented Reasoning Traces
========================================================

Problem this solves
--------------------
When GPT-5.5 (or any LLM) generates a reasoning trace that narrates over
bioinformatics tool outputs (motif detection, secondary-structure prediction,
hydrophobicity profiles, etc.), the narration is only as trustworthy as its
grounding in what the tools actually returned. An LLM asked to reason over a
tool result will often produce prose that *sounds* right ("the beta-sheet
propensity peaks around residue 340") without that number actually being in
the tool's output. Distilling on ungrounded traces teaches the policy to
sound rigorous rather than reason correctly — arguably worse than the
"simple GPT-generated thinking steps" this is meant to replace.

This module cross-checks specific, checkable claims in a reasoning trace
(numbers, residue positions/ranges, named motifs) against the actual tool
outputs that preceded them in the trace, and produces a per-trace grounding
report plus a batch-level audit summary (hallucination rate, most common
unverifiable claim types) for the Stage-1 pilot-batch review described in
the roadmap.

This is a heuristic / regex-based checker, not a semantic entailment model.
It catches the cheap, common failure modes (fabricated numbers, positions,
motif names) which in practice is most of what matters, but it will miss
higher-level reasoning fabrications (correct numbers, wrong causal claim
connecting them). Pair it with a periodic human or independent-model spot
check rather than trusting it as a complete grounding guarantee.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Trace schema
# ---------------------------------------------------------------------------
#
# A trace is a sequence of steps. Each step is either a tool call (with a
# structured, machine-readable output) or a reasoning narration segment.
# Adapt `TraceStep` to whatever structured format your trace-generation
# pipeline actually emits — this is the contract the checker assumes.

@dataclass
class TraceStep:
    step_type: str                     # "tool_call" or "reasoning"
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    text: Optional[str] = None         # reasoning narration for this step


@dataclass
class Claim:
    kind: str                          # "numeric", "position", "motif"
    raw_text: str                      # the substring making the claim
    value: Any
    step_index: int
    grounded: bool
    role: str = "direct"               # "direct" (asserted as tool-observed
                                        # fact) or "contrastive" (mentioned
                                        # for contrast/negation, e.g. "unlike
                                        # X" or "disrupts Y formation" — not
                                        # a claim the tool observed this)
    evidence: Optional[str] = None     # what tool output (if any) supports it


@dataclass
class GroundingReport:
    claims: List[Claim] = field(default_factory=list)

    @property
    def direct_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.role == "direct"]

    @property
    def contrastive_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.role == "contrastive"]

    @property
    def total_claims(self) -> int:
        """Count of *direct* (tool-observed-fact) claims — the primary metric."""
        return len(self.direct_claims)

    @property
    def grounded_claims(self) -> int:
        return sum(1 for c in self.direct_claims if c.grounded)

    @property
    def grounding_score(self) -> float:
        """
        Primary grounding score, computed over direct claims only.
        Contrastive mentions ("proline-rich turns disrupt beta-sheet
        formation") aren't claims that the tool observed something — they're
        background reasoning that happens to name a term the tools didn't
        report — so they're tracked separately (`contrastive_grounding_score`)
        rather than penalizing the same way a fabricated direct observation
        does.
        """
        if self.total_claims == 0:
            return 1.0  # no checkable direct claims made -> nothing to penalize
        return self.grounded_claims / self.total_claims

    @property
    def contrastive_grounding_score(self) -> Optional[float]:
        """Informational only — None if the trace made no contrastive mentions."""
        cc = self.contrastive_claims
        if not cc:
            return None
        return sum(1 for c in cc if c.grounded) / len(cc)

    @property
    def ungrounded(self) -> List[Claim]:
        return [c for c in self.direct_claims if not c.grounded]


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    r"(?<![\d-])(?P<value>-?\d+\.?\d*)\s*"
    r"(?P<unit>%|percent|kcal(?:/mol)?)?",
)
_POSITION_RE = re.compile(
    r"\b(?:residues?|positions?|spans?|domains?|regions?|turns?)\s+(?:at\s+|of\s+)?(?P<start>\d{1,4})\s*(?:-|to|–)\s*(?P<end>\d{1,4})\b"
    # (?!\.\d) excludes e.g. "residue 0.0194" (a per-residue decimal value,
    # not a residue index) from matching as a bare position claim — the
    # trailing \b alone fires right before the decimal point.
    r"|\b(?:residue|position)\s+(?P<single>\d{1,4})(?!\.\d)\b"
    # Fallback: a bare "N-M" number range with no keyword at all. Real model
    # output often phrases spans this way ("beta_rich_spans at 16-23, 23-35,
    # 32-37..." — only the first gets the "at", the rest are bare), and those
    # are exactly the most specific, most checkable claims in a trace, so
    # under-matching here silently skips the claims that matter most. The
    # trailing lookahead (boundary punctuation/whitespace/end) keeps this
    # from grabbing fragments of longer number sequences or unrelated
    # hyphenated numerics.
    r"|\b(?P<bare_start>\d{1,4})\s*-\s*(?P<bare_end>\d{1,4})\b(?=[\s,.;)]|$)",
    re.IGNORECASE,
)
# Extend this with the actual motif vocabulary your tools can detect.
_KNOWN_MOTIF_TERMS = [
    "poly-alanine", "polyalanine", "poly-ala", "gpgxx", "gpgqq", "gggx",
    "ggx", "beta-sheet", "β-sheet", "beta-turn", "β-turn", "alpha-helix",
    "α-helix", "random coil", "spidroin repeat", "block copolymer",
    "amorphous region", "crystalline region", "intrinsically disordered",
    "proline-rich turn", "proline rich", "glycine-rich spacer",
    "glycine rich",
]
_MOTIF_RE = re.compile(
    "|".join(re.escape(term) for term in _KNOWN_MOTIF_TERMS), re.IGNORECASE
)

# Matches markdown-style enumerated-list markers ("1. ", "2. ") at the start
# of a line. These get masked out (replaced with equal-length whitespace,
# preserving character offsets for every other regex) before claim
# extraction runs — otherwise "1.", "2.", "9." etc. from a numbered
# reasoning trace get extracted as fabricated numeric claims, and worse,
# spuriously "grounded" against small integers that coincidentally appear
# somewhere in the tool output (residue counts, list lengths, etc.), which
# silently inflates the grounding score with meaningless matches.
_LIST_MARKER_RE = re.compile(r"(?m)^(\s*)(\d{1,2})(\.)(\s)")


def _mask_list_markers(text: str) -> str:
    def _blank(m: "re.Match") -> str:
        return m.group(1) + " " * len(m.group(2)) + " " + m.group(4)
    return _LIST_MARKER_RE.sub(_blank, text)

# Cues that mark a mention as contrastive/negated background reasoning
# ("X disrupts Y", "unlike X", "rather than X") rather than a direct claim
# that a tool observed X. Deliberately conservative (biased toward "direct")
# — under-flagging contrastive mentions just means some legitimate background
# reasoning gets counted against the direct-claim score, which is the
# status quo behavior; over-flagging would let real fabrications slip into
# the "don't penalize as hard" bucket, which is the worse failure mode.
_CONTRAST_CUES = [
    "disrupt", "unlike", "rather than", "in contrast", "as opposed to",
    "instead of", "without", "lacks", "lacking", "fails to", "contrary to",
    "differs from", "not consistent with", "does not", "doesn't",
]
_CONTRAST_RE = re.compile("|".join(re.escape(c) for c in _CONTRAST_CUES), re.IGNORECASE)


def _classify_role(text: str, match_start: int, match_end: int) -> str:
    """
    Checks a local window immediately around the claim, not the whole
    enclosing sentence — sentences in real LLM prose often run long and
    contain contrast cues belonging to an entirely different clause (e.g.
    a family-classification aside like "...rather than a dragline
    protein..." earlier in the same sentence as a genuine direct claim
    later on). Whole-sentence detection swept unrelated direct claims into
    the contrastive bucket in exactly that pattern; a tight local window
    (cue immediately before or just after the claim) catches the intended
    "TERM disrupts/lacks/unlike TERM" pattern without that false-positive.
    """
    window = text[max(0, match_start - 60):match_end + 25]
    return "contrastive" if _CONTRAST_RE.search(window) else "direct"


def _extract_numeric_claims(text: str, step_index: int) -> List[Claim]:
    claims = []
    for m in _NUMERIC_RE.finditer(text):
        val = m.group("value")
        if val in (None, "", "-", "."):
            continue
        # Skip bare integers with no unit/context that are very likely just
        # residue counts or list indices mentioned in passing, not a
        # quantitative claim worth checking (heuristic; tune to taste).
        if m.group("unit") is None and "." not in val and len(val) <= 2:
            continue
        try:
            fval = float(val)
        except ValueError:
            continue
        claims.append(
            Claim(kind="numeric", raw_text=m.group(0).strip(), value=fval,
                  step_index=step_index, grounded=False,
                  role=_classify_role(text, m.start(), m.end()))
        )
    return claims


def _extract_position_claims(text: str, step_index: int) -> List[Claim]:
    claims = []
    for m in _POSITION_RE.finditer(text):
        if m.group("single"):
            span = (int(m.group("single")), int(m.group("single")))
        elif m.group("start"):
            span = (int(m.group("start")), int(m.group("end")))
        else:
            span = (int(m.group("bare_start")), int(m.group("bare_end")))
        claims.append(
            Claim(kind="position", raw_text=m.group(0), value=span,
                  step_index=step_index, grounded=False,
                  role=_classify_role(text, m.start(), m.end()))
        )
    return claims


def _extract_motif_claims(text: str, step_index: int) -> List[Claim]:
    claims = []
    for m in _MOTIF_RE.finditer(text):
        claims.append(
            Claim(kind="motif", raw_text=m.group(0), value=m.group(0).lower(),
                  step_index=step_index, grounded=False,
                  role=_classify_role(text, m.start(), m.end()))
        )
    return claims


def extract_claims(text: str, step_index: int) -> List[Claim]:
    text = _mask_list_markers(text)
    return (
        _extract_numeric_claims(text, step_index)
        + _extract_position_claims(text, step_index)
        + _extract_motif_claims(text, step_index)
    )


# ---------------------------------------------------------------------------
# Cross-checking claims against tool outputs
# ---------------------------------------------------------------------------

def _flatten_values(obj: Any) -> List[Any]:
    """Recursively pull every leaf value out of a nested tool-output dict/list."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_values(v))
    else:
        out.append(obj)
    return out


def _numeric_grounded(claim: Claim, tool_output: Dict[str, Any], tol: float = 0.02) -> Optional[str]:
    """
    Finds the *closest* numeric match within tolerance, not the first one
    encountered — with a loose absolute tolerance and "first match wins",
    a claim like 0.954 could spuriously ground against an unrelated
    tool-output field like 0.4851 just because iteration order happened to
    hit it first and it was "close enough". Tolerance itself is now mostly
    relative (scaled to the claimed value's magnitude) with a small
    absolute floor, since these propensity/hydropathy/charge scores mostly
    live in a 0-2 range where a flat 0.5 absolute tolerance made nearly any
    two decimals in that range match each other by coincidence.

    Booleans are explicitly excluded: Python's bool is a subclass of int,
    so without this check a claim of "0" could spuriously ground against
    any `False` value anywhere in the tool output, and "1" against any
    `True`.
    """
    best_value, best_diff = None, None
    for v in _flatten_values(tool_output):
        if isinstance(v, bool):
            continue
        if not isinstance(v, (int, float)):
            continue
        allowed = max(tol, abs(v) * 0.03)  # 3% relative, 0.02 absolute floor
        diff = abs(float(v) - claim.value)
        if diff <= allowed and (best_diff is None or diff < best_diff):
            best_value, best_diff = v, diff
    if best_value is not None:
        return f"matched {best_value} in tool output (diff={best_diff:.4g})"
    return None


def _find_ranges(obj: Any) -> List[Any]:
    """
    Collect [start, end]-shaped pairs *before* they get shredded by
    `_flatten_values` — a raw flatten would explode `[12, 40]` into the
    individual leaves 12 and 40, losing the pairing, so range-shaped lists
    have to be pulled out first at their original nesting level.
    """
    ranges = []
    if isinstance(obj, dict):
        for v in obj.values():
            ranges.extend(_find_ranges(v))
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            ranges.append(obj)
        else:
            for v in obj:
                ranges.extend(_find_ranges(v))
    return ranges


def _position_grounded(claim: Claim, tool_output: Dict[str, Any]) -> Optional[str]:
    start, end = claim.value
    for pair in _find_ranges(tool_output):
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        lo, hi = min(a, b), max(a, b)
        if lo <= start <= hi and lo <= end <= hi:
            return f"within reported range {lo}-{hi}"
    return None


# Maps a natural-language structural term to the tool-output *field name*
# substrings that would substantiate it. Added because tools report
# structured fields like `mean_beta_propensity` / `beta_rich_spans`, not the
# literal string "beta-sheet" — a model correctly reasoning from those
# fields ("the beta-sheet propensity output reports...") was being flagged
# as ungrounded purely because the checker only did literal value matching,
# never looked at field names at all. Extend this alongside
# `_KNOWN_MOTIF_TERMS` and your tools' actual field-naming conventions.
_STRUCTURAL_TERM_KEY_HINTS = {
    "beta-sheet": ["beta_propensity", "beta_rich", "beta_threshold", "beta_sheet"],
    "β-sheet": ["beta_propensity", "beta_rich", "beta_threshold", "beta_sheet"],
    "alpha-helix": ["alpha_propensity", "helix_propensity"],
    "α-helix": ["alpha_propensity", "helix_propensity"],
    "beta-turn": ["turn_propensity", "beta_turn"],
    "β-turn": ["turn_propensity", "beta_turn"],
    # disorder_prediction_chplot (bio_tools.py) reports `predicted_disordered`,
    # never the literal phrase "intrinsically disordered".
    "intrinsically disordered": ["predicted_disordered", "disorder"],
    # The tool's crystalline-domain signal is either the `poly_alanine`
    # motif detection, or beta-sheet propensity (silk crystallites are
    # hydrogen-bonded beta-sheets — a real trace inferred "nanocrystalline
    # regions" directly from reported beta-rich spans) — not a field
    # literally named "crystalline".
    "crystalline region": ["poly_alanine", "poly_ala", "beta_propensity", "beta_rich"],
}


def _flatten_keys(obj: Any) -> List[str]:
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_flatten_keys(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            keys.extend(_flatten_keys(v))
    return keys


def _motif_grounded(claim: Claim, tool_output: Dict[str, Any]) -> Optional[str]:
    def _normalize(s: str) -> str:
        return re.sub(r"[-_\s]+", "", s.lower())

    flat_text = _normalize(" ".join(str(v) for v in _flatten_values(tool_output)))
    if _normalize(claim.value) in flat_text:
        return "motif name present in tool output"

    # Fall back to the field-name synonym mapping — a related tool field or
    # detected-motif *value* exists even though the literal term wasn't
    # reported verbatim. Hints can match either a dict key (e.g.
    # "beta-sheet" -> the key `mean_beta_propensity`) or a value (e.g.
    # "crystalline region" -> the detected motif name `poly_alanine`,
    # which appears as a string inside `motif_detector`'s `motifs` list,
    # not as a key) — checking keys alone misses the latter shape.
    hints = _STRUCTURAL_TERM_KEY_HINTS.get(claim.value.lower())
    if hints:
        flat_keys = [_normalize(k) for k in _flatten_keys(tool_output)]
        for hint in hints:
            hint_n = _normalize(hint)
            if any(hint_n in k for k in flat_keys) or hint_n in flat_text:
                return f"inferred from related tool field/value (matched pattern {hint!r}), not a literal motif name"
    return None


def check_trace_grounding(trace: List[TraceStep], numeric_tolerance: float = 0.02) -> GroundingReport:
    """
    Walks a trace in order, and for each reasoning step, checks its claims
    against the *most recent preceding tool outputs* (all tool calls seen so
    far, most-recent-first) rather than just the single last tool call,
    since a reasoning step often synthesizes across several earlier calls.
    """
    seen_tool_outputs: List[Dict[str, Any]] = []
    all_claims: List[Claim] = []

    for i, step in enumerate(trace):
        if step.step_type == "tool_call" and step.tool_output is not None:
            seen_tool_outputs.append(step.tool_output)
            continue

        if step.step_type != "reasoning" or not step.text:
            continue

        claims = extract_claims(step.text, step_index=i)
        for claim in claims:
            for tool_output in reversed(seen_tool_outputs):
                if claim.kind == "numeric":
                    evidence = _numeric_grounded(claim, tool_output, tol=numeric_tolerance)
                elif claim.kind == "position":
                    evidence = _position_grounded(claim, tool_output)
                else:  # motif
                    evidence = _motif_grounded(claim, tool_output)
                if evidence:
                    claim.grounded = True
                    claim.evidence = evidence
                    break
        all_claims.extend(claims)

    grounded_count = sum(1 for c in all_claims if c.grounded)
    return GroundingReport(claims=all_claims)


# ---------------------------------------------------------------------------
# Batch audit (Stage 1 pilot review)
# ---------------------------------------------------------------------------

@dataclass
class AuditSummary:
    n_traces: int
    mean_grounding_score: float
    fully_grounded_fraction: float     # fraction of traces with score == 1.0
    claims_by_kind: Dict[str, int]         # direct claims only
    ungrounded_by_kind: Dict[str, int]     # direct claims only
    worst_traces: List[int]            # indices of the lowest-scoring traces
    n_contrastive_mentions: int        # informational — not counted in the score above
    contrastive_ungrounded_fraction: Optional[float]  # None if no contrastive mentions at all


def audit_trace_batch(traces: List[List[TraceStep]], n_worst: int = 10) -> AuditSummary:
    """
    Run over a pilot batch and produce the numbers you actually need before
    committing to full-scale trace generation: what fraction of traces are
    fully grounded, which claim types are hallucinated most often, and which
    specific traces to manually read first.

    Contrastive mentions ("X disrupts Y formation") are reported separately
    from the main hallucination-rate numbers, not folded in — see
    `GroundingReport.grounding_score` for why.
    """
    reports = [check_trace_grounding(t) for t in traces]
    scores = [r.grounding_score for r in reports]

    claims_by_kind: Dict[str, int] = {}
    ungrounded_by_kind: Dict[str, int] = {}
    for r in reports:
        for c in r.direct_claims:
            claims_by_kind[c.kind] = claims_by_kind.get(c.kind, 0) + 1
            if not c.grounded:
                ungrounded_by_kind[c.kind] = ungrounded_by_kind.get(c.kind, 0) + 1

    all_contrastive = [c for r in reports for c in r.contrastive_claims]
    contrastive_ungrounded_fraction = (
        sum(1 for c in all_contrastive if not c.grounded) / len(all_contrastive)
        if all_contrastive else None
    )

    worst_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:n_worst]

    return AuditSummary(
        n_traces=len(traces),
        mean_grounding_score=sum(scores) / len(scores) if scores else 1.0,
        fully_grounded_fraction=sum(1 for s in scores if s == 1.0) / len(scores) if scores else 1.0,
        claims_by_kind=claims_by_kind,
        ungrounded_by_kind=ungrounded_by_kind,
        worst_traces=worst_indices,
        n_contrastive_mentions=len(all_contrastive),
        contrastive_ungrounded_fraction=contrastive_ungrounded_fraction,
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trace = [
        TraceStep(
            step_type="tool_call",
            tool_name="motif_detector",
            tool_input={"sequence": "..."},
            tool_output={"motifs": ["poly-alanine", "GGX"], "poly_alanine_span": [12, 40]},
        ),
        TraceStep(
            step_type="reasoning",
            text=(
                "The tool detected a poly-alanine motif spanning residues 12-40, "
                "which is consistent with beta-sheet nanocrystal formation and "
                "contributes to fiber strength. It also claims a hydrophobicity "
                "peak of 3.8 kcal/mol near residue 500, which sounds fabricated."
            ),
        ),
    ]

    report = check_trace_grounding(trace)
    print(f"Grounding score (direct claims): {report.grounding_score:.2f}")
    print(f"Contrastive grounding score (informational): {report.contrastive_grounding_score}")
    for c in report.claims:
        status = "OK " if c.grounded else "FAIL"
        print(f"  [{status}] ({c.kind}, {c.role}) {c.raw_text!r} -> {c.evidence}")

    print("\n=== Batch audit (single-trace demo) ===")
    summary = audit_trace_batch([trace])
    print(summary)
