"""
ORPO Preference Pair Construction: Isolating Reasoning Quality from Answer
Correctness
=============================================================================

The naive approach — "prefer traces with grounded reasoning, dispreferred
otherwise" — silently confounds two things that are usually correlated:
answer correctness and reasoning quality. If you just sample good vs. bad
traces from the wild, ORPO will partly learn "prefer whichever trace looks
more thorough," which correlates with but isn't the same as "prefer
grounded reasoning," and it *can't* learn to penalize lucky guessing
specifically, because it never sees a case where a shortcut trace got the
right answer and a grounded trace didn't.

This module builds four buckets per prompt, then constructs three distinct
pair types that each isolate a different piece of signal:

  Bucket        answer correct?   grounded reasoning?
  ------------  ----------------  --------------------
  A: grounded-correct     yes             yes
  B: shortcut-correct     yes             no
  C: grounded-incorrect   no              yes
  D: shortcut-incorrect   no              no

  Pair type 1 (same-answer contrast): A vs B, matched on final answer.
      Isolates reasoning quality as a signal independent of correctness,
      since both sides reached the same conclusion.

  Pair type 2 (anti-shortcut contrast): C vs D.
      Prefers a wrong-but-grounded chain over a right-but-shortcut chain.
      This is the pair type that actually teaches the model not to reward
      lucky guessing. It's also the pair type that fights against naive
      "prefer correct answers" instincts, so it's exposed as a tunable
      mixing ratio (`anti_shortcut_ratio`) rather than included at full
      volume by default — flooding ORPO training with these can measurably
      hurt outcome accuracy if not balanced.

  Pair type 3 (standard contrast): A vs D (or B vs C when needed as filler).
      The default correctness-dominant pair, included for volume and to
      keep the overall preference direction aligned with getting the right
      answer most of the time.

Traces are expected to already be scored (grounding_score, correctness) —
see grounding_checker.py for how to produce the grounding score, and
whatever comparison-to-ground-truth logic you use for correctness.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

@dataclass
class ScoredTrace:
    prompt_id: str
    trace_text: str              # full rendered trace (reasoning + answer)
    final_answer: Any            # parsed answer, comparable via `answers_match`
    is_correct: bool             # answer matches ground truth (within tolerance)
    grounding_score: float       # from grounding_checker.GroundingReport, [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Bucket(Enum):
    GROUNDED_CORRECT = "grounded_correct"
    SHORTCUT_CORRECT = "shortcut_correct"
    GROUNDED_INCORRECT = "grounded_incorrect"
    SHORTCUT_INCORRECT = "shortcut_incorrect"


@dataclass
class PreferencePair:
    prompt_id: str
    chosen: str
    rejected: str
    pair_type: str
    chosen_meta: Dict[str, Any] = field(default_factory=dict)
    rejected_meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_trace(trace: ScoredTrace, grounding_threshold: float = 0.7) -> Bucket:
    grounded = trace.grounding_score >= grounding_threshold
    if trace.is_correct and grounded:
        return Bucket.GROUNDED_CORRECT
    if trace.is_correct and not grounded:
        return Bucket.SHORTCUT_CORRECT
    if not trace.is_correct and grounded:
        return Bucket.GROUNDED_INCORRECT
    return Bucket.SHORTCUT_INCORRECT


def bucket_traces(
    traces: List[ScoredTrace], grounding_threshold: float = 0.7
) -> Dict[str, Dict[Bucket, List[ScoredTrace]]]:
    """Groups traces by prompt_id, then by bucket within each prompt."""
    by_prompt: Dict[str, Dict[Bucket, List[ScoredTrace]]] = {}
    for t in traces:
        buckets = by_prompt.setdefault(
            t.prompt_id, {b: [] for b in Bucket}
        )
        buckets[bucket_trace(t, grounding_threshold)].append(t)
    return by_prompt


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

DEFAULT_ANSWERS_MATCH: Callable[[Any, Any], bool] = lambda a, b: a == b


@dataclass
class PairConstructionConfig:
    grounding_threshold: float = 0.7
    max_pairs_per_prompt: int = 4
    anti_shortcut_ratio: float = 0.25   # fraction of a prompt's pair budget
                                          # reserved for type-2 (anti-shortcut)
                                          # pairs; keep this a minority so ORPO
                                          # doesn't systematically learn to
                                          # prefer wrong answers
    same_answer_tolerance: Optional[Callable[[Any, Any], bool]] = None
    seed: int = 0


class PairConstructor:
    def __init__(self, config: Optional[PairConstructionConfig] = None):
        self.config = config or PairConstructionConfig()
        self._rng = random.Random(self.config.seed)
        self._answers_match = self.config.same_answer_tolerance or DEFAULT_ANSWERS_MATCH

    def build_pairs(self, traces: List[ScoredTrace]) -> List[PreferencePair]:
        by_prompt = bucket_traces(traces, self.config.grounding_threshold)
        all_pairs: List[PreferencePair] = []
        for prompt_id, buckets in by_prompt.items():
            all_pairs.extend(self._build_pairs_for_prompt(prompt_id, buckets))
        return all_pairs

    def _build_pairs_for_prompt(
        self, prompt_id: str, buckets: Dict[Bucket, List[ScoredTrace]]
    ) -> List[PreferencePair]:
        A = buckets[Bucket.GROUNDED_CORRECT]
        B = buckets[Bucket.SHORTCUT_CORRECT]
        C = buckets[Bucket.GROUNDED_INCORRECT]
        D = buckets[Bucket.SHORTCUT_INCORRECT]

        budget = self.config.max_pairs_per_prompt
        anti_shortcut_budget = max(0, round(budget * self.config.anti_shortcut_ratio))
        remaining_budget = budget - anti_shortcut_budget

        pairs: List[PreferencePair] = []

        # --- Pair type 1: same-answer contrast (A vs B) -------------------
        same_answer_pairs = self._same_answer_pairs(prompt_id, A, B)
        pairs.extend(same_answer_pairs)

        # --- Pair type 2: anti-shortcut contrast (C vs D) ------------------
        anti_shortcut_pairs = self._cartesian_pairs(
            prompt_id, C, D, pair_type="anti_shortcut", limit=anti_shortcut_budget
        )
        pairs.extend(anti_shortcut_pairs)

        # --- Pair type 3: standard contrast (A vs D, fallback B vs C) ------
        remaining_budget = max(0, remaining_budget - len(same_answer_pairs))
        standard_pairs = self._cartesian_pairs(
            prompt_id, A, D, pair_type="standard", limit=remaining_budget
        )
        if len(standard_pairs) < remaining_budget:
            fallback = self._cartesian_pairs(
                prompt_id, B, C, pair_type="standard_fallback",
                limit=remaining_budget - len(standard_pairs),
            )
            standard_pairs.extend(fallback)
        pairs.extend(standard_pairs)

        return pairs[:budget] if budget else pairs

    def _same_answer_pairs(
        self, prompt_id: str, correct_grounded: List[ScoredTrace], correct_shortcut: List[ScoredTrace]
    ) -> List[PreferencePair]:
        pairs = []
        for g in correct_grounded:
            for s in correct_shortcut:
                if self._answers_match(g.final_answer, s.final_answer):
                    pairs.append(self._make_pair(prompt_id, chosen=g, rejected=s, pair_type="same_answer"))
        self._rng.shuffle(pairs)
        return pairs

    def _cartesian_pairs(
        self, prompt_id: str, preferred: List[ScoredTrace], dispreferred: List[ScoredTrace],
        pair_type: str, limit: int,
    ) -> List[PreferencePair]:
        if limit <= 0 or not preferred or not dispreferred:
            return []
        candidates = [
            (p, d) for p in preferred for d in dispreferred
        ]
        self._rng.shuffle(candidates)
        return [
            self._make_pair(prompt_id, chosen=p, rejected=d, pair_type=pair_type)
            for p, d in candidates[:limit]
        ]

    @staticmethod
    def _make_pair(prompt_id: str, chosen: ScoredTrace, rejected: ScoredTrace, pair_type: str) -> PreferencePair:
        return PreferencePair(
            prompt_id=prompt_id,
            chosen=chosen.trace_text,
            rejected=rejected.trace_text,
            pair_type=pair_type,
            chosen_meta={"correct": chosen.is_correct, "grounding": chosen.grounding_score},
            rejected_meta={"correct": rejected.is_correct, "grounding": rejected.grounding_score},
        )


# ---------------------------------------------------------------------------
# Dataset-level summary (sanity-check the pair mix before training)
# ---------------------------------------------------------------------------

def summarize_pairs(pairs: List[PreferencePair]) -> Dict[str, Any]:
    """
    Check this before handing pairs to ORPO. In particular:
      - `anti_shortcut` should be a clear minority (per anti_shortcut_ratio)
      - fraction of pairs where the *chosen* side is actually incorrect
        should roughly match the anti_shortcut fraction, not exceed it —
        if it's much higher, something upstream is misclassifying traces
    """
    counts: Dict[str, int] = {}
    chosen_incorrect = 0
    for p in pairs:
        counts[p.pair_type] = counts.get(p.pair_type, 0) + 1
        if not p.chosen_meta.get("correct", True):
            chosen_incorrect += 1
    return {
        "total_pairs": len(pairs),
        "by_type": counts,
        "chosen_incorrect_fraction": chosen_incorrect / len(pairs) if pairs else 0.0,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    traces = [
        ScoredTrace("p1", "grounded reasoning, answer=0.6", final_answer=0.6, is_correct=True, grounding_score=0.9),
        ScoredTrace("p1", "shortcut reasoning, answer=0.6", final_answer=0.6, is_correct=True, grounding_score=0.2),
        ScoredTrace("p1", "grounded reasoning, answer=0.4 (wrong)", final_answer=0.4, is_correct=False, grounding_score=0.85),
        ScoredTrace("p1", "shortcut reasoning, answer=0.6 (right, lucky)", final_answer=0.6, is_correct=True, grounding_score=0.1),
        ScoredTrace("p2", "grounded reasoning, answer=0.3", final_answer=0.3, is_correct=True, grounding_score=0.95),
        ScoredTrace("p2", "shortcut reasoning, answer=0.1 (wrong)", final_answer=0.1, is_correct=False, grounding_score=0.15),
    ]

    constructor = PairConstructor(PairConstructionConfig(max_pairs_per_prompt=4, anti_shortcut_ratio=0.25))
    pairs = constructor.build_pairs(traces)
    for p in pairs:
        print(f"[{p.pair_type}] prompt={p.prompt_id} chosen={p.chosen!r} rejected={p.rejected!r}")

    print("\nSummary:", summarize_pairs(pairs))
