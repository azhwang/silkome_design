"""
Staged Multi-Component GRPO Reward Scaffold
==============================================

Implements the reward decomposition from the roadmap:

  1. outcome_accuracy   - structure/property prediction correctness
  2. step_coherence      - does each reasoning step follow from the last,
                           no logical jumps
  3. tool_use            - were tools called and their outputs synthesized
                           correctly (delegates to grounding_checker)
  4. reasoning_quality    - does the chain invoke valid biophysical
                           principles (PRM-bootstrapped, external judge)
  5. calibration          - does the model express appropriate uncertainty
                           (computed at the *group* level across GRPO's
                           k samples per prompt, not per-completion —
                           a single prediction is neither calibrated nor
                           miscalibrated on its own)

Staged, not all-at-once
-------------------------
Turning on all five simultaneously makes a training collapse undiagnosable —
if the policy starts degenerating, you won't know which reward component
it's exploiting. `StagedRewardScheduler` brings components online in an
order chosen by how *reliable* the signal is, not how important it is:

  Stage 1: outcome_accuracy + step_coherence
      Both are cheap and reasonably hard to game superficially. Coherence
      here is a heuristic discourse-structure checker, not a semantic
      judge, so it's stable from day one.

  Stage 2: + tool_use
      Delegates to the grounding checker built for trace auditing —
      already programmatically verifiable, no LLM judge needed, so it's
      trustworthy before you've validated anything about the PRM.

  Stage 3: + reasoning_quality (PRM / GPT-5.5 step judgments)
      Only add this once the PRM's judgments have been checked against
      held-out human or independent-model labels — an unvalidated judge
      steering gradients is how reward hacking sneaks in silently.

  Stage 4: + calibration
      Computed across the group of k samples GRPO already generates per
      prompt, so no extra sampling cost — just needs the trainer to pass
      grouped completions through rather than a flat list.

External judges (GPT-5.5 PRM step scoring) are wired as clearly-marked
stubs — `call_prm_judge()` — plug in your actual API client there. The rest
of this file runs standalone.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Reuses the trace-grounding logic already built for Stage-1 auditing —
# tool_use_reward IS the grounding score, just applied per-completion
# during RL instead of per-batch during pilot review.
from grounding_checker import TraceStep, check_trace_grounding


# ---------------------------------------------------------------------------
# 1. Outcome accuracy (generalized beyond strength/toughness to arbitrary
#    property dicts, so the same reward func works across protein families —
#    consistent with treating silk as one instance of a transferable method)
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _parse_answer_json(text: str) -> Optional[Dict[str, float]]:
    import json
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1).strip())
        return {k: float(v) for k, v in payload.items() if isinstance(v, (int, float, str))}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def outcome_accuracy_reward(
    completions: List[str], ground_truth: List[Dict[str, float]], **kwargs
) -> List[float]:
    """
    ground_truth[i] is a dict of property_name -> true value in [0, 1]
    (e.g. {"strength": 0.55, "toughness": 0.6}, or for a different protein
    family, whatever properties are relevant there — this is family-agnostic
    by design). Reward is the mean per-property accuracy, 0 if unparseable
    or if the predicted keys don't cover the required ones.
    """
    rewards = []
    for text, truth in zip(completions, ground_truth):
        pred = _parse_answer_json(text)
        if pred is None or not set(truth.keys()).issubset(pred.keys()):
            rewards.append(0.0)
            continue
        errors = [abs(pred[k] - v) for k, v in truth.items()]
        rewards.append(max(0.0, 1.0 - sum(errors) / len(errors)))
    return rewards


# ---------------------------------------------------------------------------
# 2. Step coherence (heuristic discourse-structure checker)
# ---------------------------------------------------------------------------

_CONNECTIVES = [
    "because", "therefore", "thus", "since", "as a result", "this suggests",
    "which means", "consequently", "given that", "this indicates",
    "due to", "so that", "leading to",
]
_CONNECTIVE_RE = re.compile("|".join(re.escape(c) for c in _CONNECTIVES), re.IGNORECASE)


def _split_steps(reasoning_text: str) -> List[str]:
    # Split on sentence boundaries only. Collapse internal whitespace first
    # so a wrapped line (single newline mid-sentence, common in prose pulled
    # from a fixed-width source) doesn't get mistaken for a step boundary —
    # only blank-line paragraph breaks and sentence-ending punctuation count.
    normalized = re.sub(r"\n\s*\n", "<PARA>", reasoning_text.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("<PARA>", "\n")
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [p.strip() for p in parts if p.strip()]


def _shares_content_word(a: str, b: str, min_len: int = 5) -> bool:
    """Very crude lexical-overlap check as a proxy for topical continuity."""
    words_a = {w.lower() for w in re.findall(r"[a-zA-Z]+", a) if len(w) >= min_len}
    words_b = {w.lower() for w in re.findall(r"[a-zA-Z]+", b) if len(w) >= min_len}
    return len(words_a & words_b) > 0


def step_coherence_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Heuristic proxy for "does each step follow from the previous one":
    for each consecutive pair of reasoning steps, credit either an explicit
    logical connective (because/therefore/since/...) or lexical continuity
    (shared content words) as evidence the step isn't an unmotivated jump.
    Single-step or missing reasoning gets a neutral 0.5, not 0 — coherence
    isn't well-defined with nothing to connect. This is deliberately a
    cheap, gameable-but-directionally-useful signal for Stage 1; swap in
    `call_prm_judge` scoring here once your PRM is validated (Stage 3).
    """
    rewards = []
    for text in completions:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if not reasoning_match:
            rewards.append(0.0)
            continue
        steps = _split_steps(reasoning_match.group(1))
        if len(steps) < 2:
            rewards.append(0.5)
            continue
        connected = 0
        for a, b in zip(steps, steps[1:]):
            if _CONNECTIVE_RE.search(b) or _shares_content_word(a, b):
                connected += 1
        rewards.append(connected / (len(steps) - 1))
    return rewards


# ---------------------------------------------------------------------------
# 3. Tool-use correctness (delegates to the grounding checker)
# ---------------------------------------------------------------------------

def tool_use_reward(
    completions: List[str], structured_traces: List[List[TraceStep]], **kwargs
) -> List[float]:
    """
    `structured_traces[i]` is the parsed tool-call/reasoning-step sequence
    for completion i (see grounding_checker.TraceStep) — your inference
    pipeline needs to hand back structured steps, not just the flat text,
    for this to work. If completions only come back as flat strings with no
    structured tool-call record, this reward can't be computed and you
    should keep tool-use out of the active reward set until the inference
    harness captures structured traces.
    """
    rewards = []
    for trace in structured_traces:
        report = check_trace_grounding(trace)
        rewards.append(report.grounding_score)
    return rewards


# ---------------------------------------------------------------------------
# 4. Reasoning quality (PRM / external judge — stub)
# ---------------------------------------------------------------------------

def call_prm_judge(step_text: str, context: Dict[str, Any]) -> float:
    """
    STUB — replace with your actual GPT-5.5 (or trained PRM) call.
    Should return a per-step score in [0, 1] reflecting whether the step
    invokes valid chemical/biophysical reasoning given `context` (e.g. the
    protein sequence, family, and prior steps).

    Before wiring this into training (Stage 3), validate it: run it against
    a held-out set of steps with independent human or separate-model labels
    and check agreement — an unvalidated judge steering RL gradients is
    exactly the kind of silent reward-hacking vector this staged approach
    is meant to avoid.
    """
    raise NotImplementedError(
        "Wire this up to your GPT-5.5 judge / trained PRM before enabling Stage 3."
    )


def reasoning_quality_reward(
    completions: List[str],
    contexts: List[Dict[str, Any]],
    judge_fn: Callable[[str, Dict[str, Any]], float] = call_prm_judge,
    **kwargs,
) -> List[float]:
    rewards = []
    for text, context in zip(completions, contexts):
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if not reasoning_match:
            rewards.append(0.0)
            continue
        steps = _split_steps(reasoning_match.group(1))
        if not steps:
            rewards.append(0.0)
            continue
        step_scores = [judge_fn(step, context) for step in steps]
        rewards.append(sum(step_scores) / len(step_scores))
    return rewards


# ---------------------------------------------------------------------------
# 5. Calibration (group-level, Brier-style — needs GRPO's k samples/prompt)
# ---------------------------------------------------------------------------

_CONFIDENCE_RE = re.compile(
    r"confiden\w*\s*(?:of|:|is)?\s*(?P<pct>\d{1,3})\s*%"
    r"|(?P<qual>low|moderate|high|very high|very low)\s+confidence",
    re.IGNORECASE,
)

_QUALITATIVE_MAP = {"very low": 0.1, "low": 0.3, "moderate": 0.5, "high": 0.75, "very high": 0.9}


def _extract_expressed_confidence(text: str) -> Optional[float]:
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return None
    if m.group("pct"):
        return max(0.0, min(1.0, int(m.group("pct")) / 100.0))
    if m.group("qual"):
        return _QUALITATIVE_MAP.get(m.group("qual").lower())
    return None


def calibration_reward_grouped(
    grouped_completions: List[List[str]],
    grouped_ground_truth: List[List[Dict[str, float]]],
    property_key: str,
) -> List[List[float]]:
    """
    Group-level calibration: for each prompt's group of k GRPO samples,
    bucket samples by their expressed confidence and compare bucketed
    confidence against bucketed empirical accuracy (Brier-score style),
    rather than trying to score a single completion's calibration in
    isolation, which isn't a well-posed quantity.

    grouped_completions[i]   = the k completions for prompt i
    grouped_ground_truth[i]  = the k (identical, repeated) ground-truth
                               dicts for prompt i's completions
    property_key             = which predicted property to check calibration
                               against, e.g. "strength"

    Returns rewards in the same [prompt][sample] nested shape as the input,
    so it drops in as one more term per-completion once flattened back out.
    Completions with no expressed confidence get a neutral 0.5 — you can't
    penalize calibration the model never attempted to express, though you
    may separately want to reward *attempting* calibration at all as part
    of outcome/format shaping.
    """
    all_rewards: List[List[float]] = []
    for completions, truths in zip(grouped_completions, grouped_ground_truth):
        confidences = []
        errors = []
        parsed_ok = []
        for text, truth in zip(completions, truths):
            conf = _extract_expressed_confidence(text)
            pred = _parse_answer_json(text)
            if conf is None or pred is None or property_key not in pred:
                confidences.append(None)
                errors.append(None)
                parsed_ok.append(False)
                continue
            confidences.append(conf)
            errors.append(abs(pred[property_key] - truth[property_key]))
            parsed_ok.append(True)

        group_rewards = []
        for conf, err, ok in zip(confidences, errors, parsed_ok):
            if not ok:
                group_rewards.append(0.5)
                continue
            # empirical correctness proxy: error below 0.15 counts as "was right"
            was_right = 1.0 if err <= 0.15 else 0.0
            # Brier-style: reward is higher when expressed confidence tracks
            # actual correctness (high confidence + right, or low confidence
            # + wrong), penalized when they diverge.
            brier = (conf - was_right) ** 2
            group_rewards.append(max(0.0, 1.0 - brier))
        all_rewards.append(group_rewards)
    return all_rewards


# ---------------------------------------------------------------------------
# Staged scheduler
# ---------------------------------------------------------------------------

@dataclass
class StageWeights:
    outcome: float = 0.0
    coherence: float = 0.0
    tool_use: float = 0.0
    reasoning_quality: float = 0.0
    calibration: float = 0.0


_STAGE_CONFIG: Dict[int, StageWeights] = {
    1: StageWeights(outcome=0.6, coherence=0.4),
    2: StageWeights(outcome=0.5, coherence=0.25, tool_use=0.25),
    3: StageWeights(outcome=0.4, coherence=0.15, tool_use=0.15, reasoning_quality=0.3),
    4: StageWeights(outcome=0.35, coherence=0.1, tool_use=0.15, reasoning_quality=0.25, calibration=0.15),
}


class StagedRewardScheduler:
    """
    Usage:
        scheduler = StagedRewardScheduler(stage=1)
        reward_funcs, reward_weights = scheduler.get_reward_funcs()
        config = GRPOConfig(..., reward_weights=reward_weights)
        trainer = GRPOTrainer(..., reward_funcs=reward_funcs, args=config)

        # once Stage 1 is stable:
        scheduler = StagedRewardScheduler(stage=2)
        ... re-instantiate trainer with the new reward_funcs/weights ...

    Calibration (stage 4) is intentionally NOT included in the flat
    reward_funcs list — its group-level computation doesn't fit TRL's
    per-completion reward_func signature cleanly. Call
    `calibration_reward_grouped` separately inside your training loop and
    add it to the other rewards before the GRPO advantage computation; see
    the note in `get_reward_funcs`.
    """

    def __init__(self, stage: int, judge_fn: Callable[[str, Dict[str, Any]], float] = call_prm_judge):
        if stage not in _STAGE_CONFIG:
            raise ValueError(f"stage must be one of {sorted(_STAGE_CONFIG)}, got {stage}")
        self.stage = stage
        self.weights = _STAGE_CONFIG[stage]
        self.judge_fn = judge_fn

    def get_reward_funcs(self):
        funcs = [outcome_accuracy_reward, step_coherence_reward]
        weights = [self.weights.outcome, self.weights.coherence]

        if self.stage >= 2:
            funcs.append(tool_use_reward)
            weights.append(self.weights.tool_use)

        if self.stage >= 3:
            def _reasoning_quality_bound(completions, contexts, **kwargs):
                return reasoning_quality_reward(completions, contexts, judge_fn=self.judge_fn, **kwargs)
            funcs.append(_reasoning_quality_bound)
            weights.append(self.weights.reasoning_quality)

        if self.stage >= 4:
            # Reminder, not a callable: see class docstring — wire
            # calibration_reward_grouped in separately at the group level.
            pass

        return funcs, weights


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    completion = """<reasoning>
    The sequence has a high glycine/alanine content in the crystalline
    region, which suggests beta-sheet nanocrystal formation. Because these
    nanocrystals act as physical crosslinks, this increases fiber strength.
    The semi-amorphous GGX-rich region, in contrast, contributes
    extensibility, which is consistent with high toughness. I estimate this
    with moderate confidence given limited species-specific data.
    </reasoning>
    <answer>
    {"strength": 0.62, "toughness": 0.58}
    </answer>"""

    ground_truth = {"strength": 0.6, "toughness": 0.55}

    print("outcome_accuracy_reward:", outcome_accuracy_reward([completion], [ground_truth]))
    print("step_coherence_reward:  ", step_coherence_reward([completion]))

    print("\n=== Stage 1 scheduler ===")
    scheduler = StagedRewardScheduler(stage=1)
    funcs, weights = scheduler.get_reward_funcs()
    print("funcs:", [f.__name__ for f in funcs], "weights:", weights)
    # All reward_funcs share the **kwargs signature TRL uses (every func gets
    # the full kwarg set and just ignores what it doesn't need), so they can
    # be called uniformly here too.
    shared_kwargs = dict(completions=[completion], ground_truth=[ground_truth])
    combined = sum(w * f(**shared_kwargs)[0] for f, w in zip(funcs, weights))
    print("combined stage-1 reward:", combined)

    print("\n=== Group-level calibration (stage 4 component) demo ===")
    grouped_completions = [[completion, completion.replace("moderate", "high")]]
    grouped_truth = [[ground_truth, ground_truth]]
    cal_rewards = calibration_reward_grouped(grouped_completions, grouped_truth, property_key="strength")
    print("calibration rewards per group:", cal_rewards)
