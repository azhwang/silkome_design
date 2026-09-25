"""
Stage 2/3: Real ORPO Trace Pool Generation
=============================================

`orpo_pair_construction.py`'s `PairConstructor` needs multiple traces that
share the same `prompt_id` to build any pair at all — same-answer contrasts
and cartesian bucket contrasts are both computed *within* one prompt's
traces, never across prompts. The Stage 1 pilot batch
(`pilot_batch_driver.py`) generated exactly one trace per sequence, so it
has zero real ORPO pairs latent in it: every prompt_id there is unique.

This script closes that gap: for a sample of N sequences from the real
silkome dataset, generate K completions each (K calls per prompt, same
prompt — the reasoning-tier model's temperature=1 default gives natural
variation in phrasing, grounding, and predicted values across repeated
calls), score each completion for grounding (grounding_checker.py) and
outcome correctness (reusing the same tolerance logic as
staged_grpo_rewards.outcome_accuracy_reward), then hand the whole pool to
PairConstructor to build real preference pairs.

Cost note: this is K times as many real LLM calls per sequence as the
Stage 1 pilot batch (default K=4 -> 4x). Start with a small N via
--n-prompts before committing to a large real run — same reasoning as the
20-sample-before-200-sample pattern used for the Stage 1 pilot.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pilot_batch_driver import load_dataset, _synthetic_demo_dataset, with_retry
from trace_generation import (
    SilkSample,
    GeneratedTrace,
    generate_trace_for_sample,
    generate_shortcut_trace_for_sample,
    call_trace_generator_llm,
    mock_trace_generator_llm,
    LLMNotConfiguredError,
)
from staged_grpo_rewards import _parse_answer_json
from orpo_pair_construction import (
    ScoredTrace,
    PairConstructor,
    PairConstructionConfig,
    bucket_trace,
    summarize_pairs,
)


# ---------------------------------------------------------------------------
# Correctness + same-answer tolerance
#
# These mirror staged_grpo_rewards.outcome_accuracy_reward's error math
# ([0, 1]-normalized properties, mean absolute error) rather than
# reinventing a second definition of "correct" that could silently drift
# from what Stage 4's reward function actually rewards.
# ---------------------------------------------------------------------------

def _mean_abs_error(pred: Dict[str, float], truth: Dict[str, float]) -> Optional[float]:
    if pred is None or not set(truth.keys()).issubset(pred.keys()):
        return None
    errors = [abs(pred[k] - truth[k]) for k in truth]
    return sum(errors) / len(errors)


def _make_same_answer_tolerance(tol: float):
    def _close(a: Any, b: Any) -> bool:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return a == b
        common = set(a) & set(b)
        if not common:
            return False
        return all(abs(a[k] - b[k]) <= tol for k in common)
    return _close


# ---------------------------------------------------------------------------
# Pool generation: K completions per prompt
# ---------------------------------------------------------------------------

def generate_scored_pool(
    samples: List[SilkSample],
    k_samples: int,
    k_shortcut_samples: int,
    llm_fn,
    correctness_tolerance: float,
    sleep_between_calls: float = 0.0,
    only: Optional[Set[Tuple[str, str, int]]] = None,
) -> (List[ScoredTrace], List[Dict[str, Any]]):
    """Returns (scored_traces, errors). Skips samples with no ground_truth —
    correctness is undefined without it, and every downstream bucket
    depends on it.

    If `only` is given, generates just those (prompt_id, trace_kind, k)
    slots and skips everything else — used by --resume-from to backfill
    calls that failed in an earlier run.

    Generates two kinds of completion per prompt: k_samples tool-grounded
    completions (generate_trace_for_sample) and k_shortcut_samples
    tool-withheld completions (generate_shortcut_trace_for_sample) — see
    trace_generation.py for why the shortcut variant exists (a real
    12-completion sanity run found the tool-grounded prompt alone never
    produces low-grounding traces at temperature=1, so
    SHORTCUT_CORRECT/SHORTCUT_INCORRECT buckets would otherwise stay
    permanently empty). Bucket assignment is still computed purely from
    the measured grounding_score, never assumed from which prompt
    generated it — trace_kind in metadata is for transparency/debugging
    only, not an input to bucketing."""
    scored: List[ScoredTrace] = []
    errors: List[Dict[str, Any]] = []

    usable = [s for s in samples if s.ground_truth]
    skipped_no_truth = len(samples) - len(usable)
    if skipped_no_truth:
        print(f"[generate_scored_pool] skipping {skipped_no_truth} sample(s) with no ground_truth "
              f"(correctness undefined)", file=sys.stderr)

    generators = [
        ("grounded_prompt", generate_trace_for_sample, k_samples),
        ("shortcut_prompt", generate_shortcut_trace_for_sample, k_shortcut_samples),
    ]

    for i, sample in enumerate(usable):
        prompt_id = f"sample_{i}"
        for trace_kind, gen_fn, n_calls in generators:
            for k in range(n_calls):
                if only is not None and (prompt_id, trace_kind, k) not in only:
                    continue
                try:
                    g: GeneratedTrace = gen_fn(sample, llm_fn=llm_fn)
                except LLMNotConfiguredError:
                    raise
                except Exception as e:  # noqa: BLE001 - real client errors are unpredictable
                    print(f"[generate_scored_pool] prompt {prompt_id} ({trace_kind} k={k}) failed: {e}",
                          file=sys.stderr)
                    errors.append({"prompt_id": prompt_id, "trace_kind": trace_kind, "k": k, "error": str(e)})
                    if sleep_between_calls:
                        time.sleep(sleep_between_calls)
                    continue

                pred = _parse_answer_json(g.llm_completion)
                mae = _mean_abs_error(pred, sample.ground_truth)
                is_correct = mae is not None and mae <= correctness_tolerance

                scored.append(ScoredTrace(
                    prompt_id=prompt_id,
                    trace_text=g.llm_completion,
                    final_answer=pred,
                    is_correct=is_correct,
                    grounding_score=g.grounding_report.grounding_score,
                    metadata={
                        "sequence": sample.sequence,
                        "protein_category": sample.protein_category,
                        "ground_truth": sample.ground_truth,
                        "predicted": pred,
                        "mean_abs_error": mae,
                        "k_index": k,
                        "trace_kind": trace_kind,
                    },
                ))
                if sleep_between_calls:
                    time.sleep(sleep_between_calls)

    return scored, errors


# ---------------------------------------------------------------------------
# Diagnostics: are we actually getting bucket diversity within a prompt?
#
# If temperature=1 doesn't produce enough variation, most prompts will land
# every one of their K completions in the same bucket, and PairConstructor
# will silently return near-zero pairs. Surfacing this before trusting an
# empty/small pair count saves a confusing "why are there no pairs" dead end.
# ---------------------------------------------------------------------------

def _bucket_diversity_report(scored: List[ScoredTrace], grounding_threshold: float) -> Dict[str, Any]:
    by_prompt: Dict[str, set] = {}
    for t in scored:
        by_prompt.setdefault(t.prompt_id, set()).add(bucket_trace(t, grounding_threshold).value)
    n_prompts = len(by_prompt)
    n_multi_bucket = sum(1 for buckets in by_prompt.values() if len(buckets) > 1)
    return {
        "n_prompts": n_prompts,
        "n_prompts_with_multiple_buckets": n_multi_bucket,
        "fraction_with_diversity": n_multi_bucket / n_prompts if n_prompts else 0.0,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_orpo_pair_generation(
    data_path: Optional[str],
    n_prompts: int,
    k_samples: int,
    k_shortcut_samples: int,
    output_path: str,
    use_mock: bool,
    seed: int = 0,
    sleep_between_calls: float = 0.0,
    max_retries: int = 3,
    correctness_tolerance: float = 0.1,
    grounding_threshold: float = 0.7,
    max_pairs_per_prompt: int = 4,
    anti_shortcut_ratio: float = 0.25,
    same_answer_tolerance: float = 0.05,
    resume_from: Optional[str] = None,
) -> None:
    if data_path:
        dataset = load_dataset(data_path)
    else:
        print("[run_orpo_pair_generation] no --data-path given, using built-in synthetic demo dataset",
              file=sys.stderr)
        dataset = _synthetic_demo_dataset()

    rng = random.Random(seed)
    with_truth = [s for s in dataset if s.ground_truth]
    batch = with_truth if len(with_truth) <= n_prompts else rng.sample(with_truth, n_prompts)
    print(f"[run_orpo_pair_generation] {len(batch)} prompts x ({k_samples} grounded + "
          f"{k_shortcut_samples} shortcut) samples each "
          f"({len(with_truth)}/{len(dataset)} dataset rows have ground_truth)", file=sys.stderr)

    base_llm_fn = mock_trace_generator_llm if use_mock else call_trace_generator_llm
    llm_fn = with_retry(base_llm_fn, max_retries=max_retries) if not use_mock else base_llm_fn

    previously_scored: List[ScoredTrace] = []
    only: Optional[Set[Tuple[str, str, int]]] = None
    if resume_from:
        prior = json.loads(Path(resume_from).read_text(encoding="utf-8"))
        if prior["n_prompts_requested"] != len(batch) or prior["k_samples"] != k_samples \
                or prior["k_shortcut_samples"] != k_shortcut_samples:
            raise SystemExit(
                f"--resume-from {resume_from} was generated with n_prompts={prior['n_prompts_requested']}, "
                f"k_samples={prior['k_samples']}, k_shortcut_samples={prior['k_shortcut_samples']}; this run "
                f"uses {len(batch)}/{k_samples}/{k_shortcut_samples}. Pass the same --n-prompts/--k-*/--seed/--data-path."
            )
        # prompt_id is the index into the seeded batch, so the resumed batch must reproduce the
        # original one exactly — otherwise new traces would be filed under the wrong sequence.
        for t in prior["scored_traces"]:
            idx = int(t["prompt_id"].split("_")[1])
            if idx >= len(batch) or batch[idx].sequence != t["metadata"]["sequence"]:
                raise SystemExit(
                    f"--resume-from {resume_from}: {t['prompt_id']} has a different sequence than this run's "
                    f"batch. Use the same --seed and --data-path as the original run."
                )
        previously_scored = [
            ScoredTrace(
                prompt_id=t["prompt_id"],
                trace_text=t["trace_text"],
                final_answer=t["final_answer"],
                is_correct=t["is_correct"],
                grounding_score=t["grounding_score"],
                metadata=t["metadata"],
            )
            for t in prior["scored_traces"]
        ]
        only = {(e["prompt_id"], e["trace_kind"], e["k"]) for e in prior["errors"]}
        print(f"[run_orpo_pair_generation] resuming: {len(previously_scored)} traces kept, "
              f"backfilling {len(only)} previously failed calls", file=sys.stderr)

    new_scored, errors = generate_scored_pool(
        batch, k_samples, k_shortcut_samples, llm_fn, correctness_tolerance, sleep_between_calls, only=only
    )
    scored = previously_scored + new_scored

    diversity = _bucket_diversity_report(scored, grounding_threshold)

    config = PairConstructionConfig(
        grounding_threshold=grounding_threshold,
        max_pairs_per_prompt=max_pairs_per_prompt,
        anti_shortcut_ratio=anti_shortcut_ratio,
        same_answer_tolerance=_make_same_answer_tolerance(same_answer_tolerance),
        seed=seed,
    )
    constructor = PairConstructor(config)
    pairs = constructor.build_pairs(scored)
    summary = summarize_pairs(pairs)

    output = {
        "n_prompts_requested": len(batch),
        "k_samples": k_samples,
        "k_shortcut_samples": k_shortcut_samples,
        "n_scored_traces": len(scored),
        "n_errors": len(errors),
        "errors": errors,
        "bucket_diversity": diversity,
        "pair_summary": summary,
        "pairs": [
            {
                "prompt_id": p.prompt_id,
                "pair_type": p.pair_type,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "chosen_meta": p.chosen_meta,
                "rejected_meta": p.rejected_meta,
            }
            for p in pairs
        ],
        "scored_traces": [
            {
                "prompt_id": t.prompt_id,
                "final_answer": t.final_answer,
                "is_correct": t.is_correct,
                "grounding_score": t.grounding_score,
                "bucket": bucket_trace(t, grounding_threshold).value,
                "metadata": t.metadata,
                "trace_text": t.trace_text,
            }
            for t in scored
        ],
    }

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n=== ORPO pair generation complete ===")
    print(f"Scored traces: {len(scored)} ({len(batch)} prompts x up to {k_samples} grounded + "
          f"{k_shortcut_samples} shortcut samples, errors: {len(errors)})")
    print(f"Prompts with bucket diversity: {diversity['n_prompts_with_multiple_buckets']}/{diversity['n_prompts']} "
          f"({diversity['fraction_with_diversity']:.0%})")
    print(f"Pairs built: {summary['total_pairs']} — by type: {summary['by_type']}")
    print(f"Chosen-incorrect fraction: {summary['chosen_incorrect_fraction']:.2f} "
          f"(should roughly match anti_shortcut_ratio={anti_shortcut_ratio}, not exceed it)")
    print(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2/3: generate K real completions per prompt and build real ORPO preference pairs."
    )
    parser.add_argument("--data-path", type=str, default=None,
                         help="Path to a .csv/.json/.jsonl silkome dataset file. Omit to use the built-in synthetic demo.")
    parser.add_argument("--n-prompts", type=int, default=20, help="Number of distinct sequences/prompts (default 20).")
    parser.add_argument("--k-samples", type=int, default=4,
                         help="Tool-grounded completions to generate per prompt (default 4).")
    parser.add_argument("--k-shortcut-samples", type=int, default=4,
                         help="Tool-withheld 'shortcut' completions to generate per prompt (default 4) — see "
                              "trace_generation.build_shortcut_prompt for why these exist: real data showed the "
                              "tool-grounded prompt alone never produces low-grounding traces at temperature=1.")
    parser.add_argument("--output", type=str, default="orpo_pairs.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls (rate limiting).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--correctness-tolerance", type=float, default=0.1,
                         help="Max mean-abs-error vs. ground_truth to count as correct (default 0.1).")
    parser.add_argument("--grounding-threshold", type=float, default=0.7)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=4)
    parser.add_argument("--anti-shortcut-ratio", type=float, default=0.25)
    parser.add_argument("--same-answer-tolerance", type=float, default=0.05,
                         help="Max per-property abs diff to count two predictions as 'the same answer' (default 0.05).")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Path to an earlier run's output JSON: keep its scored traces and regenerate only "
                              "the calls listed in its 'errors'. Requires the same --data-path/--n-prompts/--k-*/--seed "
                              "as that run (checked). Pass a different --output to keep the original untouched.")
    parser.add_argument("--use-mock", action="store_true",
                         help="Use the deterministic mock LLM instead of the real client — for testing the driver "
                              "itself. Note: the mock always returns the same completion, so no bucket diversity "
                              "or pairs are expected in mock mode — it only verifies the wiring doesn't error.")
    args = parser.parse_args()

    run_orpo_pair_generation(
        data_path=args.data_path,
        n_prompts=args.n_prompts,
        k_samples=args.k_samples,
        k_shortcut_samples=args.k_shortcut_samples,
        output_path=args.output,
        use_mock=args.use_mock,
        seed=args.seed,
        sleep_between_calls=args.sleep,
        max_retries=args.max_retries,
        correctness_tolerance=args.correctness_tolerance,
        grounding_threshold=args.grounding_threshold,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        anti_shortcut_ratio=args.anti_shortcut_ratio,
        same_answer_tolerance=args.same_answer_tolerance,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
