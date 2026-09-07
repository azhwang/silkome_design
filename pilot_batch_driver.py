"""
Pilot Batch Driver
=====================

The actual Stage 1 script: pull a real batch of sequences from your silkome
dataset, run each through the trace-generation pipeline
(tools -> prompt -> LLM -> structured trace -> grounding check), save
everything to disk, and print the aggregate audit summary — the number that
tells you whether the trace-generation prompt needs iteration before you
commit to full-scale generation.

Usage
------
    # Real run, once call_trace_generator_llm is wired to your GPT-5.5 client:
    python3 pilot_batch_driver.py --data-path silkome_dataset.csv --n-samples 200

    # Test the driver itself today, no dataset file or API key needed:
    python3 pilot_batch_driver.py --use-mock

Dataset format
---------------
CSV or JSON/JSONL, with columns/keys: sequence, family, genus, species,
protein_category, and optionally strength, toughness (ground truth, if
available — omit if you're generating traces for unlabeled sequences).
Column names are matched case-insensitively; adjust `_COLUMN_ALIASES` below
if your actual dataset uses different names (the handoff doc's silkome
dataset schema wasn't confirmed against the real file for this driver).
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from trace_generation import (
    SilkSample,
    GeneratedTrace,
    generate_trace_for_sample,
    generate_pilot_batch,
    call_trace_generator_llm,
    mock_trace_generator_llm,
    audit_trace_batch,
    LLMNotConfiguredError,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "sequence": ["sequence", "protein_sequence", "fasta", "seq"],
    "family": ["family"],
    "genus": ["genus"],
    "species": ["species"],
    "protein_category": ["protein_category", "category", "spidroin_type"],
    "strength": ["strength"],
    "toughness": ["toughness"],
}


def _resolve_column(row_keys: List[str], target: str) -> Optional[str]:
    lowered = {k.lower(): k for k in row_keys}
    for alias in _COLUMN_ALIASES[target]:
        if alias in lowered:
            return lowered[alias]
    return None


def _row_to_sample(row: Dict[str, Any]) -> Optional[SilkSample]:
    keys = list(row.keys())
    seq_col = _resolve_column(keys, "sequence")
    if seq_col is None or not row.get(seq_col):
        return None

    def get(target, default=""):
        col = _resolve_column(keys, target)
        return row.get(col, default) if col else default

    ground_truth = None
    strength_val, toughness_val = get("strength", None), get("toughness", None)
    if strength_val not in (None, "") and toughness_val not in (None, ""):
        try:
            ground_truth = {"strength": float(strength_val), "toughness": float(toughness_val)}
        except (TypeError, ValueError):
            ground_truth = None

    return SilkSample(
        sequence=str(row[seq_col]).strip(),
        family=str(get("family")),
        genus=str(get("genus")),
        species=str(get("species")),
        protein_category=str(get("protein_category")),
        ground_truth=ground_truth,
    )


def load_dataset(path: str) -> List[SilkSample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif p.suffix.lower() in (".json", ".jsonl"):
        with open(p, encoding="utf-8") as f:
            if p.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                rows = data if isinstance(data, list) else data.get("data", [])
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix} (expected .csv, .json, or .jsonl)")

    samples = []
    skipped = 0
    for row in rows:
        sample = _row_to_sample(row)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if skipped:
        print(f"[load_dataset] skipped {skipped} row(s) with no resolvable sequence column", file=sys.stderr)
    if not samples:
        raise ValueError(
            f"No usable rows found in {path}. Check that a sequence-like column exists "
            f"(tried aliases: {_COLUMN_ALIASES['sequence']}) — or edit _COLUMN_ALIASES to match your schema."
        )
    return samples


def _synthetic_demo_dataset() -> List[SilkSample]:
    """Fallback dataset for --use-mock / no --data-path, so the driver is runnable today."""
    return [
        SilkSample(
            sequence=(
                "SSSAYSGTSTGGSSVSQSQPIISSAPVYFNAQTLTSSLASSLQSDRALNFISSGQLSASDVS"
                "TSVSSAVAQTLGISQSSVQNIISQQMSSVRTGASSSSVSQAIANAVSSAVQASGAATPGQEQ"
                "SIAQRVYSSISTYLSQLISQRTAPAPAPAPRPAPMPAPAPRPAPMPAPAPRPRPAPAPRPAP"
            ),
            family="Araneidae", genus="Eriophora", species="pustulosa",
            protein_category="PySp", ground_truth={"strength": 0.5, "toughness": 0.6},
        ),
        SilkSample(
            sequence="GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA" * 3,
            family="Araneidae", genus="Synthetic", species="demo",
            protein_category="MaSp", ground_truth={"strength": 0.8, "toughness": 0.4},
        ),
    ]


# ---------------------------------------------------------------------------
# Retry wrapper (real API calls will occasionally fail transiently)
# ---------------------------------------------------------------------------

def with_retry(
    llm_fn: Callable[[str], str], max_retries: int = 3, base_delay: float = 2.0
) -> Callable[[str], str]:
    def wrapped(prompt: str) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                return llm_fn(prompt)
            except LLMNotConfiguredError:
                raise  # don't retry a missing API key or missing package
            except Exception as e:  # noqa: BLE001 - real client errors are unpredictable
                last_error = e
                delay = base_delay * (2 ** attempt)
                print(f"[with_retry] attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.0f}s...",
                      file=sys.stderr)
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {max_retries} attempts") from last_error

    return wrapped


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _serialize_generated(g: GeneratedTrace) -> Dict[str, Any]:
    report = g.grounding_report
    return {
        "sample": asdict(g.sample),
        "llm_completion": g.llm_completion,
        "grounding_score": report.grounding_score,
        "contrastive_grounding_score": report.contrastive_grounding_score,
        "claims": [asdict(c) for c in report.claims],
    }


def run_pilot_batch(
    data_path: Optional[str],
    n_samples: int,
    output_path: str,
    use_mock: bool,
    seed: int = 0,
    sleep_between_calls: float = 0.0,
    max_retries: int = 3,
) -> None:
    if data_path:
        dataset = load_dataset(data_path)
    else:
        print("[run_pilot_batch] no --data-path given, using built-in synthetic demo dataset", file=sys.stderr)
        dataset = _synthetic_demo_dataset()

    rng = random.Random(seed)
    batch = dataset if len(dataset) <= n_samples else rng.sample(dataset, n_samples)
    print(f"[run_pilot_batch] running {len(batch)} samples (dataset had {len(dataset)})", file=sys.stderr)

    base_llm_fn = mock_trace_generator_llm if use_mock else call_trace_generator_llm
    llm_fn = with_retry(base_llm_fn, max_retries=max_retries) if not use_mock else base_llm_fn

    generated: List[GeneratedTrace] = []
    errors: List[Dict[str, str]] = []
    for i, sample in enumerate(batch):
        try:
            g = generate_trace_for_sample(sample, llm_fn=llm_fn)
            generated.append(g)
        except LLMNotConfiguredError:
            raise  # surface immediately — nothing will succeed until this is wired up
        except Exception as e:  # noqa: BLE001
            print(f"[run_pilot_batch] sample {i} failed: {e}", file=sys.stderr)
            errors.append({"index": i, "error": str(e)})
        if sleep_between_calls:
            time.sleep(sleep_between_calls)

    traces_only = [g.structured_trace for g in generated]
    summary = audit_trace_batch(traces_only)

    output = {
        "n_requested": len(batch),
        "n_succeeded": len(generated),
        "n_errors": len(errors),
        "errors": errors,
        "audit_summary": {
            "n_traces": summary.n_traces,
            "mean_grounding_score": summary.mean_grounding_score,
            "fully_grounded_fraction": summary.fully_grounded_fraction,
            "claims_by_kind": summary.claims_by_kind,
            "ungrounded_by_kind": summary.ungrounded_by_kind,
            "worst_traces": summary.worst_traces,
            "n_contrastive_mentions": summary.n_contrastive_mentions,
            "contrastive_ungrounded_fraction": summary.contrastive_ungrounded_fraction,
        },
        "generated": [_serialize_generated(g) for g in generated],
    }

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n=== Pilot batch complete ===")
    print(f"Succeeded: {len(generated)}/{len(batch)} (errors: {len(errors)})")
    print(f"Mean grounding score: {summary.mean_grounding_score:.2f}")
    print(f"Fully grounded fraction: {summary.fully_grounded_fraction:.0%}")
    print(f"Claims by kind (direct): {summary.claims_by_kind}")
    print(f"Ungrounded by kind (direct): {summary.ungrounded_by_kind}")
    print(f"Contrastive mentions (informational): {summary.n_contrastive_mentions}")
    print(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1 pilot batch: generate + audit warm traces.")
    parser.add_argument("--data-path", type=str, default=None,
                         help="Path to a .csv/.json/.jsonl silkome dataset file. Omit to use the built-in synthetic demo.")
    parser.add_argument("--n-samples", type=int, default=200, help="Pilot batch size (default 200).")
    parser.add_argument("--output", type=str, default="pilot_batch_results.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls (rate limiting).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--use-mock", action="store_true",
                         help="Use the deterministic mock LLM instead of the real (unimplemented) client — for testing the driver itself.")
    args = parser.parse_args()

    run_pilot_batch(
        data_path=args.data_path,
        n_samples=args.n_samples,
        output_path=args.output,
        use_mock=args.use_mock,
        seed=args.seed,
        sleep_between_calls=args.sleep,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
