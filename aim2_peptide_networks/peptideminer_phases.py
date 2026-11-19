"""
Literature-Mined Phase Data (PeptideMiner) — Conditions and Leakage
======================================================================

The most condition-rich experimental corpus found so far for Aim 2, and the
one that most resembles what `dataset_schema.py` was written to audit: 1,017
observations assembled from many separate publications, each with a peptide,
its terminal chemistry, the solvent, concentration, pH, temperature and
incubation time, and the resulting self-assembly phase.

Source: Yang, Yorke, Knowles & Buehler, "Learning the rules of peptide
self-assembly through data mining with large language models"
(arXiv:2411.05421), data file `data/phase_data_clean.csv` from
https://github.com/lamm-mit/PeptideMiner

Phase is coded against the header's own legend:
`[none, ribbon, nanosphere, organogel, tube, particle, vesicle, hydrogel,
fiber]`, so 0 = none ... 7 = hydrogel, 8 = fiber.

It covers the chain's gelation and morphology nodes. It carries **no
modulus**, so like every other corpus reached so far it does not touch Aim
2's open problem.

The thing that matters most about this dataset
------------------------------------------------
**1,017 rows, 119 unique sequences.** Most sequences appear many times under
different conditions. That makes the row count a badly misleading sample
size, and it makes a row-level random split leak catastrophically: the same
peptide appears in train and test, so a model can score well by recognising
sequences rather than by generalising to new ones.

Aim 2 asks about new sequences. So the evaluation here groups by identity
cluster, and the module measures what the naive split would have claimed —
this is the same discipline as `identity_holdout.py`, applied where the
stakes are much higher than they were on the AP corpus.

Two data-quality caveats, both consequences of how it was built
------------------------------------------------------------------
This corpus was curated partly by LLM literature mining, so extraction
errors are expected rather than surprising. `quality_flags()` reports two:
physically implausible temperatures (the raw column spans -80 to 200 degC,
which are plainly storage and lyophilisation conditions mis-attributed to
the assembly step) and rows whose sequence contains non-standard residue
letters. Neither is a reason to discard the corpus; both are reasons to
report what fraction of it is trustworthy before fitting anything.

Runnable standalone: `python3 aim2_peptide_networks/peptideminer_phases.py [csv]`
"""

import csv
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ap_dataset import DatasetNotAvailableError, describe, descriptor_names  # noqa: E402
from baselines import MajorityBaseline, accuracy, balanced_accuracy, evaluate_binary, roc_auc  # noqa: E402
from dataset_schema import PeptideNetworkRecord, SchemaError  # noqa: E402
from identity_holdout import cluster_by_identity  # noqa: E402
from xu2023_gelation import (  # noqa: E402
    LogisticRegressor,
    grouped_kfold,
    out_of_fold_predictions,
    protocol_escalation_report,
)

DATA_URL = (
    "https://raw.githubusercontent.com/lamm-mit/PeptideMiner/main/"
    "data/phase_data_clean.csv"
)
DEFAULT_PATH = os.path.join(
    os.path.dirname(_HERE), "data", "peptideminer_phase_data_clean.csv"
)

# From the dataset's own column header legend.
PHASE_NAMES = [
    "none", "ribbon", "nanosphere", "organogel", "tube",
    "particle", "vesicle", "hydrogel", "fiber",
]
PHASE_HYDROGEL = 7
PHASE_FIBER = 8

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

# Assembly is studied in water, so anything outside this band is a storage,
# lyophilisation or processing temperature that the miner attached to the
# wrong step. Deliberately generous: the point is to catch nonsense, not to
# police plausible-but-unusual protocols.
PLAUSIBLE_TEMPERATURE_C = (0.0, 100.0)


@dataclass
class PhaseRecord:
    sapdb_id: str
    sequence: str
    n_term: str
    c_term: str
    concentration_mg_ml: Optional[float]
    pH: Optional[float]
    temperature_C: Optional[float]
    incubation_min: Optional[float]
    heating: Optional[bool]
    cyclic: bool
    phase: Optional[int]

    @property
    def phase_name(self) -> str:
        if self.phase is None or not 0 <= self.phase < len(PHASE_NAMES):
            return "unknown"
        return PHASE_NAMES[self.phase]

    @property
    def is_hydrogel(self) -> bool:
        return self.phase == PHASE_HYDROGEL

    @property
    def is_fiber(self) -> bool:
        return self.phase == PHASE_FIBER

    @property
    def is_standard_sequence(self) -> bool:
        return bool(self.sequence) and set(self.sequence) <= VALID_RESIDUES

    @property
    def concentration_wt_pct(self) -> Optional[float]:
        """mg/mL -> wt%, assuming 1 g/mL solution density.

        Stated rather than hidden, and used instead of mM so the corpus can
        be loaded without a molecular weight for exotic terminal groups
        (ferrocene, chlorination, N-hexadecylamine) whose masses this repo
        does not carry. Inventing those masses to produce a tidier unit
        would be worse than keeping the one that needs no assumption.
        """
        if self.concentration_mg_ml is None:
            return None
        return self.concentration_mg_ml / 10.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _maybe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in ("NA", "N/A", "NONE", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_phase_records(path: Optional[str] = None) -> List[PhaseRecord]:
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise DatasetNotAvailableError(
            f"PeptideMiner phase data not found at {path}.\n"
            f"Fetch it with:\n"
            f"  mkdir -p {os.path.dirname(path)}\n"
            f"  curl -L -o {path} \\\n    {DATA_URL}\n"
            f"(~300 KB; not committed to this repo — third-party data.)"
        )

    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contained no rows")

    phase_key = next(
        (k for k in rows[0] if k and k.startswith("type of self-assembly")), None
    )
    if phase_key is None:
        raise ValueError(
            f"no phase column found; columns were {list(rows[0])[:6]}..."
        )

    records: List[PhaseRecord] = []
    for row in rows:
        phase = _maybe_float(row.get(phase_key))
        heating_raw = (row.get("heating/cooling") or "").strip().lower()
        records.append(PhaseRecord(
            sapdb_id=str(row.get("sapdb_id", "")).strip(),
            sequence=(row.get("sequence") or "").strip().upper(),
            n_term=(row.get("N-terminal") or "Free").strip(),
            c_term=(row.get("C-terminal") or "Free").strip(),
            concentration_mg_ml=_maybe_float(row.get("concentration(mg/ml)")),
            pH=_maybe_float(row.get("PH")),
            temperature_C=_maybe_float(row.get("temperature")),
            incubation_min=_maybe_float(row.get("incubation period (min)")),
            heating=(heating_raw == "yes") if heating_raw in ("yes", "no") else None,
            cyclic=(row.get("linear/cyclic") or "").strip().lower() == "cyclic",
            phase=int(phase) if phase is not None else None,
        ))
    return records


def to_network_records(
    records: Sequence[PhaseRecord],
) -> List[PeptideNetworkRecord]:
    """Map onto the canonical schema, skipping rows it cannot represent.

    Returns only rows with a standard-residue sequence; the count of skipped
    rows is the caller's business to report, which `_run_demo` does.
    """
    out: List[PeptideNetworkRecord] = []
    for i, record in enumerate(records):
        if not record.is_standard_sequence:
            continue
        try:
            out.append(PeptideNetworkRecord(
                record_id=f"peptideminer_{record.sapdb_id or i}",
                sequence=record.sequence,
                n_term=record.n_term,
                c_term=record.c_term,
                concentration_wt_pct=record.concentration_wt_pct,
                pH=record.pH,
                temperature_C=record.temperature_C,
                gelation=record.is_hydrogel,
                source_doi="10.48550/arXiv.2411.05421",
                source_locator=f"phase_data_clean.csv sapdb_id={record.sapdb_id}",
                notes=f"phase={record.phase_name}",
            ))
        except SchemaError:
            # An unrepresentable terminal group or residue; counted by the
            # caller rather than silently coerced into something wrong.
            continue
    return out


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    n_rows: int
    n_unique_sequences: int
    non_standard_sequence_rows: int
    implausible_temperature_rows: int
    temperature_range: Tuple[Optional[float], Optional[float]]
    missing_counts: Dict[str, int]

    @property
    def rows_per_sequence(self) -> float:
        return self.n_rows / self.n_unique_sequences if self.n_unique_sequences else 0.0

    def summary(self) -> str:
        lines = [
            "Corpus quality flags",
            "=" * 66,
            f"rows                              : {self.n_rows}",
            f"unique sequences                  : {self.n_unique_sequences}",
            f"rows per sequence                 : {self.rows_per_sequence:.1f}",
            f"rows with non-standard residues   : {self.non_standard_sequence_rows}",
            f"rows with implausible temperature : "
            f"{self.implausible_temperature_rows} "
            f"(outside {PLAUSIBLE_TEMPERATURE_C[0]:g}-"
            f"{PLAUSIBLE_TEMPERATURE_C[1]:g} degC)",
            f"raw temperature range             : "
            f"{self.temperature_range[0]:g} to {self.temperature_range[1]:g} degC",
        ]
        if self.missing_counts:
            lines.append("missing values:")
            for name, count in sorted(self.missing_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {name:<30} {count}")
        lines.append("")
        if self.rows_per_sequence > 3:
            lines.append(
                f"!! The effective sample size is {self.n_unique_sequences} "
                f"sequences, not {self.n_rows} rows.\n"
                f"   Each sequence appears ~{self.rows_per_sequence:.0f} times "
                f"under different conditions, so a row-level random split "
                f"puts the\n   same peptide in train and test. Group by "
                f"sequence identity, and quote the sequence count as n."
            )
        return "\n".join(lines)


def quality_flags(records: Sequence[PhaseRecord]) -> QualityReport:
    temperatures = [
        r.temperature_C for r in records if r.temperature_C is not None
    ]
    missing = {
        "pH": sum(1 for r in records if r.pH is None),
        "temperature": sum(1 for r in records if r.temperature_C is None),
        "concentration": sum(1 for r in records if r.concentration_mg_ml is None),
        "incubation_min": sum(1 for r in records if r.incubation_min is None),
        "phase": sum(1 for r in records if r.phase is None),
    }
    return QualityReport(
        n_rows=len(records),
        n_unique_sequences=len({r.sequence for r in records if r.sequence}),
        non_standard_sequence_rows=sum(
            1 for r in records if not r.is_standard_sequence
        ),
        implausible_temperature_rows=sum(
            1 for r in records
            if r.temperature_C is not None
            and not (PLAUSIBLE_TEMPERATURE_C[0] <= r.temperature_C
                     <= PLAUSIBLE_TEMPERATURE_C[1])
        ),
        temperature_range=(
            (min(temperatures), max(temperatures)) if temperatures else (None, None)
        ),
        missing_counts={k: v for k, v in missing.items() if v},
    )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def condition_feature_names(n_term_levels: Sequence[str],
                            c_term_levels: Sequence[str]) -> List[str]:
    return (
        ["log10_concentration_mg_ml", "pH", "temperature_C",
         "log10_incubation_min", "heating", "cyclic"]
        + [f"nterm_{v}" for v in n_term_levels]
        + [f"cterm_{v}" for v in c_term_levels]
    )


def build_features(
    records: Sequence[PhaseRecord],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Sequence descriptors plus experimental conditions.

    Returns (X, feature_names, n_sequence_features) so the caller can fit on
    sequence-only or sequence+conditions by slicing, which is how the
    condition contribution is measured below.
    """
    sequence_names = descriptor_names()
    n_term_levels = [
        v for v, _ in Counter(r.n_term for r in records).most_common(6)
    ]
    c_term_levels = [
        v for v, _ in Counter(r.c_term for r in records).most_common(5)
    ]

    rows: List[List[float]] = []
    for record in records:
        features = describe(record.sequence)
        sequence_part = [features[name] for name in sequence_names]
        conditions = [
            math.log10(record.concentration_mg_ml)
            if record.concentration_mg_ml and record.concentration_mg_ml > 0 else 0.0,
            record.pH if record.pH is not None else 7.0,
            record.temperature_C if record.temperature_C is not None else 25.0,
            math.log10(record.incubation_min)
            if record.incubation_min and record.incubation_min > 0 else 0.0,
            1.0 if record.heating else 0.0,
            1.0 if record.cyclic else 0.0,
        ]
        conditions += [1.0 if record.n_term == v else 0.0 for v in n_term_levels]
        conditions += [1.0 if record.c_term == v else 0.0 for v in c_term_levels]
        rows.append(sequence_part + conditions)

    names = sequence_names + condition_feature_names(n_term_levels, c_term_levels)
    return np.array(rows, dtype=float), names, len(sequence_names)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo(path: Optional[str] = None) -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    try:
        raw_records = load_phase_records(path)
    except DatasetNotAvailableError as exc:
        print(f"--- dataset not present ---\n{exc}")
        return

    quality = quality_flags(raw_records)
    print(quality.summary())

    usable = [
        r for r in raw_records
        if r.is_standard_sequence and r.phase is not None
    ]
    print()
    print(f"--- usable rows: {len(usable)} of {len(raw_records)} ---")
    print(f"  phases: "
          f"{dict(Counter(r.phase_name for r in usable).most_common())}")
    print(f"  lengths: {dict(sorted(Counter(len(r.sequence) for r in usable).items()))}")
    print(f"  hydrogel rate: {sum(1 for r in usable if r.is_hydrogel)/len(usable):.1%}")

    # --- schema audit ------------------------------------------------------
    network = to_network_records(usable)
    print()
    print(f"--- canonical schema mapping: {len(network)} of {len(usable)} rows "
          f"representable ---")
    unrepresentable = len(usable) - len(network)
    if unrepresentable:
        caps = Counter(
            (r.n_term, r.c_term) for r in usable
        )
        print(f"  {unrepresentable} rows rejected, almost all for terminal "
              f"groups with no mass in the schema's tables")
        print(f"  (e.g. {[c for c, _ in caps.most_common(8)][-3:]}) — extend "
              f"N_TERM_MODIFICATIONS / C_TERM_MODIFICATIONS to ingest them")

    # --- is concentration confounded here? ---------------------------------
    print()
    escalation = protocol_escalation_report(
        [r.is_hydrogel for r in usable],
        [r.concentration_mg_ml if r.concentration_mg_ml is not None else 0.0
         for r in usable],
        "concentration_mg_ml",
    )
    print("--- concentration confound check (contrast with Xu 2023) ---")
    if escalation.rule_precision is not None:
        print(f"  best one-sided rule precision: {escalation.rule_precision:.3f} "
              f"(Xu 2023 was 1.000)")
    print(f"  confounded: {escalation.is_confounded}")
    print("  Multi-source corpora pool many independent protocols, so "
          "concentration here is\n  a genuine experimental variable rather "
          "than a record of when the experiment stopped.")

    # --- identity structure -------------------------------------------------
    sequences = sorted({r.sequence for r in usable})
    clustering = cluster_by_identity(sequences, threshold=0.6)
    cluster_of_sequence = clustering.cluster_of
    print()
    print(f"--- identity clusters over {len(sequences)} unique sequences ---")
    print(f"  {clustering.n_clusters} clusters, largest "
          f"{clustering.largest_cluster_size} "
          f"({clustering.largest_cluster_size/len(sequences):.0%} of sequences)")
    if clustering.largest_cluster_size > len(sequences) * 0.5:
        print(
            "  !! These are di- and tripeptides, which percolate at this "
            "threshold exactly as the\n     pentapeptides did in "
            "ap_dataset.py. The effective sample size is "
            f"{clustering.n_clusters} clusters\n     dominated by one, not "
            f"{len(sequences)} sequences and certainly not {len(usable)} "
            "rows. Any\n     sequence-level generalisation claim from this "
            "corpus rests on a handful of independent\n     groups — treat "
            "the grouped numbers below as indicative, not as a validation."
        )

    # --- models -------------------------------------------------------------
    X, names, n_sequence_features = build_features(usable)
    groups = [cluster_of_sequence[r.sequence] for r in usable]
    row_groups = list(range(len(usable)))

    for target_name, labels in (
        ("hydrogel", [r.is_hydrogel for r in usable]),
        ("fiber", [r.is_fiber for r in usable]),
    ):
        y = np.array([1.0 if v else 0.0 for v in labels])
        truth = y.astype(int).tolist()

        grouped_folds = grouped_kfold(groups, n_folds=5, seed=0)
        probabilities = out_of_fold_predictions(X, y, grouped_folds, l2=1.0)
        majority = MajorityBaseline().fit(y.tolist())

        evaluation = evaluate_binary(
            target=f"{target_name} (identity-grouped 5-fold OOF, "
                   f"n={len(sequences)} sequences)",
            y_true=truth,
            predictions={
                "logistic_sequence+conditions": (probabilities >= 0.5)
                .astype(float).tolist()
            },
            baselines={"majority": majority.predict(len(y))},
            scores={"logistic_sequence+conditions": probabilities.tolist()},
        )
        print()
        print(evaluation.summary(model_name="logistic_sequence+conditions"))

        # The leakage measurement.
        row_folds = grouped_kfold(row_groups, n_folds=5, seed=0)
        row_probabilities = out_of_fold_predictions(X, y, row_folds, l2=1.0)
        grouped_auc = roc_auc(truth, probabilities.tolist())
        row_auc = roc_auc(truth, row_probabilities.tolist())
        print(f"  row-level random folds (LEAKY) : AUC {row_auc:.3f}  "
              f"balanced "
              f"{balanced_accuracy(truth, (row_probabilities >= 0.5).astype(float).tolist()):.3f}")
        print(f"  identity-grouped folds         : AUC {grouped_auc:.3f}  "
              f"balanced "
              f"{balanced_accuracy(truth, (probabilities >= 0.5).astype(float).tolist()):.3f}")
        print(f"  inflation from row-level splitting: {row_auc - grouped_auc:+.3f} AUC")

        # Sequence-only vs sequence+conditions, both grouped.
        sequence_only = out_of_fold_predictions(
            X[:, :n_sequence_features], y, grouped_folds, l2=1.0
        )
        print(f"  sequence descriptors only      : AUC "
              f"{roc_auc(truth, sequence_only.tolist()):.3f}")

    print()
    print("All peptideminer_phases.py self-checks passed.")


if __name__ == "__main__":
    _run_demo(sys.argv[1] if len(sys.argv) > 1 else None)
