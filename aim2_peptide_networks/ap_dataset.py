"""
Aggregation-Propensity Dataset: Loader, Descriptors, and Density Audit
=========================================================================

What this dataset is, and which link of the chain it serves
--------------------------------------------------------------
Aim 2's decomposition (Figure 5) is

    sequence -> descriptors -> coarse-grained assembly (aggregation
    propensity) -> fiber morphology -> network modulus

This module handles the dataset for the **second** link only: aggregation
propensity (AP) from coarse-grained MD. It is NOT the modulus data, and
nothing here speaks to the open problem at the end of the chain.

Source: Liu et al., "Efficient prediction of peptide self-assembly through
sequential and graphical encoding", Briefings in Bioinformatics 24(6),
bbad409 (2023); preprint arXiv:2307.09169. Data file
`Sequential_Peptides/collection_of_peptide_data.csv` from the authors'
repository at https://github.com/Zihan-Liu-00/DL_for_Peptide

62,159 peptides of length 5-10 over the standard 20 residues, each with an
AP label. AP is the Martini/GROMACS coarse-grained solvent-accessible
surface area ratio (SASA_initial / SASA_final) used since Frederix et al.,
where AP > 2 is the conventional self-assembly cutoff. In this corpus AP
ranges 0.993-2.619 with 8.8% of peptides above that cutoff.

The label is therefore **simulation output, not measurement**. That makes
this dataset suitable for checking whether the descriptor -> AP link is
learnable at all, and unsuitable as evidence about experimental behaviour.
A model that predicts AP well has learned to emulate a coarse-grained
simulator, which is a genuinely useful thing to know and is not the same as
predicting what a peptide does in a vial. Keep that distinction in any
claim built on this file.

Why the density audit is here
--------------------------------
`identity_holdout.py` builds identity-based partitions, and whether such a
partition is even possible depends on how densely the library covers
sequence space. This module provides an exact, fast audit of that density,
because for this corpus the answer differs sharply by peptide length and
determines how the evaluation must be set up.

`substitution_clusters()` is exact for equal-length sequences and is a
*lower bound* on true identity linkage: Hamming <= k implies Levenshtein
<= k, so every edge it finds is real, and the edges it misses (indels,
cross-length pairs) can only merge clusters further. A library that already
percolates under this bound percolates under the true metric too.

Runnable standalone: `python3 aim2_peptide_networks/ap_dataset.py [csv_path]`
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Import the validated descriptor tools from the shared tooling layer rather
# than reimplementing them. Per CLAUDE.md the agent/tool layer is shared
# across Aims 2 and 3 and should be extended in place.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bio_tools import (  # noqa: E402
    KYTE_DOOLITTLE,
    CHOU_FASMAN,
    beta_sheet_propensity,
    charge_profile,
    hydropathy_profile,
)

DATA_URL = (
    "https://raw.githubusercontent.com/Zihan-Liu-00/DL_for_Peptide/main/"
    "Sequential_Peptides/collection_of_peptide_data.csv"
)

DEFAULT_PATH = os.path.join(_REPO_ROOT, "data", "collection_of_peptide_data.csv")

AROMATIC = set("FWY")
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

# The conventional self-assembly cutoff on this AP scale (Frederix et al.).
AP_SELF_ASSEMBLY_CUTOFF = 2.0


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when the AP dataset file is absent.

    Distinct from a generic FileNotFoundError, following the repo's
    convention of a specific error for missing-resource failures, so a
    caller can tell "you need to fetch the data" apart from "your path is
    wrong" and fail fast with an actionable message.
    """


@dataclass
class APRecord:
    sequence: str
    ap: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ap_dataset(path: Optional[str] = None) -> List[APRecord]:
    """Load the AP corpus, rejecting malformed rows rather than coercing them."""
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise DatasetNotAvailableError(
            f"AP dataset not found at {path}.\n"
            f"Fetch it with:\n"
            f"  mkdir -p {os.path.dirname(path)}\n"
            f"  curl -L -o {path} \\\n    {DATA_URL}\n"
            f"The file is ~940 KB and is not committed to this repo "
            f"(third-party data; see .gitignore)."
        )

    records: List[APRecord] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Feature" not in reader.fieldnames:
            raise ValueError(
                f"unexpected columns {reader.fieldnames!r}; expected a "
                f"'Feature' (sequence) and 'Label' (AP) column"
            )
        for line_no, row in enumerate(reader, start=2):
            sequence = (row.get("Feature") or "").strip().upper()
            raw_ap = (row.get("Label") or "").strip()
            if not sequence or not raw_ap:
                continue
            bad = set(sequence) - set(RESIDUES)
            if bad:
                raise ValueError(
                    f"line {line_no}: non-standard residues {sorted(bad)} "
                    f"in {sequence!r}"
                )
            records.append(APRecord(sequence=sequence, ap=float(raw_ap)))
    if not records:
        raise ValueError(f"{path} contained no usable rows")
    return records


# ---------------------------------------------------------------------------
# Descriptors (link 1 of the chain)
# ---------------------------------------------------------------------------

def descriptor_names() -> List[str]:
    """Feature order, fixed so a fitted model stays interpretable."""
    return (
        ["length", "mean_hydropathy", "max_window_hydropathy",
         "net_charge_per_residue", "abs_net_charge_per_residue",
         "mean_beta", "mean_alpha", "mean_turn", "aromatic_fraction"]
        + [f"frac_{residue}" for residue in RESIDUES]
    )


def describe(sequence: str) -> Dict[str, float]:
    """Compute the descriptor set for one peptide.

    Uses `bio_tools` for hydropathy, charge and Chou-Fasman propensities so
    the agent and this baseline see the same numbers from the same validated
    implementations. The composition fractions are added here because for
    peptides this short, composition is most of what there is — a 5-mer has
    little room for the windowed profile structure those tools were written
    for, which is itself worth knowing when reading the model's weights.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("empty sequence has no descriptors")

    n = len(sequence)
    # Short peptides: bio_tools shrinks the window to the sequence length,
    # so these calls are well-defined but the "profile" is nearly a mean.
    hydro = hydropathy_profile(sequence, window=min(3, n))
    charge = charge_profile(sequence)
    beta = beta_sheet_propensity(sequence, window=min(3, n))

    alpha_vals, beta_vals, turn_vals = [], [], []
    for residue in sequence:
        pa, pb, pt = CHOU_FASMAN.get(residue, (1.0, 1.0, 1.0))
        alpha_vals.append(pa)
        beta_vals.append(pb)
        turn_vals.append(pt)

    counts = Counter(sequence)
    features: Dict[str, float] = {
        "length": float(n),
        "mean_hydropathy": float(hydro["mean_hydropathy"]),
        "max_window_hydropathy": float(
            hydro["max_window_value"]
            if hydro["max_window_value"] is not None
            else hydro["mean_hydropathy"]
        ),
        "net_charge_per_residue": float(charge["mean_net_charge_per_residue"]),
        "abs_net_charge_per_residue": abs(
            float(charge["mean_net_charge_per_residue"])
        ),
        "mean_beta": sum(beta_vals) / n,
        "mean_alpha": sum(alpha_vals) / n,
        "mean_turn": sum(turn_vals) / n,
        "aromatic_fraction": sum(
            1 for residue in sequence if residue in AROMATIC
        ) / n,
    }
    # `beta` is computed above via the validated tool; keep its summary
    # reachable for callers that want the tool's own call rather than the
    # per-residue mean recomputed here.
    features["_beta_tool_mean"] = float(beta.get("mean_beta_propensity", 0.0))
    for residue in RESIDUES:
        features[f"frac_{residue}"] = counts.get(residue, 0) / n
    return features


def feature_matrix(
    sequences: Sequence[str],
) -> Tuple[List[List[float]], List[str]]:
    """Dense feature matrix in `descriptor_names()` order."""
    names = descriptor_names()
    matrix = []
    for sequence in sequences:
        features = describe(sequence)
        matrix.append([features[name] for name in names])
    return matrix, names


# ---------------------------------------------------------------------------
# Density audit
# ---------------------------------------------------------------------------

@dataclass
class LengthDensityRow:
    length: int
    n: int
    threshold: float
    max_substitutions: int
    n_clusters: int
    largest_cluster: int

    @property
    def fraction_in_largest(self) -> float:
        return self.largest_cluster / self.n if self.n else 0.0

    @property
    def is_percolated(self) -> bool:
        """Whether one cluster swallows most of the length class.

        Half is a deliberately generous line: once a single cluster holds
        the majority, no identity-based split can put a representative test
        set on one side of it, so the exact figure stops mattering.
        """
        return self.fraction_in_largest > 0.5


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def substitution_clusters(
    sequences: Sequence[str],
    max_substitutions: int,
) -> List[int]:
    """Cluster equal-length sequences within `max_substitutions` substitutions.

    Exact and O(n * C(L, k)) via masked buckets: two sequences differ in at
    most k positions iff they agree on some set of L-k positions, so masking
    every k-subset and grouping on the remainder finds exactly the pairs
    within k substitutions, with no pairwise comparison at all.

    Returns a cluster id per input sequence. All inputs must share a length.
    """
    if not sequences:
        return []
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("substitution_clusters requires equal-length sequences")
    if max_substitutions < 1:
        # Only exact duplicates link.
        seen: Dict[str, int] = {}
        return [seen.setdefault(s, len(seen)) for s in sequences]

    uf = _UnionFind(len(sequences))
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        for mask in combinations(range(length), max_substitutions):
            masked = tuple(
                sequence[p] for p in range(length) if p not in mask
            )
            buckets[(mask, masked)].append(index)
    for members in buckets.values():
        first = members[0]
        for other in members[1:]:
            uf.union(first, other)

    remap: Dict[int, int] = {}
    return [remap.setdefault(uf.find(i), len(remap)) for i in range(len(sequences))]


def density_audit(
    records: Sequence[APRecord],
    thresholds: Iterable[float] = (0.6, 0.8),
) -> List[LengthDensityRow]:
    """Per-length identity-cluster structure at each threshold.

    A lower bound on linkage (see module docstring), which is the useful
    direction: percolation found here is real.
    """
    by_length: Dict[int, List[str]] = defaultdict(list)
    for record in records:
        by_length[len(record.sequence)].append(record.sequence)

    rows: List[LengthDensityRow] = []
    for threshold in thresholds:
        for length in sorted(by_length):
            sequences = by_length[length]
            # identity >= t  <=>  edits <= floor((1 - t) * L) for equal lengths
            k = int((1.0 - threshold) * length)
            labels = substitution_clusters(sequences, k)
            sizes = Counter(labels)
            rows.append(LengthDensityRow(
                length=length,
                n=len(sequences),
                threshold=threshold,
                max_substitutions=k,
                n_clusters=len(sizes),
                largest_cluster=max(sizes.values()),
            ))
    return rows


def format_density_audit(rows: Sequence[LengthDensityRow]) -> str:
    lines = [
        "Identity-cluster density by peptide length",
        "(lower bound: equal-length substitutions only; indels can only merge further)",
        "=" * 78,
        f"{'len':>4} {'n':>7} {'thr':>6} {'max subs':>9} {'clusters':>9} "
        f"{'largest':>8} {'% in largest':>13}",
        "-" * 78,
    ]
    for row in rows:
        note = "  <- percolated" if row.is_percolated else ""
        subs = "exact only" if row.max_substitutions < 1 else str(row.max_substitutions)
        lines.append(
            f"{row.length:>4} {row.n:>7} {row.threshold:>6.2f} {subs:>9} "
            f"{row.n_clusters:>9} {row.largest_cluster:>8} "
            f"{row.fraction_in_largest:>12.1%}{note}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo(path: Optional[str] = None) -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    print("--- descriptors on known gelators ---")
    for sequence in ["KLVFF", "RADARADA", "GGGGG", "FFFFF"]:
        features = describe(sequence)
        print(f"  {sequence:<10} hydropathy {features['mean_hydropathy']:>6.2f}  "
              f"beta {features['mean_beta']:.2f}  "
              f"aromatic {features['aromatic_fraction']:.2f}  "
              f"charge/res {features['net_charge_per_residue']:>6.3f}")
    # KLVFF is the amyloid-beta core and should read as hydrophobic and
    # beta-prone; polyglycine should read as neither. A sanity check on the
    # descriptors before any model is fitted on them.
    klvff = describe("KLVFF")
    glycine = describe("GGGGG")
    assert klvff["mean_hydropathy"] > glycine["mean_hydropathy"]
    assert klvff["mean_beta"] > glycine["mean_beta"]
    print("  check: KLVFF reads more hydrophobic and more beta-prone than "
          "G5 -> OK")

    print()
    print("--- masked-bucket clustering is exact ---")
    # Hand-checkable: three sequences within one substitution, one far away.
    labels = substitution_clusters(["AAAAA", "AAAAC", "AAACC", "WWWWW"], 1)
    assert labels[0] == labels[1], "1-substitution pair must link"
    assert labels[1] == labels[2], "1-substitution pair must link"
    assert labels[3] != labels[0], "distant sequence must not link"
    print(f"  AAAAA/AAAAC/AAACC link, WWWWW separate -> {labels} OK")

    print()
    try:
        records = load_ap_dataset(path)
    except DatasetNotAvailableError as exc:
        print(f"--- AP corpus not present, skipping corpus audit ---\n{exc}")
        print()
        print("All ap_dataset.py self-checks that do not need the corpus passed.")
        return

    lengths = Counter(len(r.sequence) for r in records)
    aps = [r.ap for r in records]
    print(f"--- AP corpus: {len(records)} peptides ---")
    print(f"  lengths            : {dict(sorted(lengths.items()))}")
    print(f"  unique sequences   : {len(set(r.sequence for r in records))}")
    print(f"  AP range           : {min(aps):.3f} - {max(aps):.3f}")
    print(f"  AP mean / sd       : {sum(aps)/len(aps):.3f} / "
          f"{(sum((a - sum(aps)/len(aps)) ** 2 for a in aps)/len(aps)) ** 0.5:.3f}")
    print(f"  above AP>{AP_SELF_ASSEMBLY_CUTOFF} cutoff : "
          f"{sum(1 for a in aps if a > AP_SELF_ASSEMBLY_CUTOFF)/len(aps):.1%}")

    print()
    rows = density_audit(records)
    print(format_density_audit(rows))

    print()
    print("All ap_dataset.py self-checks passed.")


if __name__ == "__main__":
    _run_demo(sys.argv[1] if len(sys.argv) > 1 else None)
