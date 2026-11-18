"""
Initial Descriptor Model for Aggregation Propensity (link 2 of the chain)
===========================================================================

Fits a ridge regression from `bio_tools` sequence descriptors to the
coarse-grained aggregation propensity (AP) label, and evaluates it the way
Aim 2 requires: under BOTH a random split and an identity-based split, each
reported beside its mandatory baselines.

This is deliberately a weak model. The point is not to win a benchmark —
the source paper's Transformer already does better — but to establish three
things before any agent is built on top of this link:

  1. **Is the link learnable from descriptors at all?** If simple physical
     descriptors already carry most of the signal, a composed chain that
     calls a coarse-grained simulator for this step has to beat them to be
     worth its cost.
  2. **Does the split choice change the answer?** Section C.6's lesson is
     that a number without its comparison is not interpretable; the
     random-vs-identity gap is the comparison that reveals leakage.
  3. **What does a trivial baseline get?** In silk, nobody ran that until
     late, and it beat the pipeline.

Ridge rather than something stronger, and written out in numpy rather than
pulled from scikit-learn, because a closed-form linear fit has no training
loop to get subtly wrong, its coefficients are directly readable as "which
descriptor drives AP", and it keeps this repo's dependency surface where it
already is (numpy is present; scikit-learn is not).

What this does and does not show
-----------------------------------
The AP label is simulation output, not measurement (see `ap_dataset.py`).
A model that predicts AP is emulating a coarse-grained simulator. That is
useful — it is the chain's intermediate, and the proposal's feasibility
argument rests on intermediates being checkable — but it is not evidence
about network modulus, which is the open problem Aim 2 actually targets.

Runnable standalone: `python3 aim2_peptide_networks/ap_model.py [csv_path]`
"""

import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ap_dataset import (  # noqa: E402
    APRecord,
    DatasetNotAvailableError,
    density_audit,
    descriptor_names,
    feature_matrix,
    format_density_audit,
    load_ap_dataset,
    substitution_clusters,
)
from baselines import (  # noqa: E402
    SingleCovariateBaseline,
    TrivialBaseline,
    evaluate,
)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

class RidgeRegressor:
    """Closed-form ridge with standardised features and a centred target.

    Standardisation matters here beyond conditioning: it makes the fitted
    coefficients comparable to each other, so the weight vector can be read
    as a ranking of which descriptor drives AP. The intercept is handled by
    centring rather than by an unpenalised column, which keeps the penalty
    from shrinking the mean.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        # A constant feature carries no information; setting its scale to 1
        # leaves it centred at exactly zero rather than dividing by zero.
        scale[scale == 0] = 1.0
        self.scale_ = scale
        Xs = (X - self.mean_) / self.scale_
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        n_features = Xs.shape[1]
        gram = Xs.T @ Xs + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(gram, Xs.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit() before predict()")
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Xs @ self.coef_ + self.intercept_


def fit_with_alpha_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[RidgeRegressor, float]:
    """Pick alpha on a validation slice carved out of the TRAINING set only.

    Never touches the test set. Tuning on test is the most common way a
    reported number stops meaning what it claims, and it would be
    especially corrosive here, where the whole exercise is about making
    comparisons trustworthy.
    """
    n = len(y_train)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    cut = int((1 - validation_fraction) * n)
    inner_train, inner_val = indices[:cut], indices[cut:]

    best_alpha, best_rmse, best_model = alphas[0], float("inf"), None
    for alpha in alphas:
        model = RidgeRegressor(alpha=alpha).fit(
            X_train[inner_train], y_train[inner_train]
        )
        predictions = model.predict(X_train[inner_val])
        error = float(np.sqrt(np.mean((predictions - y_train[inner_val]) ** 2)))
        if error < best_rmse:
            best_alpha, best_rmse, best_model = alpha, error, model

    # Refit on the full training set at the chosen alpha.
    return RidgeRegressor(alpha=best_alpha).fit(X_train, y_train), best_alpha


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def random_split(
    n: int, test_fraction: float = 0.2, seed: int = 0
) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    cut = int((1 - test_fraction) * n)
    return indices[:cut], indices[cut:]


def identity_split(
    records: Sequence[APRecord],
    threshold: float = 0.6,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Whole-cluster split using the exact per-length substitution clusters.

    Clusters are computed within each length class (see `ap_dataset`), which
    is a lower bound on identity linkage: real indel and cross-length edges
    would only merge clusters further, making the split coarser, never
    leakier in the direction that would flatter a model. So a model's score
    under this split is, if anything, slightly optimistic — and that is the
    safe direction for a guard to err in.
    """
    by_length: Dict[int, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_length[len(record.sequence)].append(index)

    cluster_of: Dict[int, Tuple[int, int]] = {}
    for length, members in by_length.items():
        k = int((1.0 - threshold) * length)
        labels = substitution_clusters([records[i].sequence for i in members], k)
        for member, label in zip(members, labels):
            cluster_of[member] = (length, label)

    clusters: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, key in cluster_of.items():
        clusters[key].append(index)

    keys = sorted(clusters)
    random.Random(seed).shuffle(keys)

    target = test_fraction * len(records)
    test: List[int] = []
    for key in keys:
        if len(test) >= target:
            break
        test.extend(clusters[key])
    test_set = set(test)
    train = [i for i in range(len(records)) if i not in test_set]

    stats = {
        "n_clusters": len(clusters),
        "largest_cluster": max(len(v) for v in clusters.values()),
    }
    return train, sorted(test), stats


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def describe_split_composition(
    records: Sequence[APRecord],
    random_train: Sequence[int],
    random_test: Sequence[int],
    identity_train: Sequence[int],
    identity_test: Sequence[int],
) -> str:
    """Compare what each split actually holds out, by peptide length.

    Worth printing every time, because an identity split does not merely
    rearrange the same test set: whole clusters move together, so a length
    class that forms one big cluster lands entirely on one side. When that
    happens the two splits are testing on different populations and their
    scores are not comparable, however similar the numbers look. Catching
    that is the same discipline as C.6 — the number is only as good as the
    comparison it sits in.
    """
    def by_length(indices: Sequence[int]) -> Dict[int, int]:
        return dict(sorted(Counter(
            len(records[i].sequence) for i in indices
        ).items()))

    random_counts = by_length(random_test)
    identity_counts = by_length(identity_test)
    all_lengths = sorted(set(random_counts) | set(identity_counts))

    lines = [
        "  test-set composition by length:",
        f"    {'len':>5} {'random':>9} {'identity':>9}",
    ]
    for length in all_lengths:
        lines.append(
            f"    {length:>5} {random_counts.get(length, 0):>9} "
            f"{identity_counts.get(length, 0):>9}"
        )

    missing = [
        length for length in all_lengths if identity_counts.get(length, 0) == 0
    ]
    if missing:
        lines.append("")
        lines.append(
            f"  !! length(s) {missing} are entirely absent from the identity "
            f"test set."
        )
        lines.append(
            "     Those peptides form a single identity cluster, so the split "
            "had to assign them whole. The two splits therefore evaluate on "
            "different populations and their scores are NOT directly "
            "comparable — read each against its own baselines, not against "
            "each other."
        )
    return "\n".join(lines)


@dataclass
class SplitResult:
    name: str
    n_train: int
    n_test: int
    alpha: float
    report: object  # EvaluationReport


def run_split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    hydropathy: np.ndarray,
    train: Sequence[int],
    test: Sequence[int],
    seed: int = 0,
) -> SplitResult:
    train = list(train)
    test = list(test)
    model, alpha = fit_with_alpha_search(X[train], y[train], seed=seed)
    predictions = model.predict(X[test])

    trivial = TrivialBaseline("mean").fit(y[train].tolist())
    covariate = SingleCovariateBaseline("mean_hydropathy").fit(
        hydropathy[train].tolist(), y[train].tolist()
    )

    report = evaluate(
        target=f"aggregation propensity ({name} split)",
        y_true=y[test].tolist(),
        predictions={"ridge_descriptors": predictions.tolist()},
        baselines={
            "trivial": trivial.predict(len(test)),
            "single_covariate_hydropathy": covariate.predict(
                hydropathy[test].tolist()
            ),
        },
        # The conventional AP > 2 cutoff is a classification boundary, not a
        # tolerance, so a tolerance here is a modelling choice. 0.1 is ~half
        # the corpus standard deviation (0.211), chosen to be the same shape
        # of criterion the silk pipeline used so the audit is comparable.
        tolerance=0.1,
        primary_model="ridge_descriptors",
    )
    return SplitResult(name, len(train), len(test), alpha, report)


def top_coefficients(
    model: RidgeRegressor, names: Sequence[str], k: int = 10
) -> List[Tuple[str, float]]:
    if model.coef_ is None:
        raise RuntimeError("fit() before reading coefficients")
    pairs = list(zip(names, model.coef_.tolist()))
    pairs.sort(key=lambda p: -abs(p[1]))
    return pairs[:k]


def _run_demo(path: Optional[str] = None) -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    # --- self-check the regressor on data with a known answer -------------
    rng = np.random.default_rng(0)
    X_synth = rng.normal(size=(400, 3))
    true_w = np.array([2.0, -1.0, 0.0])
    y_synth = X_synth @ true_w + 5.0 + rng.normal(scale=0.01, size=400)
    model = RidgeRegressor(alpha=1e-6).fit(X_synth, y_synth)
    recovered = model.coef_ / model.scale_
    assert np.allclose(recovered, true_w, atol=0.05), recovered
    # The intercept is the training mean by construction, which carries the
    # sampling noise of X@w (sd ~0.11 here) — so check what it is for, that
    # predictions land on the truth, not that it equals 5.0 exactly.
    assert np.allclose(model.predict(X_synth), y_synth, atol=0.05)
    print("--- ridge recovers known coefficients on synthetic data -> OK ---")
    print(f"  true {true_w.tolist()}  recovered "
          f"{[round(v, 3) for v in recovered.tolist()]}")

    print()
    try:
        records = load_ap_dataset(path)
    except DatasetNotAvailableError as exc:
        print(f"--- AP corpus not present ---\n{exc}")
        return

    sequences = [r.sequence for r in records]
    y = np.array([r.ap for r in records], dtype=float)
    matrix, names = feature_matrix(sequences)
    X = np.array(matrix, dtype=float)
    hydropathy = X[:, names.index("mean_hydropathy")]
    print(f"--- featurised {X.shape[0]} peptides into {X.shape[1]} descriptors ---")

    results: List[SplitResult] = []

    train, test = random_split(len(records), test_fraction=0.2, seed=0)
    results.append(run_split("random", X, y, hydropathy, train, test))

    train_id, test_id, stats = identity_split(records, threshold=0.6,
                                              test_fraction=0.2, seed=0)
    print(f"--- identity split: {stats['n_clusters']} clusters, "
          f"largest {stats['largest_cluster']} peptides ---")
    print(describe_split_composition(records, train, test, train_id, test_id))
    results.append(run_split("identity-0.60", X, y, hydropathy,
                             train_id, test_id))

    for result in results:
        print()
        print(f"[{result.name}]  train {result.n_train}  test {result.n_test}  "
              f"alpha {result.alpha:g}")
        print(result.report.summary(model_name="ridge_descriptors"))

    # --- what the model actually learned ---------------------------------
    full_model, _ = fit_with_alpha_search(X[train], y[train], seed=0)
    print()
    print("--- top standardised ridge coefficients (random-split fit) ---")
    for name, weight in top_coefficients(full_model, names, k=12):
        bar = "+" * int(abs(weight) * 40 / max(
            abs(w) for _, w in top_coefficients(full_model, names, k=1)
        ))
        print(f"  {name:<26} {weight:>8.4f}  {bar}")

    # --- per-length performance on the random split ----------------------
    print()
    print("--- random-split test performance by peptide length ---")
    predictions = full_model.predict(X[test])
    by_length: Dict[int, List[int]] = defaultdict(list)
    for position, index in enumerate(test):
        by_length[len(sequences[index])].append(position)
    print(f"  {'len':>4} {'n':>6} {'rho':>8} {'MAE':>9} {'trivial MAE':>12}")
    from baselines import mae as _mae, spearman as _spearman  # noqa: E402
    for length in sorted(by_length):
        positions = by_length[length]
        truth = y[[test[p] for p in positions]].tolist()
        predicted = [predictions[p] for p in positions]
        constant = [float(np.mean(y[train]))] * len(truth)
        rho = _spearman(truth, predicted)
        print(f"  {length:>4} {len(positions):>6} "
              f"{(rho if rho is not None else float('nan')):>8.3f} "
              f"{_mae(truth, predicted):>9.4f} {_mae(truth, constant):>12.4f}")

    print()
    print("All ap_model.py self-checks passed.")


if __name__ == "__main__":
    _run_demo(sys.argv[1] if len(sys.argv) > 1 else None)
