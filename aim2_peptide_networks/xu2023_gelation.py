"""
Experimental Gelation + Morphology Data (Xu et al. 2023)
===========================================================

The first **experimental** dataset wired into Aim 2. It covers two nodes of
the Figure 5 chain:

    sequence -> descriptors -> CG assembly -> [gelation]        <- here
                                           -> [fiber morphology] <- here
                                           -> network modulus    <- still missing

Source: Xu et al., "Accelerating the prediction and discovery of peptide
hydrogels with human-in-the-loop", Nature Communications 14, 3880 (2023),
doi:10.1038/s41467-023-39648-2, Supplementary Data 7 ("Statistical results
of gelation experiments of 165 synthetic tetrapeptides").

165 synthesised tetrapeptides, each with an observed state (gel / sol /
suspension / precipitate), a TEM morphology, the concentration at which the
observation was recorded, and the authors' own computed APH / APHC ranks.

This is the proposal's "upper branch, already solved" positive control,
plus a morphology observable. It carries **no modulus**, so it does not
touch Aim 2's open problem. What it does give is the first real test of
whether sequence descriptors predict an experimentally measured assembly
outcome.

THE PROTOCOL CONFOUND — read before using `concentration_mM` for anything
---------------------------------------------------------------------------
Concentration in this table is **determined by the label**, not chosen
independently. The experiment escalates: test at 30 mM, record a gel if one
forms, otherwise retry higher, and record the final state at 120 mM. The
resulting cross-tabulation is total:

    state         30mM  60mM  90mM  120mM
    gel             46    30     6     18
    sol              0     0     0     30
    suspension       0     0     0     28
    precipitate      0     0     0      7

So "concentration < 120" implies gel with **perfect precision** (82 of 82,
zero false positives) and reaches 89.1% accuracy against a 60.6%
majority-class baseline — using a column that only has its value because
the outcome was already known. Any model given this feature will look
excellent and have learned nothing transferable.

`protocol_escalation_report()` below detects this pattern generically
rather than hard-coding it, because the same escalation design is standard
across the gelation literature and any corpus compiled per
`dataset_schema.py` is likely to contain it. It is the single most
dangerous column in this dataset and the reason `dataset_schema.py`'s
treatment of concentration as "a covariate the chain should reason about"
needs qualifying: that is right for an independently-set concentration and
badly wrong for an escalated one.

A SECOND, SOFTER CAVEAT — the 165 are not a random sample
-----------------------------------------------------------
They were chosen by the authors' own APHC ranking, enriched at both ends.
The APHC baseline's AUC on this set is therefore an optimistic estimate of
what it would achieve on randomly drawn tetrapeptides, and so is any model
evaluated here. Report it as what it is: performance on a deliberately
bimodal selection.

Runnable standalone: `python3 aim2_peptide_networks/xu2023_gelation.py [xlsx]`
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

from ap_dataset import DatasetNotAvailableError, feature_matrix, substitution_clusters  # noqa: E402
from baselines import MajorityBaseline, evaluate_binary, roc_auc  # noqa: E402
from dataset_schema import PeptideNetworkRecord, heterogeneity_report  # noqa: E402

DATA_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-023-39648-2/MediaObjects/"
    "41467_2023_39648_MOESM10_ESM.xlsx"
)

DEFAULT_PATH = os.path.join(
    os.path.dirname(_HERE), "data", "xu2023_supplementary_data_7.xlsx"
)

# The source table contains a typo ("hybird") that would otherwise split one
# morphology class in two. Normalising it here, visibly, rather than leaving
# a silent 1-row category.
_MORPHOLOGY_FIXES = {"hybird(sphere+fibre)": "hybrid(sphere+fibre)"}

# Morphologies that are fibrous, i.e. the Figure 5 "fiber morphology" node.
# Hybrids containing a fibre component count as fibrous: the chain's question
# is whether fibres form at all, not whether they form exclusively.
_FIBROUS = {"fibre", "needle", "rod"}


@dataclass
class GelationRecord:
    peptide: str
    state: str
    is_gel: bool
    morphology: str
    concentration_mM: float
    aph_rank: Optional[int]
    aphc_rank: Optional[int]

    @property
    def is_fibrous(self) -> bool:
        morphology = self.morphology.lower()
        if morphology in _FIBROUS:
            return True
        return morphology.startswith("hybrid") and "fibre" in morphology


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gelation_records(path: Optional[str] = None) -> List[GelationRecord]:
    """Load Supplementary Data 7, normalising states and morphologies."""
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise DatasetNotAvailableError(
            f"Xu 2023 Supplementary Data 7 not found at {path}.\n"
            f"Fetch it with:\n"
            f"  mkdir -p {os.path.dirname(path)}\n"
            f"  curl -L -A 'Mozilla/5.0' -o {path} \\\n    '{DATA_URL}'\n"
            f"(~18 KB; not committed to this repo — third-party data.)"
        )
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise DatasetNotAvailableError(
            "reading this dataset needs openpyxl: pip install openpyxl"
        ) from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = [row for row in sheet.iter_rows(values_only=True)]
    workbook.close()

    # Row 0 is a title banner, row 1 the header, data from row 2.
    header = [str(c).strip() if c is not None else "" for c in rows[1]]
    expected = ["Peptide", "State"]
    if header[:2] != expected:
        raise ValueError(
            f"unexpected header {header[:4]!r}; expected to start with "
            f"{expected!r} — is this Supplementary Data 7?"
        )

    records: List[GelationRecord] = []
    for row in rows[2:]:
        if not row or not row[0]:
            continue
        peptide = str(row[0]).strip().upper()
        state = str(row[1]).strip().lower()
        morphology = str(row[5]).strip().lower() if row[5] else "unknown"
        morphology = _MORPHOLOGY_FIXES.get(morphology, morphology)
        records.append(GelationRecord(
            peptide=peptide,
            state=state,
            is_gel=(state == "gel"),
            morphology=morphology,
            concentration_mM=float(row[6]),
            aph_rank=int(row[2]) if row[2] is not None else None,
            aphc_rank=int(row[3]) if row[3] is not None else None,
        ))
    if not records:
        raise ValueError(f"{path} contained no usable rows")
    return records


def to_network_records(
    records: Sequence[GelationRecord],
) -> List[PeptideNetworkRecord]:
    """Map onto the canonical schema so the standard audit applies.

    Produces exactly the shape the audit is designed to complain about: a
    corpus with gelation labels and no mechanics. That complaint is the
    correct output here, not a bug.
    """
    return [
        PeptideNetworkRecord(
            record_id=f"xu2023_{i:03d}",
            sequence=record.peptide,
            gelation=record.is_gel,
            concentration_mM=record.concentration_mM,
            source_doi="10.1038/s41467-023-39648-2",
            source_locator="Supplementary Data 7",
            notes=f"state={record.state}; morphology={record.morphology}",
        )
        for i, record in enumerate(records)
    ]


# ---------------------------------------------------------------------------
# Protocol confound detection
# ---------------------------------------------------------------------------

@dataclass
class EscalationReport:
    feature_name: str
    crosstab: Dict[Tuple[bool, float], int]
    levels: List[float]
    rule_threshold: Optional[float]
    rule_precision: Optional[float]
    rule_accuracy: Optional[float]
    majority_accuracy: float

    @property
    def is_confounded(self) -> bool:
        """Whether a threshold on the feature predicts the label too well.

        The test is perfect-or-near-perfect precision, not high accuracy:
        a genuinely predictive covariate makes mistakes in both directions,
        while a label-determined one is one-sided by construction.
        """
        return (
            self.rule_precision is not None
            and self.rule_precision >= 0.98
            and self.rule_accuracy is not None
            and self.rule_accuracy > self.majority_accuracy + 0.05
        )

    def summary(self) -> str:
        lines = [
            f"Protocol-escalation check on '{self.feature_name}'",
            "=" * 62,
            f"{'label':<12}" + "".join(f"{level:>9g}" for level in self.levels),
            "-" * 62,
        ]
        for label in (True, False):
            name = "gel" if label else "non-gel"
            counts = [self.crosstab.get((label, lv), 0) for lv in self.levels]
            lines.append(f"{name:<12}" + "".join(f"{c:>9d}" for c in counts))
        if self.rule_threshold is not None:
            lines.append("")
            lines.append(
                f"rule '{self.feature_name} < {self.rule_threshold:g} => positive': "
                f"precision {self.rule_precision:.3f}, "
                f"accuracy {self.rule_accuracy:.3f} "
                f"(majority baseline {self.majority_accuracy:.3f})"
            )
        lines.append("")
        if self.is_confounded:
            lines.append(
                f"!! CONFOUNDED: '{self.feature_name}' is determined by the "
                f"outcome, not chosen independently.\n"
                f"   The measurement protocol escalates until the outcome "
                f"occurs, so this column encodes the label.\n"
                f"   Do NOT use it as a predictive feature. It is admissible "
                f"only as a description of how each row was measured."
            )
        else:
            lines.append(
                f"   No one-sided escalation pattern detected on "
                f"'{self.feature_name}'."
            )
        return "\n".join(lines)


def protocol_escalation_report(
    labels: Sequence[bool],
    feature: Sequence[float],
    feature_name: str = "concentration_mM",
) -> EscalationReport:
    """Detect a feature whose value is set by the measurement protocol.

    Looks for a threshold below which the positive class is essentially
    pure. That one-sidedness is the signature of an escalating protocol:
    rows only take a low value when the outcome already occurred.
    """
    if len(labels) != len(feature):
        raise ValueError("labels and feature must be the same length")

    levels = sorted(set(float(v) for v in feature))
    crosstab: Dict[Tuple[bool, float], int] = Counter(
        (bool(label), float(value)) for label, value in zip(labels, feature)
    )
    n = len(labels)
    positives = sum(1 for label in labels if label)
    majority_accuracy = max(positives, n - positives) / n

    best = (None, None, None)
    for threshold in levels[1:]:
        tp = sum(1 for label, value in zip(labels, feature)
                 if label and value < threshold)
        fp = sum(1 for label, value in zip(labels, feature)
                 if not label and value < threshold)
        tn = sum(1 for label, value in zip(labels, feature)
                 if not label and value >= threshold)
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        acc = (tp + tn) / n
        if best[1] is None or (precision, acc) > (best[1], best[2]):
            best = (threshold, precision, acc)

    return EscalationReport(
        feature_name=feature_name,
        crosstab=dict(crosstab),
        levels=levels,
        rule_threshold=best[0],
        rule_precision=best[1],
        rule_accuracy=best[2],
        majority_accuracy=majority_accuracy,
    )


# ---------------------------------------------------------------------------
# Logistic regression (numpy, Newton-Raphson)
# ---------------------------------------------------------------------------

class LogisticRegressor:
    """L2-penalised logistic regression fitted by Newton-Raphson.

    Written out rather than imported because scikit-learn is not a
    dependency of this repo and the fit is twenty lines. The penalty is not
    optional at this sample size: 165 rows against 29 descriptors separates
    easily, and an unpenalised fit would drive weights to infinity and
    report perfect training accuracy.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        Xs = (X - self.mean_) / self.scale_
        Xa = np.hstack([np.ones((Xs.shape[0], 1)), Xs])

        penalty = np.full(Xa.shape[1], self.l2)
        penalty[0] = 0.0  # never penalise the intercept
        w = np.zeros(Xa.shape[1])

        for _ in range(self.max_iter):
            eta = np.clip(Xa @ w, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            gradient = Xa.T @ (y - p) - penalty * w
            weights = np.clip(p * (1 - p), 1e-9, None)
            hessian = (Xa.T * weights) @ Xa + np.diag(penalty)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:  # pragma: no cover
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            w = w + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit() before predict_proba()")
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        Xa = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        return 1.0 / (1.0 + np.exp(-np.clip(Xa @ self.coef_, -30, 30)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(float)


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------

def grouped_kfold(
    groups: Sequence[int], n_folds: int = 5, seed: int = 0
) -> List[List[int]]:
    """Fold assignment that never splits a group across folds.

    With 165 rows a single holdout would be ~33 points and far too noisy to
    compare models on, so the evaluation is out-of-fold cross-validation.
    Keeping identity clusters whole preserves exactly the no-leakage
    guarantee `identity_holdout.py` provides for a single split.
    """
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of_group = {g: i % n_folds for i, g in enumerate(unique)}
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for index, group in enumerate(groups):
        folds[fold_of_group[group]].append(index)
    return folds


def out_of_fold_predictions(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Sequence[int]],
    l2: float = 1.0,
) -> np.ndarray:
    """Probabilities for every row, each predicted by a model that never saw it."""
    probabilities = np.zeros(len(y))
    for fold in folds:
        test = list(fold)
        train = [i for i in range(len(y)) if i not in set(test)]
        if not train or not test:
            continue
        model = LogisticRegressor(l2=l2).fit(X[train], y[train])
        probabilities[test] = model.predict_proba(X[test])
    return probabilities


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo(path: Optional[str] = None) -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    # --- confound detector self-check on a constructed case ---------------
    synthetic_labels = [True] * 8 + [False] * 8
    escalated = [1.0] * 8 + [2.0] * 8          # label-determined
    independent = [1.0, 2.0] * 8               # genuinely independent
    assert protocol_escalation_report(
        synthetic_labels, escalated, "synthetic").is_confounded
    assert not protocol_escalation_report(
        synthetic_labels, independent, "synthetic").is_confounded
    print("--- escalation detector: flags a label-determined feature, "
          "clears an independent one -> OK ---")

    # --- logistic regression self-check -----------------------------------
    rng = np.random.default_rng(0)
    X_synth = rng.normal(size=(300, 2))
    y_synth = (X_synth[:, 0] * 2.0 - X_synth[:, 1] > 0).astype(float)
    clf = LogisticRegressor(l2=1e-3).fit(X_synth, y_synth)
    assert (clf.predict(X_synth) == y_synth).mean() > 0.95
    print("--- logistic regression recovers a linear boundary -> OK ---")

    print()
    try:
        records = load_gelation_records(path)
    except DatasetNotAvailableError as exc:
        print(f"--- dataset not present ---\n{exc}")
        return

    print(f"--- Xu 2023 Supplementary Data 7: {len(records)} tetrapeptides ---")
    print(f"  states      : {dict(Counter(r.state for r in records).most_common())}")
    print(f"  gel rate    : {sum(1 for r in records if r.is_gel)/len(records):.1%}")
    print(f"  morphologies: {dict(Counter(r.morphology for r in records).most_common(6))}")
    print(f"  fibrous     : {sum(1 for r in records if r.is_fibrous)} "
          f"of {len(records)}")
    print(f"  lengths     : {dict(Counter(len(r.peptide) for r in records))}")

    # --- the confound ------------------------------------------------------
    print()
    report = protocol_escalation_report(
        [r.is_gel for r in records],
        [r.concentration_mM for r in records],
        "concentration_mM",
    )
    print(report.summary())
    assert report.is_confounded, "the escalation confound should be detected"

    # --- schema audit ------------------------------------------------------
    print()
    audit = heterogeneity_report(to_network_records(records))
    print(f"--- canonical schema audit ---")
    print(f"  records with a modulus       : {audit.n_with_modulus}")
    print(f"  comparable protocol blocks   : {audit.n_blocks}")
    print(f"  {audit.verdict()}")

    # --- identity structure ------------------------------------------------
    print()
    peptides = [r.peptide for r in records]
    # identity >= 0.6 on tetrapeptides means at most floor(0.4*4) = 1 edit
    clusters = substitution_clusters(peptides, 1)
    sizes = Counter(clusters)
    largest = max(sizes.values())
    print(f"--- identity clusters at 0.60 (<=1 substitution) ---")
    print(f"  {len(sizes)} clusters over {len(peptides)} peptides; "
          f"largest {largest} ({largest/len(peptides):.0%} of the set)")
    print(f"  singletons: {sum(1 for v in sizes.values() if v == 1)}")
    if largest > len(peptides) * 0.25:
        print(
            "  -> SUBSTANTIAL near-duplicate structure. These 165 were "
            "selected around top-ranked\n     sequences, so they form "
            "single-substitution families rather than a random sample. A "
            "random\n     split would train and test on the same family; "
            "the grouped CV below holds clusters whole."
        )
    else:
        print("  -> little near-duplicate structure; a random split would "
              "behave similarly.")

    # --- models ------------------------------------------------------------
    matrix, names = feature_matrix(peptides)
    X = np.array(matrix, dtype=float)
    folds = grouped_kfold(clusters, n_folds=5, seed=0)

    for target_name, y_values in (
        ("gelation", [r.is_gel for r in records]),
        ("fibrous morphology", [r.is_fibrous for r in records]),
    ):
        y = np.array([1.0 if v else 0.0 for v in y_values])
        probabilities = out_of_fold_predictions(X, y, folds, l2=1.0)
        predictions = (probabilities >= 0.5).astype(float)

        majority = MajorityBaseline().fit(y.tolist())
        # The authors' own computed score, as the strong baseline. Lower rank
        # means more assembly-prone, so negate it to make "higher is more
        # positive" hold for AUC.
        aphc = np.array([
            -(r.aphc_rank if r.aphc_rank is not None else 10 ** 6)
            for r in records
        ], dtype=float)

        evaluation = evaluate_binary(
            target=f"{target_name} (identity-grouped 5-fold OOF)",
            y_true=y.astype(int).tolist(),
            predictions={"logistic_descriptors": predictions.tolist()},
            baselines={
                "majority": majority.predict(len(y)),
                "authors_APHC_rank": (aphc > np.median(aphc)).astype(float).tolist(),
            },
            scores={
                "logistic_descriptors": probabilities.tolist(),
                "authors_APHC_rank": aphc.tolist(),
            },
        )
        print()
        print(evaluation.summary(model_name="logistic_descriptors"))
        print(
            "NOTE: authors_APHC_rank is not a fair comparator. These 165\n"
            "      peptides were SELECTED by that score, enriched at both\n"
            "      ends of its ranking, so its separation here is inflated by\n"
            "      the selection and is an upper bound on what it would\n"
            "      achieve on randomly drawn tetrapeptides. The descriptor\n"
            "      model faces no such advantage. Treat the gap as an\n"
            "      unresolved comparison, not as a measured defeat."
        )

    # --- the leakage demonstration ----------------------------------------
    print()
    print("--- what happens if the confounded column IS used as a feature ---")
    y = np.array([1.0 if r.is_gel else 0.0 for r in records])
    X_leaky = np.hstack([
        X, np.array([[r.concentration_mM] for r in records], dtype=float)
    ])
    leaky = out_of_fold_predictions(X_leaky, y, folds, l2=1.0)
    clean = out_of_fold_predictions(X, y, folds, l2=1.0)
    from baselines import accuracy as _accuracy, balanced_accuracy as _bal
    print(f"  descriptors only        : acc {_accuracy(y.astype(int).tolist(), (clean >= 0.5).astype(float).tolist()):.3f}  "
          f"balanced {_bal(y.astype(int).tolist(), (clean >= 0.5).astype(float).tolist()):.3f}  "
          f"AUC {roc_auc(y.astype(int).tolist(), clean.tolist()):.3f}")
    print(f"  + concentration (LEAKY) : acc {_accuracy(y.astype(int).tolist(), (leaky >= 0.5).astype(float).tolist()):.3f}  "
          f"balanced {_bal(y.astype(int).tolist(), (leaky >= 0.5).astype(float).tolist()):.3f}  "
          f"AUC {roc_auc(y.astype(int).tolist(), leaky.tolist()):.3f}")
    print("  The leaky model looks better and is worthless: its advantage "
          "comes from a column\n  that exists only because the outcome was "
          "already observed.")

    # --- what the identity grouping is worth ------------------------------
    print()
    print("--- random folds vs identity-grouped folds (same model, same data) ---")
    truth = y.astype(int).tolist()
    random_folds = grouped_kfold(list(range(len(records))), n_folds=5, seed=0)
    random_oof = out_of_fold_predictions(X, y, random_folds, l2=1.0)
    grouped_auc = roc_auc(truth, clean.tolist())
    random_auc = roc_auc(truth, random_oof.tolist())
    print(f"  random folds (each peptide its own group) : AUC {random_auc:.3f}  "
          f"balanced {_bal(truth, (random_oof >= 0.5).astype(float).tolist()):.3f}")
    print(f"  identity-grouped folds                    : AUC {grouped_auc:.3f}  "
          f"balanced {_bal(truth, (clean >= 0.5).astype(float).tolist()):.3f}")
    print(f"  inflation from splitting families across folds: "
          f"{random_auc - grouped_auc:+.3f} AUC")
    print("  Both numbers are honest out-of-fold estimates. Only the second "
          "answers the\n  question Aim 2 asks, which is about new sequences "
          "rather than new members of\n  families already seen.")

    print()
    print("All xu2023_gelation.py self-checks passed.")


if __name__ == "__main__":
    _run_demo(sys.argv[1] if len(sys.argv) > 1 else None)
