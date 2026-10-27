"""
Mandatory Baselines and Criterion Audit for Aim 2
====================================================

This module exists because of Section C.6 of the proposal.

The silk pipeline produced traces that were 99.4% grounded across 5,259
individually checked claims, with 86% fully grounded. Those same traces
correlated -0.06 (strength) and -0.40 (toughness) with the measured
property, while chain length alone reached +0.57 and +0.98. Predicting the
dataset mean beat the traces on absolute error, and the pipeline's own
correctness criterion was satisfied by a constant guess 93% of the time
against the traces' 8%.

Every one of those numbers is a *comparison*. None of them is visible in a
grounding score, and none of them is visible in an accuracy number reported
on its own. The failure was not that the pipeline scored badly; it was that
nothing in the reporting path forced the comparison that would have shown
the score was meaningless. Training against that criterion would have
improved every grounding metric while driving the model toward confident,
well-cited and uninformative reasoning.

So this module makes the comparison structural rather than optional:

  * `evaluate()` will not report a bare score. Every report carries the
    trivial baseline beside the model, and names any required baseline that
    was not supplied instead of quietly omitting it.

  * `SingleCovariateBaseline` is first-class, because in silk the thing that
    embarrassed the pipeline was a one-variable baseline nobody had run.
    For peptide networks the analogous covariate is concentration (modulus
    scales steeply with it) or peptide length. Run them before the agent,
    not after.

  * `criterion_informativeness()` generalises Figure 4c into a reusable
    guard: given a correctness predicate, it computes how often the *best
    possible constant guess* satisfies it. If a constant can satisfy the
    criterion more often than the model, the criterion is not measuring
    prediction and must not be used as a training signal.

The proposal commits to two baselines simultaneously — "a trivial predictor
and the best available sequence-embedding model". The embedding baseline
needs a model this repo does not carry, so it is a named required slot that
reports as MISSING rather than a stub that silently returns zeros.

Runnable standalone: `python3 aim2_peptide_networks/baselines.py`
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# The two baselines Aim 2's evaluation section commits to reporting.
REQUIRED_BASELINES = ("trivial", "sequence_embedding")


class BaselineError(ValueError):
    """Raised when an evaluation is malformed or cannot be carried out."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _check_pair(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    if len(y_true) != len(y_pred):
        raise BaselineError(
            f"length mismatch: {len(y_true)} true values, {len(y_pred)} predictions"
        )
    if not y_true:
        raise BaselineError("cannot evaluate an empty set")


def rankdata(values: Sequence[float]) -> List[float]:
    """Ranks with ties averaged, matching the usual Spearman convention.

    Ties are not a corner case here: gelation-adjacent datasets contain
    repeated moduli and censored values ("> 10 kPa"), and a ranking that
    broke ties by position would manufacture correlation out of row order.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    """Pearson r, or None if either side has zero variance.

    None rather than 0.0 on purpose: a constant predictor has an *undefined*
    correlation, not an uncorrelated one, and reporting 0.0 would let a
    degenerate predictor look merely mediocre instead of inadmissible.
    """
    _check_pair(y_true, y_pred)
    n = len(y_true)
    mean_t = sum(y_true) / n
    mean_p = sum(y_pred) / n
    cov = sum((t - mean_t) * (p - mean_p) for t, p in zip(y_true, y_pred))
    var_t = sum((t - mean_t) ** 2 for t in y_true)
    var_p = sum((p - mean_p) ** 2 for p in y_pred)
    if var_t <= 0 or var_p <= 0:
        return None
    return cov / math.sqrt(var_t * var_p)


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    """Spearman rho via Pearson on tie-averaged ranks."""
    _check_pair(y_true, y_pred)
    return pearson(rankdata(y_true), rankdata(y_pred))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _check_pair(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _check_pair(y_true, y_pred)
    return math.sqrt(
        sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    )


def hit_rate(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    tolerance: float,
) -> float:
    """Fraction of predictions within `tolerance` of the truth.

    This is the shape of the silk pipeline's correctness criterion (error
    <= 0.1). Always pair it with `criterion_informativeness()`.
    """
    _check_pair(y_true, y_pred)
    return sum(
        1 for t, p in zip(y_true, y_pred) if abs(t - p) <= tolerance
    ) / len(y_true)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TrivialBaseline:
    """Predict a single constant fitted on the training set.

    The constant is the training mean by default, or the median with
    `strategy="median"`. For a heavy-tailed target like modulus, the median
    is the stronger adversary on absolute error — and the point of a
    baseline is to be as strong as it honestly can be, not to be easy to
    beat.
    """

    def __init__(self, strategy: str = "mean") -> None:
        if strategy not in ("mean", "median"):
            raise BaselineError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.constant: Optional[float] = None

    def fit(self, y_train: Sequence[float]) -> "TrivialBaseline":
        if not y_train:
            raise BaselineError("cannot fit a trivial baseline on no data")
        values = sorted(y_train)
        if self.strategy == "mean":
            self.constant = sum(values) / len(values)
        else:
            mid = len(values) // 2
            self.constant = (
                values[mid] if len(values) % 2
                else (values[mid - 1] + values[mid]) / 2.0
            )
        return self

    def predict(self, n: int) -> List[float]:
        if self.constant is None:
            raise BaselineError("fit() before predict()")
        return [self.constant] * n


class SingleCovariateBaseline:
    """Ordinary least squares on exactly one feature.

    This is the baseline that mattered most in silk: chain length alone
    reached Spearman +0.98 on toughness, against -0.40 for the full
    tool-grounded pipeline. A composed chain that cannot beat one
    hand-picked covariate has not demonstrated anything, however well
    grounded each of its steps is.

    `log_target=True` fits in log10 space, which is the right default for
    network modulus: published values span several decades, so a linear fit
    would be dominated by the stiffest few gels and would understate how
    much of the variance one covariate already explains.
    """

    def __init__(self, name: str, log_target: bool = False) -> None:
        self.name = name
        self.log_target = log_target
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None

    def _transform(self, values: Sequence[float]) -> List[float]:
        if not self.log_target:
            return list(values)
        if any(v <= 0 for v in values):
            raise BaselineError(
                f"{self.name}: log_target=True requires strictly positive "
                "targets; a zero or negative modulus in the training set "
                "means the corpus needs cleaning, not a silent offset"
            )
        return [math.log10(v) for v in values]

    def fit(
        self,
        x_train: Sequence[float],
        y_train: Sequence[float],
    ) -> "SingleCovariateBaseline":
        _check_pair(x_train, y_train)
        y = self._transform(y_train)
        n = len(x_train)
        mean_x = sum(x_train) / n
        mean_y = sum(y) / n
        var_x = sum((x - mean_x) ** 2 for x in x_train)
        if var_x <= 0:
            raise BaselineError(
                f"{self.name}: covariate is constant in the training set"
            )
        cov = sum((x - mean_x) * (yi - mean_y) for x, yi in zip(x_train, y))
        self.slope = cov / var_x
        self.intercept = mean_y - self.slope * mean_x
        return self

    def predict(self, x_test: Sequence[float]) -> List[float]:
        if self.slope is None or self.intercept is None:
            raise BaselineError("fit() before predict()")
        predictions = [self.intercept + self.slope * x for x in x_test]
        if self.log_target:
            return [10.0 ** p for p in predictions]
        return predictions


# ---------------------------------------------------------------------------
# Criterion audit (Figure 4c, generalised)
# ---------------------------------------------------------------------------

@dataclass
class CriterionAudit:
    tolerance: float
    model_hit_rate: float
    mean_guess_hit_rate: float
    best_constant: float
    best_constant_hit_rate: float

    @property
    def is_informative(self) -> bool:
        """Whether beating the criterion requires actually predicting.

        False when a single constant satisfies the criterion at least as
        often as the model does. In silk this was 93% against 8%.
        """
        return self.model_hit_rate > self.best_constant_hit_rate

    def summary(self) -> str:
        lines = [
            f"Correctness criterion: |error| <= {self.tolerance:g}",
            f"  model                      : {self.model_hit_rate:6.1%}",
            f"  constant = training mean   : {self.mean_guess_hit_rate:6.1%}",
            f"  best possible constant     : {self.best_constant_hit_rate:6.1%} "
            f"(at {self.best_constant:g})",
        ]
        if self.is_informative:
            lines.append(
                "  -> INFORMATIVE: no constant matches the model's hit rate."
            )
        else:
            lines.append(
                "  -> UNINFORMATIVE: a constant guess satisfies this criterion "
                "at least as often as the model.\n"
                "     Do not use it as a preference or reward signal. "
                "Optimising against it would reward confident, well-grounded "
                "and uninformative reasoning (proposal C.6). Tighten the "
                "tolerance, score in log space, or use a ranking objective."
            )
        return "\n".join(lines)


def best_constant_hit_rate(
    y_true: Sequence[float],
    tolerance: float,
) -> Tuple[float, float]:
    """The constant guess that satisfies |error| <= tolerance most often.

    A sweep over interval overlaps: each true value y admits any constant in
    [y - tolerance, y + tolerance], so the best constant sits where the most
    such intervals overlap. Exact, O(n log n), and it makes the baseline the
    strongest constant rather than merely the mean — the honest adversary.
    """
    if not y_true:
        raise BaselineError("cannot audit a criterion on no data")
    if tolerance < 0:
        raise BaselineError(f"tolerance must be non-negative, got {tolerance}")

    events: List[Tuple[float, int]] = []
    for y in y_true:
        events.append((y - tolerance, 1))
        events.append((y + tolerance, -1))
    # Opening before closing at the same coordinate: a constant exactly on a
    # boundary is within tolerance, since the criterion uses <=.
    events.sort(key=lambda e: (e[0], -e[1]))

    best_count = 0
    best_point = y_true[0]
    running = 0
    for point, delta in events:
        running += delta
        if running > best_count:
            best_count = running
            best_point = point
    return best_point, best_count / len(y_true)


def criterion_informativeness(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    tolerance: float,
    train_mean: Optional[float] = None,
) -> CriterionAudit:
    """Check whether a tolerance-based correctness criterion measures anything."""
    _check_pair(y_true, y_pred)
    if train_mean is None:
        train_mean = sum(y_true) / len(y_true)
    constant, constant_rate = best_constant_hit_rate(y_true, tolerance)
    return CriterionAudit(
        tolerance=tolerance,
        model_hit_rate=hit_rate(y_true, y_pred, tolerance),
        mean_guess_hit_rate=hit_rate(
            y_true, [train_mean] * len(y_true), tolerance
        ),
        best_constant=constant,
        best_constant_hit_rate=constant_rate,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class MetricRow:
    name: str
    spearman: Optional[float]
    pearson: Optional[float]
    mae: float
    rmse: float
    is_baseline: bool = False


@dataclass
class EvaluationReport:
    target: str
    n_test: int
    rows: List[MetricRow]
    missing_required_baselines: List[str] = field(default_factory=list)
    criterion: Optional[CriterionAudit] = None

    def row(self, name: str) -> Optional[MetricRow]:
        for candidate in self.rows:
            if candidate.name == name:
                return candidate
        return None

    def beats_all_baselines(self, model_name: str) -> Optional[bool]:
        """True only if the model beats every baseline on MAE and on rho.

        None when the comparison cannot be made — an absent Spearman (a
        constant predictor) or a missing model row. None is not a pass.
        """
        model = self.row(model_name)
        if model is None or model.spearman is None:
            return None
        for other in self.rows:
            if other.is_baseline:
                if other.mae <= model.mae:
                    return False
                if other.spearman is not None and other.spearman >= model.spearman:
                    return False
        return True

    def summary(self, model_name: Optional[str] = None) -> str:
        lines = [
            f"Evaluation: {self.target}  (n = {self.n_test} held-out)",
            "=" * 76,
            f"{'predictor':<30} {'rho':>8} {'r':>8} {'MAE':>12} {'RMSE':>12}",
            "-" * 76,
        ]
        for row in self.rows:
            rho = "    n/a " if row.spearman is None else f"{row.spearman:8.3f}"
            r = "    n/a " if row.pearson is None else f"{row.pearson:8.3f}"
            tag = "  [baseline]" if row.is_baseline else ""
            lines.append(
                f"{row.name + tag:<30} {rho} {r} {row.mae:12.4g} {row.rmse:12.4g}"
            )

        if self.missing_required_baselines:
            lines.append("")
            lines.append(
                "!! REQUIRED BASELINE NOT SUPPLIED: "
                + ", ".join(self.missing_required_baselines)
            )
            lines.append(
                "   Aim 2 commits to reporting against a trivial predictor AND "
                "the best available sequence-embedding model. This report is "
                "incomplete until both are present."
            )

        if self.criterion is not None:
            lines.append("")
            lines.append(self.criterion.summary())

        if model_name is not None:
            lines.append("")
            verdict = self.beats_all_baselines(model_name)
            if verdict is True:
                lines.append(
                    f"VERDICT: {model_name} beats every supplied baseline on both "
                    "MAE and Spearman."
                )
            elif verdict is False:
                lines.append(
                    f"VERDICT: {model_name} does NOT beat every supplied baseline. "
                    "Per C.6 this is the silk outcome: report it as such and "
                    "diagnose which link in the chain fails before training on "
                    "the signal."
                )
            else:
                lines.append(
                    f"VERDICT: cannot be determined for {model_name!r} "
                    "(missing row, or an undefined correlation)."
                )
        return "\n".join(lines)


def evaluate(
    target: str,
    y_true: Sequence[float],
    predictions: Dict[str, Sequence[float]],
    baselines: Optional[Dict[str, Sequence[float]]] = None,
    tolerance: Optional[float] = None,
    primary_model: Optional[str] = None,
    required_baselines: Sequence[str] = REQUIRED_BASELINES,
) -> EvaluationReport:
    """Score predictors against the truth, always beside their baselines.

    `baselines` is separate from `predictions` so the report can tell the two
    apart and so a missing required baseline is detectable. Supplying no
    baselines at all is an error rather than a permitted shortcut: the whole
    point of this module is that a bare score is not reportable.
    """
    if not predictions:
        raise BaselineError("nothing to evaluate")
    baselines = dict(baselines or {})
    if not baselines:
        raise BaselineError(
            "no baselines supplied; a score without a baseline is not "
            "interpretable (see proposal C.6)"
        )

    rows: List[MetricRow] = []
    for name, y_pred in list(predictions.items()) + list(baselines.items()):
        _check_pair(y_true, y_pred)
        rows.append(MetricRow(
            name=name,
            spearman=spearman(y_true, y_pred),
            pearson=pearson(y_true, y_pred),
            mae=mae(y_true, y_pred),
            rmse=rmse(y_true, y_pred),
            is_baseline=name in baselines,
        ))

    missing = [name for name in required_baselines if name not in baselines]

    criterion = None
    if tolerance is not None and primary_model is not None:
        if primary_model not in predictions:
            raise BaselineError(
                f"primary_model {primary_model!r} is not among the predictions"
            )
        criterion = criterion_informativeness(
            y_true, predictions[primary_model], tolerance
        )

    return EvaluationReport(
        target=target,
        n_test=len(y_true),
        rows=rows,
        missing_required_baselines=missing,
        criterion=criterion,
    )


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _silk_shaped_demo() -> None:
    """Reproduce the C.6 failure shape, so the guard is shown to catch it.

    Synthetic numbers chosen to mimic the reported situation: a predictor
    that is uncorrelated-to-anticorrelated with truth, a strong single
    covariate, and a tolerance criterion a constant satisfies most of the
    time. If this module cannot flag this, it is not doing its job.
    """
    import random as _random
    rng = _random.Random(7)

    n = 120
    covariate = [rng.uniform(1.0, 10.0) for _ in range(n)]
    # Truth is mostly the covariate, tightly clustered -- the regime where a
    # constant guess does well on a loose tolerance.
    y_true = [1.0 + 0.05 * c + rng.gauss(0.0, 0.02) for c in covariate]
    # A "well-grounded but uninformative" predictor: right ballpark, wrong
    # ordering.
    y_model = [1.25 + rng.gauss(0.0, 0.12) for _ in range(n)]

    split = int(0.7 * n)
    trivial = TrivialBaseline("mean").fit(y_true[:split])
    covariate_baseline = SingleCovariateBaseline("concentration").fit(
        covariate[:split], y_true[:split]
    )

    y_test = y_true[split:]
    report = evaluate(
        target="network modulus (synthetic, C.6-shaped)",
        y_true=y_test,
        predictions={"composed_chain": y_model[split:]},
        baselines={
            "trivial": trivial.predict(len(y_test)),
            "single_covariate": covariate_baseline.predict(covariate[split:]),
        },
        tolerance=0.1,
        primary_model="composed_chain",
    )
    print(report.summary(model_name="composed_chain"))

    assert report.beats_all_baselines("composed_chain") is False, (
        "the C.6-shaped demo should NOT beat its baselines"
    )
    assert report.criterion is not None
    assert not report.criterion.is_informative, (
        "a loose tolerance on a tightly clustered target should be flagged "
        "uninformative"
    )
    assert "sequence_embedding" in report.missing_required_baselines


def _run_demo() -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    print("--- metrics ---")
    assert rankdata([10.0, 20.0, 20.0, 40.0]) == [1.0, 2.5, 2.5, 4.0]
    print("  tie-averaged ranks -> OK")
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    print("  spearman on monotone / reversed -> OK")
    assert pearson([1, 2, 3], [5, 5, 5]) is None
    print("  constant predictor gives an undefined r, not 0.0 -> OK")

    print()
    print("--- best-constant sweep ---")
    point, rate = best_constant_hit_rate([1.0, 1.05, 1.1, 5.0], tolerance=0.1)
    assert abs(rate - 0.75) < 1e-12, rate
    print(f"  three values within +/-0.1 of {point:g}: hit rate {rate:.0%} -> OK")
    point, rate = best_constant_hit_rate([0.0, 10.0, 20.0], tolerance=0.1)
    assert abs(rate - 1 / 3) < 1e-12
    print(f"  well-separated values: best constant only reaches {rate:.0%} -> OK")

    print()
    print("--- a bare score is not reportable ---")
    try:
        evaluate("x", [1.0, 2.0], {"model": [1.0, 2.0]}, baselines={})
    except BaselineError as exc:
        print(f"  refused: {exc}")
    else:
        raise AssertionError("evaluate() should refuse to report without a baseline")

    print()
    print("--- the C.6 failure shape, caught ---")
    print()
    _silk_shaped_demo()

    print()
    print("--- and a genuinely predictive model, for contrast ---")
    import random as _random
    rng = _random.Random(11)
    n = 120
    covariate = [rng.uniform(1.0, 10.0) for _ in range(n)]
    y_true = [1.0 + 0.5 * c + rng.gauss(0.0, 0.1) for c in covariate]
    y_model = [t + rng.gauss(0.0, 0.05) for t in y_true]
    split = int(0.7 * n)
    trivial = TrivialBaseline("mean").fit(y_true[:split])
    y_test = y_true[split:]
    report = evaluate(
        target="network modulus (synthetic, predictive)",
        y_true=y_test,
        predictions={"composed_chain": y_model[split:]},
        baselines={
            "trivial": trivial.predict(len(y_test)),
            "sequence_embedding": [
                t + rng.gauss(0.0, 0.4) for t in y_test
            ],
        },
        tolerance=0.1,
        primary_model="composed_chain",
    )
    print(report.summary(model_name="composed_chain"))
    assert report.beats_all_baselines("composed_chain") is True
    assert report.criterion is not None and report.criterion.is_informative
    assert not report.missing_required_baselines

    print()
    print("All baselines.py self-checks passed.")


if __name__ == "__main__":
    _run_demo()
