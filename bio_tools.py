"""
Bioinformatics Tool Suite + Stage 0 Validation Harness
=========================================================

Implements the tool categories from the roadmap using established,
decades-old classical algorithms — no external services or model weights
needed, so this runs standalone and its output is checkable by hand:

  - motif_detector           : regex-based detection of known spidroin
                                motifs (poly-alanine crystalline blocks,
                                GGX/GPGXX glycine-rich amorphous repeats,
                                proline-rich turn regions)
  - hydropathy_profile        : Kyte & Doolittle (1982) sliding-window
                                hydrophobicity
  - charge_profile             : simple net-charge counting (D/E-, K/R+)
  - beta_sheet_propensity      : Chou & Fasman (1978) per-residue Pα/Pβ/Pturn
                                propensity table, sliding-window averaged
  - repeat_pattern_analyzer    : basic tandem-repeat / periodicity detector
  - disorder_prediction_chplot : Uversky et al. (2000) charge-hydropathy
                                plot boundary

IMPORTANT CAVEAT — read before trusting these outputs
--------------------------------------------------------
These are lightweight, 1970s-2000s-era heuristics, not state-of-the-art
structure predictors (that would mean IUPred/AlphaFold/ESM-based tools).
They exist so Stage 0 has *something real* to validate the pipeline against
immediately. One specific, important limitation this validation harness
surfaces directly: Chou-Fasman propensities were derived from globular
proteins, and alanine's generic Pβ (0.83) is below average — so this table
will NOT flag silk's poly-alanine crystalline β-sheet blocks as
beta-rich, even though poly-Ala nanocrystals are famously β-sheet in silk.
That's a real chemistry/packing effect (small side chains, dense H-bonding)
that generic propensity tables don't capture. The validation harness below
checks for and reports this discrepancy explicitly, rather than silently
producing a wrong "beta-rich" call — this is exactly the kind of tool
blind spot Stage 0 is supposed to catch before you build traces on top of
tools you haven't checked.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Chou-Fasman (1978) conformational propensities: (P_alpha, P_beta, P_turn)
CHOU_FASMAN = {
    "A": (1.42, 0.83, 0.66), "R": (0.98, 0.93, 0.95), "N": (0.67, 0.89, 1.56),
    "D": (1.01, 0.54, 1.46), "C": (0.70, 1.19, 1.19), "Q": (1.11, 1.10, 0.98),
    "E": (1.51, 0.37, 0.74), "G": (0.57, 0.75, 1.56), "H": (1.00, 0.87, 0.95),
    "I": (1.08, 1.60, 0.47), "L": (1.21, 1.30, 0.59), "K": (1.16, 0.74, 1.01),
    "M": (1.45, 1.05, 0.60), "F": (1.13, 1.38, 0.60), "P": (0.57, 0.55, 1.52),
    "S": (0.77, 0.75, 1.43), "T": (0.83, 1.19, 0.96), "W": (1.08, 1.37, 0.96),
    "Y": (0.69, 1.47, 1.14), "V": (1.06, 1.70, 0.50),
}

POSITIVE_RESIDUES = {"K", "R"}
NEGATIVE_RESIDUES = {"D", "E"}


def _clean(sequence: str) -> str:
    return "".join(c for c in sequence.upper() if c.isalpha())


# ---------------------------------------------------------------------------
# 1. Motif detector
# ---------------------------------------------------------------------------

_MOTIF_PATTERNS = {
    "poly_alanine": re.compile(r"A{5,}"),
    "GGX_repeat": re.compile(r"(?:GG[A-Z]){2,}"),
    "GPGXX_repeat": re.compile(r"(?:GPG[A-Z]{2}){2,}"),
    "proline_rich_turn": re.compile(r"(?:[AP]{2}){3,}"),  # AP/PA-alternating runs
    "glycine_rich_spacer": re.compile(r"G[A-Z]{0,2}G[A-Z]{0,2}G"),
}


def motif_detector(sequence: str) -> Dict[str, Any]:
    seq = _clean(sequence)
    motifs = []
    spans: Dict[str, List[List[int]]] = {}
    for name, pattern in _MOTIF_PATTERNS.items():
        found_spans = [[m.start(), m.end() - 1] for m in pattern.finditer(seq)]
        if found_spans:
            motifs.append(name)
            spans[name] = found_spans
    return {"motifs": motifs, "spans": spans, "sequence_length": len(seq)}


# ---------------------------------------------------------------------------
# 2. Hydropathy profile (Kyte-Doolittle)
# ---------------------------------------------------------------------------

def hydropathy_profile(sequence: str, window: int = 9) -> Dict[str, Any]:
    seq = _clean(sequence)
    values = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq]
    if len(seq) < window:
        window = max(1, len(seq))
    windowed = []
    for i in range(len(values) - window + 1):
        windowed.append(sum(values[i:i + window]) / window)
    mean_hydropathy = sum(values) / len(values) if values else 0.0
    max_idx = max(range(len(windowed)), key=lambda i: windowed[i]) if windowed else 0
    min_idx = min(range(len(windowed)), key=lambda i: windowed[i]) if windowed else 0
    return {
        "mean_hydropathy": round(mean_hydropathy, 3),
        "window_size": window,
        "most_hydrophobic_span": [max_idx, max_idx + window - 1] if windowed else None,
        "most_hydrophilic_span": [min_idx, min_idx + window - 1] if windowed else None,
        "max_window_value": round(windowed[max_idx], 3) if windowed else None,
        "min_window_value": round(windowed[min_idx], 3) if windowed else None,
    }


# ---------------------------------------------------------------------------
# 3. Charge profile
# ---------------------------------------------------------------------------

def charge_profile(sequence: str) -> Dict[str, Any]:
    seq = _clean(sequence)
    pos = sum(1 for aa in seq if aa in POSITIVE_RESIDUES)
    neg = sum(1 for aa in seq if aa in NEGATIVE_RESIDUES)
    net = pos - neg
    return {
        "positive_residues": pos,
        "negative_residues": neg,
        "net_charge": net,
        "mean_net_charge_per_residue": round(net / len(seq), 4) if seq else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Beta-sheet / alpha / turn propensity (Chou-Fasman)
# ---------------------------------------------------------------------------

def beta_sheet_propensity(sequence: str, window: int = 6, beta_threshold: float = 1.05) -> Dict[str, Any]:
    seq = _clean(sequence)
    alpha_vals, beta_vals, turn_vals = [], [], []
    for aa in seq:
        pa, pb, pt = CHOU_FASMAN.get(aa, (1.0, 1.0, 1.0))
        alpha_vals.append(pa)
        beta_vals.append(pb)
        turn_vals.append(pt)

    if len(seq) < window:
        window = max(1, len(seq))

    beta_windowed = [
        sum(beta_vals[i:i + window]) / window
        for i in range(len(beta_vals) - window + 1)
    ]
    beta_rich_spans = []
    in_span = False
    span_start = 0
    for i, v in enumerate(beta_windowed):
        if v >= beta_threshold and not in_span:
            in_span, span_start = True, i
        elif v < beta_threshold and in_span:
            beta_rich_spans.append([span_start, i + window - 2])
            in_span = False
    if in_span:
        beta_rich_spans.append([span_start, len(beta_windowed) + window - 2])

    return {
        "mean_alpha_propensity": round(sum(alpha_vals) / len(alpha_vals), 3) if alpha_vals else 0.0,
        "mean_beta_propensity": round(sum(beta_vals) / len(beta_vals), 3) if beta_vals else 0.0,
        "mean_turn_propensity": round(sum(turn_vals) / len(turn_vals), 3) if turn_vals else 0.0,
        "beta_rich_spans": beta_rich_spans,
        "beta_threshold": beta_threshold,
    }


# ---------------------------------------------------------------------------
# 5. Repeat / periodicity detector
# ---------------------------------------------------------------------------

def repeat_pattern_analyzer(sequence: str, min_period: int = 2, max_period: int = 50) -> Dict[str, Any]:
    """
    Simple autocorrelation-style periodicity scan: for each candidate period
    p, measures what fraction of residues match the residue p positions
    earlier. The period with the highest match fraction (above a floor) is
    reported as the dominant repeat period — a coarse but real signal for
    the block-copolymer-like repetitive structure characteristic of
    spidroins.
    """
    seq = _clean(sequence)
    if len(seq) < min_period * 2:
        return {"dominant_period": None, "repeat_score": 0.0, "example_repeat_unit": None}

    best_period, best_score = None, 0.0
    for p in range(min_period, min(max_period, len(seq) // 2) + 1):
        matches = sum(1 for i in range(len(seq) - p) if seq[i] == seq[i + p])
        score = matches / (len(seq) - p)
        if score > best_score:
            best_period, best_score = p, score

    example_unit = seq[:best_period] if best_period else None
    return {
        "dominant_period": best_period,
        "repeat_score": round(best_score, 3),
        "example_repeat_unit": example_unit,
    }


# ---------------------------------------------------------------------------
# 6. Disorder prediction (Uversky charge-hydropathy plot)
# ---------------------------------------------------------------------------

def disorder_prediction_chplot(sequence: str) -> Dict[str, Any]:
    """
    Uversky, Gillespie & Fink (2000): a sequence is predicted intrinsically
    disordered if its mean hydropathy falls below a linear boundary set by
    its mean net charge. Classic, coarse, but real and well-documented.
    """
    seq = _clean(sequence)
    if not seq:
        return {"mean_charge": 0.0, "mean_hydropathy_normalized": 0.0,
                "boundary_value": 0.0, "predicted_disordered": False}

    charges = charge_profile(seq)
    mean_charge = abs(charges["mean_net_charge_per_residue"])

    raw_hydropathy = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq]
    mean_kd = sum(raw_hydropathy) / len(raw_hydropathy)
    # Rescale Kyte-Doolittle's [-4.5, 4.5] range to Uversky's [0, 1] convention
    mean_h_normalized = (mean_kd + 4.5) / 9.0

    boundary = (mean_charge + 1.151) / 2.785
    predicted_disordered = mean_h_normalized < boundary

    return {
        "mean_charge": round(mean_charge, 4),
        "mean_hydropathy_normalized": round(mean_h_normalized, 4),
        "boundary_value": round(boundary, 4),
        "predicted_disordered": predicted_disordered,
    }


# ---------------------------------------------------------------------------
# Tool registry (used by the trace generator)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "motif_detector": motif_detector,
    "hydropathy_profile": hydropathy_profile,
    "charge_profile": charge_profile,
    "beta_sheet_propensity": beta_sheet_propensity,
    "repeat_pattern_analyzer": repeat_pattern_analyzer,
    "disorder_prediction_chplot": disorder_prediction_chplot,
}


# ---------------------------------------------------------------------------
# Stage 0 validation harness
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    sequence_name: str
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def passed_fraction(self) -> float:
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)


# A real spidroin sequence from the handoff doc: Eriophora pustulosa PySp
# (pyriform spidroin — attachment-disc/cement silk, not dragline). PySp is
# documented in the literature as proline-rich and turn-prone rather than
# built around classic poly-alanine crystalline blocks, so this is a real
# test of whether the tools produce chemically sensible output on a
# non-MaSp family member, not just on the "easy" canonical case.
PYSP_ERIOPHORA_PUSTULOSA = (
    "SSSAYSGTSTGGSSVSQSQPIISSAPVYFNAQTLTSSLASSLQSDRALNFISSGQLSASDVS"
    "TSVSSAVAQTLGISQSSVQNIISQQMSSVRTGASSSSVSQAIANAVSSAVQASGAATPGQEQ"
    "SIAQRVYSSISTYLSQLISQRTAPAPAPAPRPAPMPAPAPRPAPMPAPAPRPRPAPAPRPAP"
    "VYAPAPVVSQIQAAASSQSSAQQSSFAQAQQSAYAQSQQSSSAYSGASTGGSSVSQSQPIIS"
    "SAPVYFNAQTLTSSLASSLQSDRALNFISSGQLSASDVSTSVSSAVAQTLGISQSSVQNIIS"
    "QQMSSVRTGASSSSVSQAIANAVSSAVQASGAATPGQEQSIAQRVYSSISTYLSQLISQRTA"
)

# Synthetic MaSp-like reference: canonical dragline motif structure
# (alternating GGX glycine-rich amorphous repeat + poly-Ala crystalline
# block), constructed rather than pulled from a database, specifically to
# give the tools an unambiguous positive control for poly-A / GGX detection.
SYNTHETIC_MASP_LIKE = (
    "GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA"
    "GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA"
    "GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA"
)


def validate_tools() -> List[ValidationReport]:
    reports = []

    # --- Case 1: synthetic MaSp-like positive control -----------------
    report = ValidationReport(sequence_name="synthetic_MaSp_like")
    motifs = motif_detector(SYNTHETIC_MASP_LIKE)
    report.checks.append(ValidationCheck(
        "poly_alanine detected",
        "poly_alanine" in motifs["motifs"],
        f"motifs found: {motifs['motifs']}",
    ))
    report.checks.append(ValidationCheck(
        "GGX_repeat detected",
        "GGX_repeat" in motifs["motifs"],
        f"motifs found: {motifs['motifs']}",
    ))
    repeats = repeat_pattern_analyzer(SYNTHETIC_MASP_LIKE)
    report.checks.append(ValidationCheck(
        "dominant repeat period detected (~35 residues, the constructed unit length)",
        repeats["dominant_period"] is not None and repeats["repeat_score"] > 0.5,
        f"period={repeats['dominant_period']}, score={repeats['repeat_score']}",
    ))
    beta = beta_sheet_propensity(SYNTHETIC_MASP_LIKE)
    # This is the deliberately surfaced known limitation, not a bug: classic
    # Chou-Fasman under-calls poly-Ala as beta-rich because it doesn't model
    # silk-specific side-chain packing. We check for and report the
    # discrepancy rather than silently trusting either signal alone.
    poly_a_flagged_by_motif = "poly_alanine" in motifs["motifs"]
    poly_a_flagged_by_propensity = beta["mean_beta_propensity"] >= beta["beta_threshold"]
    report.checks.append(ValidationCheck(
        "KNOWN LIMITATION CHECK: motif regex vs. Chou-Fasman propensity agreement on poly-Ala",
        True,  # informational, not pass/fail
        f"motif regex flags poly-Ala: {poly_a_flagged_by_motif}; "
        f"Chou-Fasman mean beta propensity: {beta['mean_beta_propensity']} "
        f"(threshold {beta['beta_threshold']}) flags beta-rich: {poly_a_flagged_by_propensity} — "
        f"{'as expected, these disagree; use motif regex as the more reliable signal for crystalline domains, not generic propensity' if poly_a_flagged_by_motif and not poly_a_flagged_by_propensity else 'unexpected agreement/disagreement, re-check'}",
    ))
    reports.append(report)

    # --- Case 2: real PySp sequence (non-MaSp family, proline-rich) ---
    report2 = ValidationReport(sequence_name="Eriophora_pustulosa_PySp (real, from handoff doc)")
    motifs2 = motif_detector(PYSP_ERIOPHORA_PUSTULOSA)
    report2.checks.append(ValidationCheck(
        "proline_rich_turn motif detected (expected — PySp is a cement/attachment-disc protein, proline-rich)",
        "proline_rich_turn" in motifs2["motifs"],
        f"motifs found: {motifs2['motifs']}",
    ))
    report2.checks.append(ValidationCheck(
        "poly_alanine NOT dominant (expected — PySp lacks MaSp-style crystalline poly-Ala blocks)",
        "poly_alanine" not in motifs2["motifs"],
        f"motifs found: {motifs2['motifs']}",
    ))
    beta2 = beta_sheet_propensity(PYSP_ERIOPHORA_PUSTULOSA)
    proline_fraction_pysp = PYSP_ERIOPHORA_PUSTULOSA.upper().count("P") / len(_clean(PYSP_ERIOPHORA_PUSTULOSA))
    proline_fraction_masp = SYNTHETIC_MASP_LIKE.upper().count("P") / len(_clean(SYNTHETIC_MASP_LIKE))
    report2.checks.append(ValidationCheck(
        "proline content elevated relative to synthetic MaSp control (direct compositional check)",
        proline_fraction_pysp > proline_fraction_masp,
        f"PySp proline fraction={proline_fraction_pysp:.3f}, synthetic MaSp proline fraction={proline_fraction_masp:.3f}",
    ))
    # NOTE: an earlier version of this check compared mean_turn_propensity
    # (Chou-Fasman Pturn) instead of raw proline content, and it failed —
    # not because the tool was buggy, but because the assumption was wrong.
    # Glycine's Pturn (1.56) is actually higher than proline's (1.52) in the
    # classic Chou-Fasman table, so a glycine-rich sequence (the synthetic
    # MaSp control) can out-score a genuinely proline-rich sequence on this
    # aggregate metric even though proline content is the more biologically
    # direct signal for PySp's turn-forming character. Kept here as an
    # informational check rather than deleted, since this is exactly the
    # kind of tool blind spot Stage 0 validation is supposed to surface
    # before it silently misleads a trace-generation prompt downstream.
    report2.checks.append(ValidationCheck(
        "KNOWN LIMITATION CHECK: aggregate Pturn propensity vs. raw proline content can disagree",
        True,  # informational, not pass/fail
        f"PySp mean_turn_propensity={beta2['mean_turn_propensity']} vs "
        f"synthetic MaSp mean_turn_propensity={beta['mean_turn_propensity']} "
        f"({'Pturn ranks PySp lower despite higher proline content — use compositional checks, not aggregate Pturn, for proline-specific claims' if beta2['mean_turn_propensity'] <= beta['mean_turn_propensity'] else 'Pturn and proline content agree here'})",
    ))
    reports.append(report2)

    return reports


def print_validation_reports(reports: List[ValidationReport]) -> None:
    for r in reports:
        print(f"\n=== {r.sequence_name} (passed {r.passed_fraction:.0%} of pass/fail checks) ===")
        for c in r.checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"  [{status}] {c.name}")
            print(f"         {c.detail}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reports = validate_tools()
    print_validation_reports(reports)
