"""
Aim 2 Dataset Schema + Heterogeneity Audit
=============================================

Aim 2 asks whether an agent can predict the *network modulus* of a
self-assembling peptide hydrogel by composing checkable sub-questions
(sequence descriptors -> assembly propensity -> fiber morphology ->
modulus) rather than regressing sequence directly onto the answer.

The proposal names its own top risk for this aim plainly:

    "the mechanical data may be too sparse or too heterogeneously measured
     to support a holdout"

That risk is not a thing to discover halfway through building an agent. It
is answerable *now*, from the data, and this module exists to answer it
before any modelling work is done. It does two jobs:

  1. `PeptideNetworkRecord` — one canonical record per (peptide, condition,
     measurement), with units normalised to SI and measurement conditions
     treated as first-class required fields rather than optional metadata.

  2. `heterogeneity_report()` — an audit that reports, for a candidate
     compiled dataset, whether a comparable subset large enough to train
     and hold out actually exists.

Why measurement conditions are mandatory, not metadata
--------------------------------------------------------
A storage modulus G' is not a property of a peptide. It is a property of a
peptide *at a concentration, pH, ionic strength and temperature, measured at
a particular angular frequency and strain amplitude, in a particular
geometry, after a particular time*. Supramolecular gels are shear-history
dependent and their moduli routinely vary by an order of magnitude across
those axes. A dataset that pools G' across papers without recording them
is a dataset whose label noise can exceed its signal — and per C.6 of the
proposal, a pipeline that cannot beat a trivial baseline is exactly what
that produces.

So this schema refuses to let conditions be silently absent. They can be
`None`, but `heterogeneity_report()` counts every `None` and tells you what
fraction of the corpus is unusable for a like-for-like comparison.

Scope note
------------
This module deliberately does NOT convert between concentration units that
require information the record does not carry, does not impute missing
conditions, and does not harmonise moduli measured at different frequencies
via any power-law assumption. Each of those is a modelling decision that
should be made explicitly and visibly, not buried in a loader. The one
derived quantity computed here is peptide molecular weight, because that is
a deterministic function of the sequence and the recorded terminal
modifications and is checkable by hand.

Runnable standalone: `python3 aim2_peptide_networks/dataset_schema.py`
"""

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Average (not monoisotopic) residue masses in Da, i.e. the mass each residue
# contributes to a polypeptide chain. Peptide mass = sum(residues) + water.
AVERAGE_RESIDUE_MASS = {
    "G": 57.0519, "A": 71.0788, "S": 87.0782, "P": 97.1167, "V": 99.1326,
    "T": 101.1051, "C": 103.1388, "L": 113.1594, "I": 113.1594,
    "N": 114.1038, "D": 115.0886, "Q": 128.1307, "K": 128.1741,
    "E": 129.1155, "M": 131.1926, "H": 137.1411, "F": 147.1766,
    "R": 156.1875, "Y": 163.1760, "W": 186.2132,
}

WATER_MASS = 18.0153

# Mass deltas for the terminal modifications that dominate the self-assembling
# peptide literature. Fmoc- and naphthalene-capped dipeptides are a large
# fraction of the published gelation corpus, and a C-terminal amide is close
# to standard for designer scaffold peptides, so getting these wrong would
# bias molecular weight (and hence any mM <-> wt% reasoning) systematically.
#
# Values are the net mass change relative to the unmodified free-amine /
# free-acid terminus.
N_TERM_MODIFICATIONS = {
    "H": 0.0,              # free amine (unmodified)
    "AC": 42.0367,         # acetyl
    "FMOC": 222.2400,      # fluorenylmethyloxycarbonyl
    "NAP": 168.1900,       # naphthalene-2-yl acetyl ("Nap-")
    "CBZ": 134.1300,       # benzyloxycarbonyl
    "BOC": 100.1200,       # tert-butyloxycarbonyl
}

C_TERM_MODIFICATIONS = {
    "OH": 0.0,             # free acid (unmodified)
    "NH2": -0.9847,        # primary amide (-OH -> -NH2)
    "OME": 14.0266,        # methyl ester
}

# Modulus units -> multiplier to Pascals.
MODULUS_UNITS_TO_PA = {
    "PA": 1.0,
    "KPA": 1e3,
    "MPA": 1e6,
    "GPA": 1e9,
    "DYN/CM2": 0.1,
    "DYN/CM^2": 0.1,
}

# Length units -> multiplier to nanometres.
LENGTH_UNITS_TO_NM = {
    "NM": 1.0,
    "A": 0.1,
    "ANGSTROM": 0.1,
    "UM": 1e3,
    "ΜM": 1e3,
    "MICRON": 1e3,
    "MM": 1e6,
}

VALID_RESIDUES = set(AVERAGE_RESIDUE_MASS)


class SchemaError(ValueError):
    """Raised for a record that cannot be interpreted at all.

    Deliberately distinct from the soft `issues` a record can carry. A
    SchemaError means the record is malformed (bad residue, negative
    modulus, unparseable unit) and must be fixed or dropped at ingest. An
    `issue` means the record is interpretable but incomplete or internally
    odd, which is a fact about the corpus that the audit should count
    rather than a reason to throw the row away.
    """


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------

def _normalise_unit(unit: str) -> str:
    return re.sub(r"\s+", "", str(unit)).upper()


def to_pascals(value: float, unit: str) -> float:
    """Convert a modulus to Pa. Raises SchemaError on an unknown unit."""
    key = _normalise_unit(unit)
    if key not in MODULUS_UNITS_TO_PA:
        raise SchemaError(
            f"unknown modulus unit {unit!r}; known: {sorted(MODULUS_UNITS_TO_PA)}"
        )
    return float(value) * MODULUS_UNITS_TO_PA[key]


def to_nanometres(value: float, unit: str) -> float:
    """Convert a length to nm. Raises SchemaError on an unknown unit."""
    key = _normalise_unit(unit)
    if key not in LENGTH_UNITS_TO_NM:
        raise SchemaError(
            f"unknown length unit {unit!r}; known: {sorted(LENGTH_UNITS_TO_NM)}"
        )
    return float(value) * LENGTH_UNITS_TO_NM[key]


def peptide_molecular_weight(
    sequence: str,
    n_term: str = "H",
    c_term: str = "OH",
) -> float:
    """Average molecular weight in Da from sequence + terminal modifications.

    Deterministic and hand-checkable: sum of average residue masses, plus
    one water, plus the two terminal deltas. Raises SchemaError rather than
    guessing if a residue or modification is unrecognised — a silently wrong
    molecular weight would propagate into every wt% <-> mM comparison in the
    corpus.
    """
    seq = str(sequence).strip().upper()
    if not seq:
        raise SchemaError("empty sequence has no molecular weight")

    bad = sorted(set(seq) - VALID_RESIDUES)
    if bad:
        raise SchemaError(f"sequence contains non-standard residues: {bad}")

    n_key = _normalise_unit(n_term)
    c_key = _normalise_unit(c_term)
    if n_key not in N_TERM_MODIFICATIONS:
        raise SchemaError(
            f"unknown N-terminal modification {n_term!r}; "
            f"known: {sorted(N_TERM_MODIFICATIONS)}"
        )
    if c_key not in C_TERM_MODIFICATIONS:
        raise SchemaError(
            f"unknown C-terminal modification {c_term!r}; "
            f"known: {sorted(C_TERM_MODIFICATIONS)}"
        )

    mass = sum(AVERAGE_RESIDUE_MASS[r] for r in seq) + WATER_MASS
    return mass + N_TERM_MODIFICATIONS[n_key] + C_TERM_MODIFICATIONS[c_key]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class PeptideNetworkRecord:
    """One measurement of one peptide under one set of conditions.

    A peptide measured at three concentrations is three records, not one
    record with three moduli. That keeps concentration a covariate the agent
    must reason about rather than an ambiguity hidden inside a row.

    All stored numeric fields are in SI-ish canonical units (Pa, nm, mM,
    degrees Celsius, rad/s, percent strain). Use `from_reported()` to build
    a record from literature-reported values in their original units.
    """

    # --- identity -------------------------------------------------------
    record_id: str
    sequence: str
    n_term: str = "H"
    c_term: str = "OH"

    # --- conditions (the reason a modulus means anything) ---------------
    concentration_mM: Optional[float] = None
    concentration_wt_pct: Optional[float] = None
    pH: Optional[float] = None
    ionic_strength_mM: Optional[float] = None
    temperature_C: Optional[float] = None

    # --- measurement protocol -------------------------------------------
    angular_frequency_rad_s: Optional[float] = None
    strain_pct: Optional[float] = None
    geometry: Optional[str] = None          # e.g. "parallel plate 20mm"
    equilibration_min: Optional[float] = None

    # --- the chain's observables (Figure 5 of the proposal) -------------
    gelation: Optional[bool] = None             # upper branch, positive control
    aggregation_propensity: Optional[float] = None   # from CG simulation
    fiber_diameter_nm: Optional[float] = None        # checkable vs microscopy
    crosslink_density_per_um3: Optional[float] = None
    storage_modulus_Pa: Optional[float] = None       # the open problem
    loss_modulus_Pa: Optional[float] = None

    # --- provenance ------------------------------------------------------
    source_doi: Optional[str] = None
    source_locator: Optional[str] = None    # e.g. "Table S3 row 12", "Fig 2b"
    notes: Optional[str] = None

    # --- derived / audit --------------------------------------------------
    issues: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sequence = str(self.sequence).strip().upper()
        bad = sorted(set(self.sequence) - VALID_RESIDUES)
        if bad:
            raise SchemaError(
                f"{self.record_id}: non-standard residues {bad} in "
                f"{self.sequence!r}"
            )
        if not self.sequence:
            raise SchemaError(f"{self.record_id}: empty sequence")

        for name in ("storage_modulus_Pa", "loss_modulus_Pa",
                     "fiber_diameter_nm", "crosslink_density_per_um3",
                     "concentration_mM", "concentration_wt_pct",
                     "ionic_strength_mM", "angular_frequency_rad_s"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise SchemaError(f"{self.record_id}: {name} is negative ({value})")

        self.issues = self._audit()

    # -- derived ----------------------------------------------------------

    @property
    def molecular_weight(self) -> float:
        return peptide_molecular_weight(self.sequence, self.n_term, self.c_term)

    @property
    def display_sequence(self) -> str:
        """Sequence with terminal modifications, e.g. 'Fmoc-FF-OH'."""
        n = "" if _normalise_unit(self.n_term) == "H" else f"{self.n_term}-"
        c = "" if _normalise_unit(self.c_term) == "OH" else f"-{self.c_term}"
        return f"{n}{self.sequence}{c}"

    @property
    def identity_key(self) -> str:
        """What counts as 'the same peptide' for holdout purposes.

        Terminal modifications are included: Fmoc-FF and FF are chemically
        different molecules with different assembly behaviour, and treating
        them as one peptide would leak between partitions.
        """
        return f"{_normalise_unit(self.n_term)}|{self.sequence}|{_normalise_unit(self.c_term)}"

    def derived_concentration_mM(self) -> Optional[float]:
        """mM implied by wt%, via molecular weight.

        Assumes 1 g/mL solution density, which is the standard approximation
        for dilute aqueous peptide solutions and is stated here rather than
        hidden. Returns None if wt% is absent — never fabricates a value.
        """
        if self.concentration_wt_pct is None:
            return None
        grams_per_litre = 10.0 * self.concentration_wt_pct
        return 1000.0 * grams_per_litre / self.molecular_weight

    # -- audit ------------------------------------------------------------

    def _audit(self) -> List[str]:
        """Soft issues: interpretable but incomplete or internally odd.

        These are counted by the corpus audit rather than raised, because
        how many records carry them IS the feasibility answer Aim 2 needs.
        """
        issues: List[str] = []

        has_modulus = self.storage_modulus_Pa is not None
        if has_modulus:
            if self.angular_frequency_rad_s is None:
                issues.append("modulus_without_frequency")
            if self.strain_pct is None:
                issues.append("modulus_without_strain")
            if self.concentration_mM is None and self.concentration_wt_pct is None:
                issues.append("modulus_without_concentration")
            if self.temperature_C is None:
                issues.append("modulus_without_temperature")
            if self.pH is None:
                issues.append("modulus_without_pH")

        # A gel is conventionally G' > G''. A record labelled as gelling whose
        # moduli say otherwise is either a mislabel, a transcription error, or
        # a measurement outside the linear regime. Any of those matters.
        if (self.gelation is True and has_modulus
                and self.loss_modulus_Pa is not None
                and self.storage_modulus_Pa <= self.loss_modulus_Pa):
            issues.append("gel_label_contradicts_moduli")

        if self.gelation is False and has_modulus and self.storage_modulus_Pa > 0:
            issues.append("nongel_label_with_modulus")

        if has_modulus and self.storage_modulus_Pa == 0:
            issues.append("zero_modulus")

        if self.source_doi is None:
            issues.append("no_provenance")

        if (self.concentration_mM is not None
                and self.concentration_wt_pct is not None):
            implied = self.derived_concentration_mM()
            if implied and implied > 0:
                ratio = self.concentration_mM / implied
                # A factor of two apart means one of the two reported numbers
                # is wrong, or the peptide's assumed modification set is wrong.
                if ratio > 2.0 or ratio < 0.5:
                    issues.append("concentration_units_inconsistent")

        return issues

    # -- construction ------------------------------------------------------

    @classmethod
    def from_reported(
        cls,
        record_id: str,
        sequence: str,
        *,
        storage_modulus: Optional[Tuple[float, str]] = None,
        loss_modulus: Optional[Tuple[float, str]] = None,
        fiber_diameter: Optional[Tuple[float, str]] = None,
        **kwargs: Any,
    ) -> "PeptideNetworkRecord":
        """Build a record from values in their originally reported units.

        Pass `(value, unit)` pairs, e.g. `storage_modulus=(3.2, "kPa")`.
        Unknown units raise rather than being silently dropped.
        """
        if storage_modulus is not None:
            kwargs["storage_modulus_Pa"] = to_pascals(*storage_modulus)
        if loss_modulus is not None:
            kwargs["loss_modulus_Pa"] = to_pascals(*loss_modulus)
        if fiber_diameter is not None:
            kwargs["fiber_diameter_nm"] = to_nanometres(*fiber_diameter)
        return cls(record_id=record_id, sequence=sequence, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Corpus audit — the actual feasibility answer
# ---------------------------------------------------------------------------

# Conditions that must match for two modulus measurements to be compared
# like for like. Concentration is deliberately NOT in this tuple: it is a
# covariate the chain should reason about (modulus scales strongly with it),
# not a nuisance to filter on. Frequency, strain, temperature and pH are
# protocol choices that shift the number without being part of the question.
COMPARABILITY_KEYS = (
    "angular_frequency_rad_s",
    "strain_pct",
    "temperature_C",
    "pH",
)


def _bucket(value: Optional[float], resolution: float) -> Optional[str]:
    """Round a condition onto a grid so near-identical protocols group."""
    if value is None:
        return None
    return f"{round(value / resolution) * resolution:g}"


_BUCKET_RESOLUTION = {
    "angular_frequency_rad_s": 1.0,
    "strain_pct": 0.1,
    "temperature_C": 1.0,
    "pH": 0.5,
}


def comparability_block(record: PeptideNetworkRecord) -> Optional[Tuple[str, ...]]:
    """The protocol block a record belongs to, or None if under-specified.

    None means the record cannot be placed in any block — which is the point:
    such records are not usable for a like-for-like modulus comparison, and
    counting them is how the sparsity risk gets quantified.
    """
    parts: List[str] = []
    for key in COMPARABILITY_KEYS:
        bucketed = _bucket(getattr(record, key), _BUCKET_RESOLUTION[key])
        if bucketed is None:
            return None
        parts.append(f"{key}={bucketed}")
    return tuple(parts)


@dataclass
class HeterogeneityReport:
    n_records: int
    n_unique_peptides: int
    field_coverage: Dict[str, float]
    issue_counts: Dict[str, int]
    n_with_modulus: int
    n_modulus_fully_specified: int
    n_blocks: int
    largest_block: Optional[Tuple[str, ...]]
    largest_block_size: int
    largest_block_unique_peptides: int
    conflicting_gelation_labels: List[str]
    modulus_span_decades: Optional[float]

    def summary(self) -> str:
        lines = [
            "Aim 2 dataset heterogeneity audit",
            "=" * 60,
            f"records                      : {self.n_records}",
            f"unique peptides              : {self.n_unique_peptides}",
            f"records with a modulus       : {self.n_with_modulus}",
            f"  of those, fully specified  : {self.n_modulus_fully_specified}",
            f"protocol blocks              : {self.n_blocks}",
            f"largest comparable block     : {self.largest_block_size} records "
            f"({self.largest_block_unique_peptides} unique peptides)",
        ]
        if self.largest_block:
            lines.append(f"  block                      : {', '.join(self.largest_block)}")
        if self.modulus_span_decades is not None:
            lines.append(
                f"modulus dynamic range        : {self.modulus_span_decades:.1f} decades"
            )
        if self.conflicting_gelation_labels:
            lines.append(
                f"peptides with contradictory gelation labels: "
                f"{len(self.conflicting_gelation_labels)}"
            )
            for key in self.conflicting_gelation_labels[:5]:
                lines.append(f"  - {key}")

        lines.append("")
        lines.append("field coverage (fraction non-null):")
        for name, frac in sorted(self.field_coverage.items(), key=lambda kv: kv[1]):
            bar = "#" * int(round(frac * 30))
            lines.append(f"  {name:<28} {frac:5.1%} |{bar}")

        if self.issue_counts:
            lines.append("")
            lines.append("issues:")
            for name, count in sorted(
                self.issue_counts.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {name:<36} {count}")

        lines.append("")
        lines.append(self.verdict())
        return "\n".join(lines)

    def verdict(self) -> str:
        """The blunt answer to 'can this corpus support a holdout?'

        Thresholds are deliberately crude and stated rather than tuned. The
        point is to force an explicit go/no-go before an agent is built on
        top of the corpus, not to be precise about where the line is.
        """
        if self.largest_block_unique_peptides >= 100:
            return (
                "VERDICT: a like-for-like block with >=100 unique peptides exists. "
                "An identity-based holdout is plausible; check the identity "
                "clustering next (identity_holdout.py)."
            )
        if self.largest_block_unique_peptides >= 30:
            return (
                "VERDICT: marginal. The largest comparable block has "
                f"{self.largest_block_unique_peptides} unique peptides. This may "
                "support a single-family holdout only — the proposal's first "
                "alternative approach (narrow to one peptide family with "
                "consistent measurement conditions) likely applies."
            )
        return (
            "VERDICT: insufficient as compiled. No comparable block reaches 30 "
            "unique peptides, which is the sparsity risk the proposal names. "
            "Either widen the compilation, relax the comparability keys "
            "explicitly (and record that you did), or fall back to the stated "
            "alternative approaches before building the agent."
        )


def heterogeneity_report(
    records: List[PeptideNetworkRecord],
) -> HeterogeneityReport:
    """Audit a compiled corpus for holdout feasibility."""
    n = len(records)
    if n == 0:
        raise SchemaError("cannot audit an empty corpus")

    tracked_fields = [
        "concentration_mM", "concentration_wt_pct", "pH", "ionic_strength_mM",
        "temperature_C", "angular_frequency_rad_s", "strain_pct", "geometry",
        "gelation", "fiber_diameter_nm", "crosslink_density_per_um3",
        "storage_modulus_Pa", "loss_modulus_Pa", "source_doi",
    ]
    coverage = {
        name: sum(1 for r in records if getattr(r, name) is not None) / n
        for name in tracked_fields
    }

    issue_counts: Dict[str, int] = {}
    for record in records:
        for issue in record.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    with_modulus = [r for r in records if r.storage_modulus_Pa is not None]

    blocks: Dict[Tuple[str, ...], List[PeptideNetworkRecord]] = {}
    for record in with_modulus:
        block = comparability_block(record)
        if block is None:
            continue
        blocks.setdefault(block, []).append(record)

    if blocks:
        largest_key = max(
            blocks, key=lambda k: len({r.identity_key for r in blocks[k]})
        )
        largest = blocks[largest_key]
        largest_unique = len({r.identity_key for r in largest})
    else:
        largest_key, largest, largest_unique = None, [], 0

    # Contradictory gelation labels for the same peptide. Concentration can
    # legitimately flip gelation, so only flag conflicts at a matching
    # concentration — otherwise this would fire on perfectly good data.
    by_peptide_conc: Dict[Tuple[str, Optional[str]], set] = {}
    for record in records:
        if record.gelation is None:
            continue
        conc = record.concentration_mM
        if conc is None:
            conc = record.derived_concentration_mM()
        key = (record.identity_key, _bucket(conc, 0.5))
        by_peptide_conc.setdefault(key, set()).add(record.gelation)
    conflicting = sorted(
        f"{key[0]} @ {key[1]} mM" for key, labels in by_peptide_conc.items()
        if len(labels) > 1
    )

    positive_moduli = [
        r.storage_modulus_Pa for r in with_modulus if r.storage_modulus_Pa > 0
    ]
    if len(positive_moduli) >= 2:
        span = math.log10(max(positive_moduli)) - math.log10(min(positive_moduli))
    else:
        span = None

    return HeterogeneityReport(
        n_records=n,
        n_unique_peptides=len({r.identity_key for r in records}),
        field_coverage=coverage,
        issue_counts=issue_counts,
        n_with_modulus=len(with_modulus),
        n_modulus_fully_specified=sum(
            1 for r in with_modulus if comparability_block(r) is not None
        ),
        n_blocks=len(blocks),
        largest_block=largest_key,
        largest_block_size=len(largest),
        largest_block_unique_peptides=largest_unique,
        conflicting_gelation_labels=conflicting,
        modulus_span_decades=span,
    )


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _demo_corpus() -> List[PeptideNetworkRecord]:
    """A small synthetic corpus exercising each audit path.

    Synthetic on purpose: this is a smoke test of the audit logic, not
    evidence about any real dataset. Per CLAUDE.md, the real validation
    comes when this is run against actual compiled literature records.
    """
    records = [
        # A well-specified block of four at matched protocol.
        PeptideNetworkRecord.from_reported(
            "r1", "FF", n_term="Fmoc", storage_modulus=(10.0, "kPa"),
            loss_modulus=(1.2, "kPa"), concentration_mM=20.0, pH=7.4,
            temperature_C=25.0, angular_frequency_rad_s=6.28, strain_pct=0.1,
            gelation=True, fiber_diameter=(18.0, "nm"),
            source_doi="10.0000/demo1",
        ),
        PeptideNetworkRecord.from_reported(
            "r2", "FL", n_term="Fmoc", storage_modulus=(2.0, "kPa"),
            loss_modulus=(0.4, "kPa"), concentration_mM=20.0, pH=7.4,
            temperature_C=25.0, angular_frequency_rad_s=6.28, strain_pct=0.1,
            gelation=True, fiber_diameter=(12.0, "nm"),
            source_doi="10.0000/demo1",
        ),
        PeptideNetworkRecord.from_reported(
            "r3", "YY", n_term="Fmoc", storage_modulus=(350.0, "Pa"),
            loss_modulus=(70.0, "Pa"), concentration_mM=20.0, pH=7.4,
            temperature_C=25.0, angular_frequency_rad_s=6.28, strain_pct=0.1,
            gelation=True, source_doi="10.0000/demo1",
        ),
        PeptideNetworkRecord.from_reported(
            "r4", "VV", n_term="Fmoc", storage_modulus=(40.0, "Pa"),
            loss_modulus=(60.0, "Pa"), concentration_mM=20.0, pH=7.4,
            temperature_C=25.0, angular_frequency_rad_s=6.28, strain_pct=0.1,
            gelation=True, source_doi="10.0000/demo1",
        ),  # -> gel_label_contradicts_moduli

        # Under-specified: a modulus with no protocol. Unusable for comparison.
        PeptideNetworkRecord.from_reported(
            "r5", "KLVFF", storage_modulus=(1.5, "kPa"), gelation=True,
            notes="frequency and strain not reported in source",
        ),  # -> several *_without_* issues, no_provenance

        # A different protocol block entirely.
        PeptideNetworkRecord.from_reported(
            "r6", "RADARADARADARADA", c_term="NH2",
            storage_modulus=(800.0, "Pa"), concentration_wt_pct=1.0, pH=7.0,
            temperature_C=37.0, angular_frequency_rad_s=1.0, strain_pct=1.0,
            gelation=True, source_doi="10.0000/demo2",
        ),

        # Gelation-only record (upper branch of Figure 5, no mechanics).
        PeptideNetworkRecord(
            record_id="r7", sequence="GG", gelation=False,
            concentration_mM=20.0, source_doi="10.0000/demo1",
        ),
        # Same peptide, same concentration, opposite label -> conflict.
        PeptideNetworkRecord(
            record_id="r8", sequence="GG", gelation=True,
            concentration_mM=20.0, source_doi="10.0000/demo3",
        ),
    ]
    return records


def _run_demo() -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    print("--- molecular weight (hand-checkable) ---")
    for seq, n_term, c_term in [("FF", "H", "OH"), ("FF", "Fmoc", "OH"),
                                ("RADARADARADARADA", "H", "NH2")]:
        mw = peptide_molecular_weight(seq, n_term, c_term)
        print(f"  {n_term}-{seq}-{c_term:<4} {mw:9.2f} Da")
    # Fmoc-FF is a canonical gelator; its formula weight is ~534.6 Da.
    fmoc_ff = peptide_molecular_weight("FF", "Fmoc", "OH")
    assert 530.0 < fmoc_ff < 540.0, f"Fmoc-FF mass looks wrong: {fmoc_ff}"
    print(f"  check: Fmoc-FF within the expected ~534.6 Da -> OK")

    print()
    print("--- unit normalisation ---")
    assert to_pascals(1.0, "kPa") == 1000.0
    assert to_pascals(1.0, "dyn/cm2") == 0.1
    assert to_nanometres(1.0, "um") == 1000.0
    print("  kPa/dyn-cm^-2/um conversions -> OK")
    try:
        to_pascals(1.0, "psi")
    except SchemaError as exc:
        print(f"  unknown unit rejected rather than guessed: {exc}")

    print()
    print("--- malformed records are rejected at ingest ---")
    for bad_kwargs, label in [
        (dict(record_id="bad1", sequence="FXF"), "non-standard residue"),
        (dict(record_id="bad2", sequence="FF", storage_modulus_Pa=-5.0),
         "negative modulus"),
    ]:
        try:
            PeptideNetworkRecord(**bad_kwargs)
        except SchemaError as exc:
            print(f"  {label}: rejected ({exc})")
        else:
            raise AssertionError(f"{label} should have raised")

    print()
    records = _demo_corpus()
    report = heterogeneity_report(records)
    print(report.summary())

    # Assertions on the demo corpus so this file is a real regression check.
    assert report.n_records == 8
    assert report.n_with_modulus == 6
    assert report.n_modulus_fully_specified == 5, report.n_modulus_fully_specified
    assert report.largest_block_unique_peptides == 4
    assert "gel_label_contradicts_moduli" in report.issue_counts
    assert "modulus_without_frequency" in report.issue_counts
    assert len(report.conflicting_gelation_labels) == 1
    print()
    print("All dataset_schema.py self-checks passed.")


if __name__ == "__main__":
    _run_demo()
