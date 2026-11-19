# Aim 2 — Self-Assembling Peptide Network Mechanics

> **Specific Aim 2:** To determine whether an agent can predict the mechanical
> properties of self-assembling peptide networks by composition rather than
> regression.

The hypothesis, from the proposal: *network stiffness can be predicted by
decomposition into independently checkable sub-questions, and this is feasible
here specifically because the intermediate quantities — fiber diameter and
cross-link density — are themselves measurable rather than merely computed.*

The decomposition (Figure 5):

```
sequence → descriptors → coarse-grained assembly ─┬→ gelation      (solved; positive control, 87% hit rate)
                                                  └→ fiber morphology → network modulus   (the open problem)
```

## Why this directory starts with evaluation, not with the agent

Section C.6 is the governing result for this aim. The silk pipeline produced
traces that were **99.4% grounded** across 5,259 checked claims, and those same
traces carried **no predictive signal** — Spearman −0.06 on strength and −0.40
on toughness, against +0.57 and +0.98 for chain length alone. The pipeline's own
correctness criterion was satisfied by a constant guess **93%** of the time
versus **8%** for the tool-grounded traces.

Nothing in a grounding score surfaced that. Nothing in a bare accuracy number
would have either. The failure was in the reporting path: no comparison was
structurally required, so none was made until late.

So the first thing built for Aim 2 is the floor that makes the C.6 outcome
impossible to report as a success, plus the feasibility check for the aim's own
stated top risk ("the mechanical data may be too sparse or too heterogeneously
measured to support a holdout"). Both are answerable before any agent exists.

## Modules

Each is dependency-free and runnable standalone, per the repo convention:

```bash
python3 aim2_peptide_networks/dataset_schema.py
python3 aim2_peptide_networks/identity_holdout.py
python3 aim2_peptide_networks/baselines.py
```

### `dataset_schema.py` — canonical records + heterogeneity audit

`PeptideNetworkRecord` is one measurement of one peptide under one set of
conditions. A peptide measured at three concentrations is three records.

Measurement conditions are first-class, not metadata. A storage modulus is not a
property of a peptide — it is a property of a peptide at a concentration, pH,
ionic strength and temperature, measured at a given angular frequency and strain
amplitude. Supramolecular gel moduli routinely vary by an order of magnitude
across those axes, so a corpus that pools G′ across papers without recording
them has label noise that can exceed its signal.

`heterogeneity_report()` answers the feasibility question directly: how many
records carry a modulus, how many are fully protocol-specified, how many
*comparable blocks* exist, and how many unique peptides sit in the largest one.
It ends in a blunt go / marginal / no-go verdict, so the sparsity risk is
resolved before modelling rather than during it.

It also computes peptide molecular weight from sequence plus terminal
modifications (Fmoc-, Nap-, Ac-, -NH₂, …), which is deterministic and
hand-checkable — the Fmoc-FF self-check pins it against the known ~534.6 Da.

### `identity_holdout.py` — the split the proposal requires

The proposal specifies a holdout "constructed by sequence identity rather than
by random partition", and that requirement is load-bearing. Peptide gelator
libraries are dense with near-duplicates: single-residue variants, scrambled
controls, the same core with a different cap. Under a random split a test
peptide's near-twin sits in training, and the leak is invisible in the metrics.

The guarantee here is explicit: **no held-out peptide exceeds the threshold
identity to any training peptide.** Single-linkage clustering at the threshold
makes that true by construction (single linkage is the correct choice precisely
because "different clusters" then implies "no pair above threshold"), and
`verify_partition()` re-checks it from scratch against the final split anyway.

Identity is `1 − levenshtein(a, b) / max(len(a), len(b))`, which reduces to
percent identity for equal-length peptides and handles the indels these
libraries contain. Terminal modifications are treated conservatively: Fmoc-FF
and FF are separate entries (different molecules, different assembly) but are
forced into the same cluster so they cannot straddle the split.

All-pairs edit distance is pruned by a sound composition bound,
`identity(a,b) ≤ |multiset ∩| / max(|a|,|b|)`, which is an upper bound and so
never discards a pair that would have been linked. On the demo library it skips
98 of 105 pairs before any DP runs.

The module's self-check contrasts the two splits on the same library: the
identity split reaches max cross-partition identity 0.286, while a random split
leaks pairs at 0.71, 0.86 and 1.00.

### `baselines.py` — mandatory comparison + criterion audit

- `evaluate()` refuses to report a bare score. Every report carries baselines
  beside the model, and *names* any required baseline that was not supplied. The
  proposal commits to two simultaneously — a trivial predictor and the best
  available sequence-embedding model — so an absent embedding baseline prints
  `!! REQUIRED BASELINE NOT SUPPLIED` rather than being quietly omitted.
- `SingleCovariateBaseline` is first-class, because in silk the thing that
  embarrassed the pipeline was a one-variable baseline nobody had run. For
  peptide networks the analogue is concentration or peptide length. It fits in
  log₁₀ space by default, since published moduli span decades.
- `criterion_informativeness()` generalises Figure 4c into a reusable guard.
  Given a tolerance criterion, it computes the hit rate of the **best possible
  constant** (exact interval-overlap sweep, not just the mean — the honest
  adversary). If a constant matches the model, the criterion is flagged
  UNINFORMATIVE and must not be used as a preference or reward signal.

The self-check reconstructs the C.6 failure shape synthetically and confirms all
three guards fire on it, then shows a genuinely predictive model passing.

## Preliminary analysis on real public data

A real dataset is now wired in — but it serves **link 2 of the chain, not the
open problem.**

**Source:** Liu et al., *Briefings in Bioinformatics* 24(6), bbad409 (2023),
data file `Sequential_Peptides/collection_of_peptide_data.csv` from
[Zihan-Liu-00/DL_for_Peptide](https://github.com/Zihan-Liu-00/DL_for_Peptide).
62,159 peptides of length 5–10, each labelled with coarse-grained
aggregation propensity (AP). AP range 0.993–2.619; 8.8% exceed the classical
AP > 2 self-assembly cutoff.

**The label is simulation output, not measurement.** A model that predicts AP
is emulating a Martini/GROMACS coarse-grained simulator. That is worth
knowing — it is the chain's intermediate — but it is not evidence about
network modulus, and no claim about experimental behaviour should rest on it.
The data is fetched, not committed (`.gitignore`); `ap_dataset.py` prints the
exact `curl` command if it is missing.

### Finding 1 — the library percolates at length 5, and nowhere else

Exact per-length identity-cluster structure (masked-bucket clustering; a
*lower bound* on linkage, since indels can only merge clusters further):

| len | n | clusters @ 0.6 | largest | % in largest |
|---|---|---|---|---|
| 5 | 10,450 | **1** | **10,450** | **100%** |
| 6 | 9,713 | 5,739 | 65 | 0.7% |
| 7 | 9,644 | 9,333 | 4 | 0.0% |
| 8 | 9,892 | 9,036 | 5 | 0.1% |
| 9 | 9,656 | 9,614 | 2 | 0.0% |
| 10 | 12,804 | 12,598 | 3 | 0.0% |

Every pentapeptide in the corpus collapses into a **single** identity cluster,
while lengths 6–10 are essentially all singletons. This is the combinatorial
percolation effect: at length 5 each peptide has C(5,2)·19² = 3,610 neighbours
within 2 substitutions, and at this sampling density that is far above the
connectivity threshold. At length 10 the same calculation falls off a cliff.

**Consequence:** an identity holdout cannot divide the pentapeptides at all —
they must be assigned whole. Lengths 6–10 have so little near-duplicate
structure that an identity split and a random split are nearly the same thing.
The split discipline matters exactly where density is high, and here that is
one length class.

### Finding 2 — descriptors predict AP well, and beat both baselines

Ridge regression on 29 `bio_tools` descriptors (hydropathy, charge,
Chou-Fasman propensities, composition), 80/20 split:

| split | predictor | ρ | MAE | RMSE |
|---|---|---|---|---|
| random | **ridge (descriptors)** | **0.910** | **0.0716** | 0.0939 |
| random | trivial | n/a | 0.1617 | 0.2138 |
| random | hydropathy alone | 0.346 | 0.1521 | 0.1991 |
| identity-0.60 | **ridge (descriptors)** | **0.907** | **0.0652** | 0.0845 |
| identity-0.60 | trivial | n/a | 0.1433 | 0.1890 |
| identity-0.60 | hydropathy alone | 0.363 | 0.1344 | 0.1754 |

The model beats every supplied baseline on both MAE and ρ, and the
correctness criterion passes the constant-guess audit (75.2% model vs 41.8%
for the best possible constant) — i.e. **this criterion is informative**,
unlike the silk one. Mean hydropathy dominates the fitted weights (1.01
standardised, next largest 0.33 for arginine fraction), which is the
physically expected answer and a useful sanity check on the descriptors.

Per-length accuracy is flat (ρ 0.92–0.94 across lengths 5–10), so the model is
not quietly working on one subpopulation only.

### Finding 3 — the two splits are not comparable, and the module says so

The identity split forces all 10,450 pentapeptides into train, so the
held-out set contains **zero** length-5 peptides. The random and identity
numbers above therefore describe different populations, and `ap_model.py`
prints an explicit warning rather than letting them sit side by side as if
they were a clean comparison. Read each against its own baselines.

### What this does and does not license

It establishes that the descriptor → AP link is strongly learnable, cheaply,
and that a composed chain calling a coarse-grained simulator for this step has
a real bar to clear: ρ ≈ 0.91 from a linear fit on nine physical descriptors
plus composition. It says nothing yet about fiber morphology or modulus, which
is where the aim's actual difficulty lives.

## Experimental data: Xu et al. 2023 (`xu2023_gelation.py`)

The first **experimentally measured** dataset in the project. 165 synthesised
tetrapeptides from [Nature Communications 14, 3880
(2023)](https://doi.org/10.1038/s41467-023-39648-2), Supplementary Data 7:
observed state (gel 100 / sol 30 / suspension 28 / precipitate 7), TEM
morphology (13 classes), recorded concentration, and the authors' computed
APH/APHC ranks. This covers the chain's **gelation** node (the positive
control) and its **fiber morphology** node. It carries no modulus.

### Finding 4 — concentration is label-determined, and it is a trap

| state | 30 mM | 60 mM | 90 mM | 120 mM |
|---|---|---|---|---|
| gel | 46 | 30 | 6 | 18 |
| non-gel | 0 | 0 | 0 | 65 |

The protocol escalates: test at 30 mM, record a gel if one forms, otherwise
retry higher, and record the final state at 120 mM. So `concentration < 120`
implies gel with **perfect precision — 82 of 82, zero false positives** — and
reaches 89.1% accuracy against a 60.6% majority baseline, using a column that
only has its value because the outcome was already known.

Handing it to the model does exactly what you'd expect:

| features | accuracy | balanced | AUC |
|---|---|---|---|
| descriptors only | 0.697 | 0.680 | 0.703 |
| **+ concentration (leaky)** | 0.806 | 0.800 | **0.902** |

`protocol_escalation_report()` detects this generically — it tests for
*one-sided* precision rather than high accuracy, because a genuinely
predictive covariate errs in both directions while a label-determined one
cannot. This matters beyond this dataset: escalation is standard practice in
the gelation literature, so any corpus compiled per `dataset_schema.py` will
likely contain it, and it qualifies that module's treatment of concentration
as "a covariate the chain should reason about" — right for an independently
set concentration, badly wrong for an escalated one.

### Finding 5 — descriptors are much weaker on experiment than on simulation

Identity-grouped 5-fold out-of-fold, logistic regression on the same 29
descriptors:

| target | model | AUC | balanced acc |
|---|---|---|---|
| gelation | logistic (descriptors) | 0.703 | 0.680 |
| gelation | majority | 0.500 | 0.500 |
| gelation | authors' APHC rank | 0.926 | 0.821 |
| fibrous morphology | logistic (descriptors) | 0.568 | 0.544 |
| fibrous morphology | majority | 0.500 | 0.500 |
| fibrous morphology | authors' APHC rank | 0.753 | 0.690 |

Set against Finding 2 (ρ = 0.910 predicting simulated AP), this is the
informative contrast: **descriptors reproduce the simulator's output almost
perfectly but predict the experiment only modestly.** The coarse-grained step
is not redundant with cheap descriptors where it actually matters. Fibrous
morphology — the intermediate the proposal's feasibility argument leans on —
is the harder of the two, barely above chance from descriptors alone.

**The APHC comparison is not fair, and the module says so.** These 165
peptides were *selected* by that score, enriched at both ends of its ranking,
so its separation here is inflated by the selection and is an upper bound on
its performance on randomly drawn tetrapeptides. Treat the gap as an
unresolved comparison, not a measured defeat.

### Finding 6 — near-duplicate structure is present but costs little here

The 165 form 80 identity clusters at threshold 0.6, with the largest holding
56 peptides (34% of the set) — they are single-substitution families around
top-ranked sequences, not a random sample. Yet random folds inflate AUC by
only **+0.016** over identity-grouped folds. The structural leak is real; for
a linear model it barely bites. A higher-capacity model that can memorise
families would likely show a much larger gap, so the grouping stays.

## Literature-mined phase data (`peptideminer_phases.py`)

The most condition-rich experimental corpus found, and the closest thing yet
to what `dataset_schema.py` was written to audit: 1,017 observations pooled
from many publications, each with peptide, terminal chemistry, solvent,
concentration, pH, temperature, incubation time and resulting phase.

**Source:** Yang, Yorke, Knowles & Buehler, [arXiv:2411.05421](https://arxiv.org/abs/2411.05421),
`data/phase_data_clean.csv` from [lamm-mit/PeptideMiner](https://github.com/lamm-mit/PeptideMiner).
Phases: none 407, hydrogel 219, tube 121, fiber 96, vesicle 43, nanosphere 33,
ribbon 33, particle 16, organogel 5. Still **no modulus**.

### Finding 7 — row counts are not sample sizes, and here the gap is fatal

1,017 rows, **119 unique sequences**, ~8.5 rows per sequence. Worse, those
119 are di- and tripeptides, which percolate at identity 0.6 exactly as the
pentapeptides did: **23 clusters, the largest holding 90 (76%)**. The
effective sample size is ~23 groups dominated by one — not 1,017.

### Finding 8 — the leakage this causes, measured

Same model, same data, same folds count; only the grouping changes:

| target | row-level folds (leaky) | identity-grouped | inflation |
|---|---|---|---|
| hydrogel | AUC 0.840 | AUC 0.631 | **+0.209** |
| fiber | AUC 0.857 | **AUC 0.500** | **+0.357** |

The fiber row is the one to look at. Split by row, the model scores 0.857 and
looks strong. Split by sequence identity, it scores **exactly chance**. It had
learned which sequences form fibers, not what makes a sequence form them —
and nothing in the row-level number reveals that.

Set against Finding 6 (the Xu data, where identity grouping cost only 0.016
AUC), this is the calibration: near-duplicate structure is cheap when a
corpus samples sequence space sparsely and ruinous when it re-measures a few
sequences many times. You cannot tell which case you are in from the row
count — only from the cluster count.

Sequence descriptors alone reach AUC 0.456 for hydrogel (below chance) and
0.537 for fiber, so what little signal the grouped model has comes from the
experimental conditions, not the sequence.

### Finding 9 — concentration is *not* confounded here

The same `protocol_escalation_report()` that fired at precision 1.000 on the
Xu data reports 0.272 here and declines to flag it. Pooling many independent
protocols breaks the escalation pattern: concentration is a genuine
experimental variable in a multi-source corpus and a label proxy in a
single-protocol one. The detector distinguishes them without being told which
is which.

### Data-quality flags

Curated partly by LLM literature mining, so extraction errors are expected:
44 rows carry non-standard residue letters and 6 have temperatures outside
0–100 °C (the raw column spans −80 to 200 °C — plainly storage and
lyophilisation conditions attached to the assembly step). `quality_flags()`
reports both rather than silently dropping them.

## Where the modulus data actually is — and isn't

Searched and came up empty, which is itself a result worth recording:

- **Simpson et al. 2025** ([10.1002/adma.202415031](https://doi.org/10.1002/adma.202415031),
  the proposal's ref 18) is the single most on-point paper — Bayesian models
  predicting G′ and G″ for di/tripeptide gels. Open access on PMC, but PMC
  serves a CAPTCHA to automated requests and Wiley is outside the session's
  network allowlist. **Not retrieved.**
- **Buehler group, arXiv:2505.08850** computes Young's modulus for *all* 400
  dipeptides and *all* 8,000 tripeptides by coarse-grained MD with AFM
  validation. Its accession code points at `github.com/lamm-mit/PeptideModulus`,
  which **404s on every branch** — cited but never made public.
- **Leeds 10.5518/975** (peptide:GAG rheology) and **Montazeri et al.**
  (70 formulations, frequency-dependent moduli) are both on hosts outside the
  allowlist.
- Zenodo and figshare searches for sequence-varied peptide rheology returned
  only single-system studies: one peptide characterised deeply, which is the
  literature's normal shape and the opposite of what a sequence→modulus model
  needs.

**The pattern across every corpus reached: gelation and morphology are
available; modulus is not.** Three independent datasets now cover the chain's
upper branch and its morphology node, and none carries the mechanical
property Aim 2 is actually about.

## What is deliberately *not* here

- **No embedding baseline implementation.** It needs a model this repo does not
  carry. It is a named required slot that reports as missing — not a stub
  returning zeros, which would make an incomplete report look complete.
- **No imputation of missing conditions, and no frequency-harmonisation of
  moduli.** Both are modelling decisions that should be explicit and visible,
  not buried in a loader.
- **No candidate sequence generation.** The invertible-decomposition goal
  ("propose sequences meeting a target stiffness") is downstream of a chain that
  demonstrably predicts, and per the proposal's own logic there is nothing to
  invert until Aim 2's forward direction beats its baselines.

## Next steps, in order

1. **Compile the corpus and run the audit.** Published gelation outcomes plus
   the corresponding rheology, largely in supplementary materials — the
   proposal's reference list points at Li 2019 (PNAS), Xu 2023 (Nat Commun) and
   Simpson 2025 (Adv Mater) as the starting points. Run
   `heterogeneity_report()` and get a go / marginal / no-go verdict. If marginal,
   the proposal's first alternative (narrow to one peptide family with
   consistent measurement conditions, and state the reduced generality) applies.
2. **Fix the identity threshold and freeze the split** before any modelling,
   recording the cluster count — not the peptide count — as the effective
   sample size.
3. **Extend the coarse-grained engine** to these assembly regimes with
   morphological observables (fiber diameter, cross-link density) that can be
   checked directly against microscopy, keeping the existing validation harness.
4. **Build the agent** as a tool-calling loop that selects and parameterises
   tools but supplies no numerical values of its own, with the final narrative
   audited claim by claim against tool output (`grounding_checker.py`).
5. **Report faithfulness and accuracy as two independent quantities**, which is
   the direct lesson of C.6.
