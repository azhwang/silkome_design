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
