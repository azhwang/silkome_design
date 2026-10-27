# Multi-chain silk simulation benchmark — design spec (draft)

**Status:** draft for discussion with the lab's MD group. Numbers marked
*[TBD]* need their input; nothing here has been run.

## Why this is needed

The model has to learn how sequence determines silk mechanics. Neither label
set we have supports that:

- **Single-chain MD (Lu & Buehler 2025).** Each fragment is pulled alone
  (implicit solvent, 0.1 Å/ps, to 3.6 Å per residue), so inter-chain β-sheet
  crystals — the main source of silk strength — cannot form. Fragments were
  chosen by predicted-structure confidence (pLDDT ≥ 50), which favors folded
  terminal domains: only ~10% of fragments lie mostly inside the repetitive
  region, and about half lie mostly outside it. Molecular strength is mostly
  the final backbone stretch; toughness scales with length. Using the
  paper's own fiber-property assignment, these labels do not correlate with
  measured fiber mechanics across 226 samples (all |ρ| ≤ 0.10), nor for the
  57 samples whose fragments come from the repeat region
  (`analyses/md_provenance.py`, `analyses/md_vs_fiber.py`).
- **Measured fibers (Spider Silkome).** Strength and modulus vary as much
  between individual spiders as between genera, so they cannot supervise a
  sequence-only model (`silkome_fiber_model.py`).

The benchmark should produce labels that (1) depend on sequence through the
known mechanism, (2) are measured at **fixed chain length**, so length cannot
explain them, and (3) come with mechanism read-outs that a reasoning model's
explanation can be checked against.

## Target definition

"Intrinsic properties of a spidroin repeat design under a reference
process": mechanical response of a multi-chain assembly built from that
sequence under one fixed protocol. Natural-fiber values are evaluation only.

## Sequence design (factorial, fixed length)

Base unit: a MaSp-like repeat = amorphous segment + crystalline block, tiled
to a fixed total length per chain *[TBD: e.g. 3–4 repeats]*. Amorphous
segments are padded so every design has the same residue count.

| Factor | Levels (proposed) | Why |
|---|---|---|
| Poly-Ala block length | 4, 6, 8, 10, 12 | Crystallite size; the textbook strength/stiffness driver |
| Crystalline-block composition | pure A; A with S/V substitutions (SAV) | Arakawa et al. define crystalline blocks as S/A/V runs |
| Amorphous motif type | GGX-rich (MaSp1-like); GPGXX-rich (MaSp2-like) | Extensibility; Pro content |
| Tyr/Gln motifs per repeat | 0, 1, 2 (YGQGG / QQ insertions) | Top published toughness motif (YGQGG; not reproducible from public data) and MaSp2 Q-dipeptides |

Full factorial: 5 × 2 × 2 × 3 = 60 designs. Add ~30–50 natural repeat
regions sampled across genera from Spider Silkome (exact repeat sequences,
trimmed to the same length) to anchor the designs to real sequences and to
allow a natural-fiber comparison.

## System and protocol (to agree with the MD group)

- Assembly: *[TBD]* chains per system, crystalline blocks pre-aligned as
  stacked β-sheets (crystal-plus-amorphous model), so crystals *can* form or
  fail to form depending on sequence.
- Solvent: *[TBD]* explicit vs. implicit. Explicit or hydrated runs would also
  allow a supercontraction proxy, the most learnable measured property.
- Loading: tensile pulling of the assembly at two rates *[TBD]*, slower than
  0.1 Å/ps, to separate rate- and drag-dependent effects from intrinsic ones.
- Replicates: 3 seeds per design, to estimate the noise ceiling (the
  single-chain set's repeats agree within ~5%).

## Read-outs per run

- Mechanics: initial modulus, yield/peak stress, strain at failure,
  toughness **per residue**.
- Mechanism (for checking reasoning): β-sheet H-bond count and crystallite
  integrity over strain, strain at which crystals rupture or slip, amorphous
  end-to-end extension. These turn "more poly-Ala, so stiffer because larger
  crystallites" into a checkable claim.

## Acceptance checks before using the labels

1. **Known directions hold:** longer poly-Ala blocks raise modulus/strength;
   Pro-rich amorphous segments raise extensibility. If they do not, the
   protocol is not capturing the mechanism.
2. **Not a length or rate artifact:** results stable across the two pulling
   rates in rank order; length is fixed by design.
3. **Reproducible:** replicate spread well below the effect sizes between
   factor levels.
4. **Natural anchor:** report (do not tune toward) the correlation of
   natural-repeat results with measured strain and modulus across species.
   A null here is informative, not disqualifying.

## How the labels feed the model

- **Stage A** (sequence reading, `sequence_reading_curriculum.py`) — already
  built; no simulation needed.
- **Stage B** (mechanism): any two designs differing in one factor become a
  pair — "which is stiffer/tougher, and why?" — scored on direction and on
  the mechanism read-outs.
- **Stage C** (properties): predict the read-outs with uncertainty, scored by
  skill over a length/composition baseline and calibration to the replicate
  noise ceiling.

## Open questions for the MD group

- Existing multi-chain or crystal-plus-amorphous silk setups we can reuse?
- Chains per system, solvent model and pulling rates that are affordable for
  ~100 designs × 3 replicates × 2 rates?
- Compute budget and timeline *[TBD]*.
