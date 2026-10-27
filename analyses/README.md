# Analyses

One-off analyses behind the Aim 2 findings. Each script runs standalone from
the repo root (`python3 analyses/<name>.py`) and prints its results. They
read data from the repo (`d5ma00154d1_suppl.csv`, `silkome_data/` from
`silkome_download.py`, cached embeddings in `embeddings/`).

## Simulated (MD) labels — Lu & Buehler 2025, `d5ma00154d1_suppl.csv`

| Script | Question | Result |
|---|---|---|
| `beyond_length.py` | Does anything beyond chain length predict MD strength/toughness on held-out sequence clusters? | Toughness: length R² 0.97 (ceiling 0.99), nothing adds. Strength: length 0.22 → composition 0.32 (ceiling 0.88); ESM, dipeptides, termini add nothing. Composition gain still rising with more training clusters. |
| `contact_features.py` | Fold-proxy features from ESM-2 650M contact maps (writes the cache `eval_both.py` uses). | Sanity check passes (folded C-terminal domain vs. disordered repeat). |
| `eval_both.py` | (A) Do fold features help? (B) Are the d2 `pre_*` columns learnable? | (A) +0.01 for strength, nothing for toughness. (B) R² ≈ 0 for all four; `pre_*` are SilkomeGPT predictions made from the fragments (confirmed by Wei Lu). |
| `curve_shape.py` | Is the length-independent shape of the force curves learnable? | No: rupture-peak density and curvature R² ≤ 0.09; unfolding-phase forces are length-dominated. |
| `md_provenance.py` | Using the authors' input/extraction files (`silkome_data/md_input_sequences.csv`, `md_fragment_extraction.csv`, from Wei Lu): which fragments are natural, where do they sit, and does MD track fiber properties under the paper's own property assignment? | 1,025 fragments natural, 1,152 SilkomeGPT-generated; 47% of generated fragments appear verbatim in natural Silkome genes. Only ~10% of fragments lie mostly inside the repeat region (~half mostly outside it). MD vs. measured fiber across 226 samples: all \|ρ\| ≤ 0.10; repeat-region fragments only (57 samples): all within chance. |
| `md_vs_fiber.py` | Does MD unfolding mechanics track measured fiber mechanics? Maps fragments to species by exact substring match; writes `silkome_data/md_fragment_species_map.csv`. | 76% of fragments match Silkome genes; across 185 species every MD–fiber correlation is below chance (\|ρ\| ≤ 0.09). |

Simulation setup (Wei Lu, Oct 2026): single chain, implicit solvent (NAMD),
one end fixed, pulled at 0.1 Å/ps with a 1.0 kcal/mol/Å² spring to 3.6 Å per
residue; force in pN, extension in Å; strength = peak force, toughness = area.
Fragments are OmegaFold regions with denoised pLDDT ≥ 50 and length ≥ 10,
re-folded before MD. Repeated fragments were re-simulated with different
automatic NAMD seeds (same OmegaFold structure), so their ~5% spread reflects
MD stochasticity, not a controlled replicate design.

## Measured fiber data — Spider Silkome (Arakawa et al. 2022)

`silkome_fiber_model.py` (repo root) is the main held-out-genus model; these
replicate and stress-test the paper's motif screen. `s4_compare.py` needs the
paper's data S4 (`../sciadv.abo6043_data_s1_to_s4/` by default, or set
`SILKOME_S4`).

| Script | Question | Result |
|---|---|---|
| `arakawa_motifs.py` | Do the paper's Table 1 motifs hold within families, and does their screen generalize to held-out genera? | Mechanical motifs ≈ 0 on public data; screen-then-predict R² < 0 on held-out genera. |
| `s4_compare.py` | Do our correlations track theirs across all 1,724 S4 motifs, and are top hits beyond chance? | Agreement r 0.40–0.53 (0.79 supercontraction). Their MaSp1/2 top hits exceed the chance level (~0.31); ours do not. YGQGG–toughness 0.55 vs. 0.00. |
| `ratio_calibration.py` | Can we reproduce their Fig. S9 / Table S1 amorphous:poly-Ala ratio correlations? | MaSp2–supercontraction reproduces (0.41 vs. 0.39); MaSp1 fails (0.17 vs. 0.53) → paralog labels differ (repeat-based groups vs. terminal-domain labels). |
| `supercontraction_cv.py` | Do the paper's poly-Ala/AGQG motifs predict supercontraction on held-out genera? | Correlations replicate (r ≈ −0.5, also within families) but n ≈ 54, almost all Araneidae; leave-one-genus-out R² −0.09. |
