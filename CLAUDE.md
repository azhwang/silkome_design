# Silkome Reasoning Pipeline

## What this project is

A staged pipeline for training a tool-augmented reasoning model to predict
protein fiber-level mechanical properties from sequence — validated on
spider silk (the "silkome"), with the goal of a **transferable reasoning
capability**, not a silk-specific regressor. Silk is the sandbox because it
has strong domain priors and a large labeled dataset; the deliverable is a
method that should generalize to other protein families.

Full roadmap (5 stages):
0. **Tool validation** — confirm bioinformatics tools match known biology
1. **Warm traces** — GPT-5.5 + tools generate grounded reasoning traces;
   audit hallucination rate
2. **Grounded ORPO** — preference pairs that isolate reasoning quality from
   lucky guessing
3. **Staged GRPO** — multi-component reward, brought online one signal at a
   time
4. **Transfer check** — held-out protein family confirms the method
   generalizes

## Current status (update this section as work progresses)

- **Stage 0 (tool validation): DONE.** `bio_tools.py` implements 5
  dependency-free bioinformatics analyses with a built-in validation
  harness (`python3 bio_tools.py`) checked against known biology + the real
  PySp sequence from the original handoff doc.
- **Stage 1 (grounding + trace generation): DONE — real pilot run.**
  `grounding_checker.py` cross-checks trace claims against tool outputs.
  `trace_generation.py`'s `call_trace_generator_llm` is wired to a real
  OpenAI-compatible client (GPT-5.5). A real 200-sample pilot batch against
  `d5ma00154d1_suppl.csv` (via `pilot_batch_driver.py`) succeeded 200/200
  with 0 API errors: mean grounding score 0.994, 86% of traces fully
  grounded, 5,259 claims checked. Getting there required manually reading
  failing claims (not trusting the aggregate score) and fixing several real
  grounding-checker bugs along the way — see "Bugs found and fixed" in
  README.md.
- **Stage 2/3 (grounded ORPO): VALIDATED on real data at small scale.**
  `orpo_trace_pool.py` generates real grounded + shortcut (tool-withheld)
  completions and builds preference pairs via `orpo_pair_construction.py`.
  A real n=30 run (120 LLM calls, 0 errors) produced 60 pairs across 4
  types (`same_answer`, `standard`, `anti_shortcut`, `standard_fallback`),
  with 97% of prompts showing genuine reasoning-quality diversity between
  the grounded and shortcut completions. **Not yet scaled to the full
  warm-trace set.**
- **Baseline check (`llm_vs_regressor_comparison.py`): traces do NOT predict
  well.** On 437 sequences from the n=500 run, LLM traces have Spearman
  -0.06 (strength) / -0.40 (toughness) vs. +0.57 / +0.98 for length alone;
  predict-the-mean beats them on absolute error; and a constant mean guess
  is "correct" (error <= 0.1) 93% of the time vs. 8% for tool-grounded
  traces, so the ORPO correctness label is not informative as defined.
  Labels derive from force-extension vectors (likely simulated), and
  toughness is ~entirely chain length. Details in README.md.
- **Label audit (Sep–Oct 2026): neither dataset supports a sequence-only
  fiber-mechanics target as-is.** Details and scripts in `analyses/README.md`.
  - *Simulated labels* (`d5ma00154d1_suppl.csv`, Lu & Buehler 2025; setup
    confirmed by Wei Lu): single chains, implicit solvent, pulled at
    0.1 Å/ps to 3.6 Å/residue; force in pN. Strength is mostly the final
    backbone stretch, toughness scales with length. On held-out sequence
    clusters length gives R² 0.97 (toughness) / 0.22 (strength), composition
    lifts strength to 0.32; ESM-2, contact-map and curve-shape features add
    nothing (`silk_regressor.py`, `silk_attribution.py`, `analyses/`).
    The d2 `pre_*` columns are SilkomeGPT predictions made from the
    fragments, not measurements.
  - *Measured fibers* (Spider Silkome, `silkome_download.py` →
    `silkome_fiber_model.py`): 446 samples / 296 species. Strength and
    modulus vary as much between individuals as between genera; only strain
    (and supercontraction) has species-level signal. Held-out-genus R² ≈ 0
    except strain 0.15, matched by family alone.
  - *MD vs. fiber*: 76% of MD fragments are exact substrings of Silkome
    genes; across 185 species no MD measure correlates with any fiber
    property (`analyses/md_vs_fiber.py`).
  - *Provenance* (authors' files, `analyses/md_provenance.py`): 1,025 MD
    fragments are natural and 1,152 SilkomeGPT-generated; only ~10% lie
    mostly inside the repeat region. The null holds under the paper's own
    property assignment (226 measured samples, all |ρ| ≤ 0.10).
  - *Arakawa et al. motif screen*: their supercontraction–poly-Ala result
    replicates; the MaSp1 toughness motifs (YGQGG, r = 0.55) do not reproduce
    from public sequences, traced to paralog labeling (repeat-based groups
    vs. the database's terminal-domain labels).
- **Other Silkome properties (Nov 2026; `analyses/other_properties.py`,
  `supercontraction_validation.py`): they do not rescue a sequence-only
  target.** Mechanical labels are mostly individual-level noise (species ICC
  0.13-0.26, ceiling R² 0.18-0.33), so the ~0 held-out-genus R² is close to
  the achievable ceiling for strength/modulus/toughness. No other measured
  property (crystallinity, birefringence, diameter, body size, water, thermal
  loss) is predictable beyond family. Diameter-adjusting strength doubles its
  species-level ICC (0.13 → 0.25) but it stays unpredictable. Supercontraction
  looked predictable (R² +0.26) but is a completeness/lineage proxy (carried
  by non-dragline gene presence; vanishes within Araneidae) — do not use it as
  evidence that sequence predicts a fiber property.
- **Stage A sequence-reading curriculum: BUILT.**
  `sequence_reading_curriculum.py` generates ~12k genus-split tasks
  (poly-Ala spans, crystalline blocks, motif counts, composition,
  comparisons) with exact gold answers and a field-level checker usable as a
  verifiable reward. A non-reading baseline scores 0.04–0.10 on the block
  and composition tasks. No model has been evaluated or trained on it yet
  (`--eval-llm N` makes real API calls).
- **Stage 4 (staged GRPO reward): SCAFFOLDED, not runnable yet.**
  `staged_grpo_rewards.py`'s reward functions are tested on synthetic
  examples. `call_prm_judge()` is an unimplemented stub — needed for Stage
  3 of the reward schedule (not to be confused with pipeline Stage 3/ORPO;
  the staged reward schedule's own stages 1-4 are internal to that file).
- **No training has been run.** No SFT, ORPO, or GRPO training has been
  executed. Everything above is data/reward pipeline scaffolding.

**Immediate next steps:** the goal is an end-to-end model that predicts
properties from sequence alone (no tools at inference), trained in stages:
A (sequence reading — built), B (mechanism pairs), C (calibrated property
prediction). The bottleneck is labels that sequence can explain, so:
1. Decide the target definition with the PI: intrinsic properties under a
   reference process (recommended) vs. natural-dragline ranges.
2. Scope the multi-chain simulation benchmark with the lab's MD group
   (`docs/multichain_benchmark_spec.md`) — the labels for stages B and C.
3. Request the gene-to-paralog/group table behind the Arakawa et al. motif
   screen from the Silkome authors.
4. Evaluate a frontier-model baseline on the stage A test split, then train
   on it. Do not scale `orpo_trace_pool.py` on the current labels — a
   constant mean guess is "correct" 93% of the time under its definition.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in OPENAI_API_KEY
```

Every script reads `OPENAI_API_KEY` from the environment (never hardcode
it). Loading `.env` requires either `python-dotenv` (not currently a hard
dependency — add it and a `load_dotenv()` call if you want this instead of
`export`-ing the variable manually) or just `export OPENAI_API_KEY=...` in
your shell before running anything.

## Running things

```bash
# Regression-check every module (each has a __main__ demo/test)
for f in grounding_checker orpo_pair_construction staged_grpo_rewards bio_tools trace_generation; do
  python3 $f.py
done

# Measured-fiber data (one-time download from spider-silkome.org, ~15 min, cached)
python3 silkome_download.py
python3 silkome_fiber_model.py        # held-out-genus models

# Tool-free baselines on the simulated labels
python3 silk_regressor.py             # add --esm-model facebook/esm2_t12_35M_UR50D for ESM

# Stage A sequence-reading curriculum (needs silkome_data/)
python3 sequence_reading_curriculum.py

# One-off analyses behind the findings: see analyses/README.md

# Test the pilot batch driver without a dataset file or API key
python3 pilot_batch_driver.py --use-mock

# Real pilot batch (once OPENAI_API_KEY is set and the dataset is available)
python3 pilot_batch_driver.py --data-path <path-to-silkome-dataset> --n-samples 200
```

## Non-obvious gotchas (read before touching the LLM-calling code)

- **The model is reasoning-tier and has a narrow parameter surface.**
  `temperature` cannot be overridden (must stay at the default of 1) — the
  API rejects any other value with a 400. If a future error complains about
  an unsupported parameter (e.g. `max_tokens` vs `max_completion_tokens`),
  it's likely the same root cause: this model class supports fewer
  standard chat-completion parameters than typical.
- **Response decompression can break** depending on the local
  `openai`/`httpx`/`brotli`/`zstandard` version combination — surfaces
  misleadingly as `openai.APIConnectionError`, but the real cause is a
  `TypeError` inside the decompressor, not an actual network problem. The
  code already sends `Accept-Encoding: identity` to sidestep this by
  avoiding compressed responses entirely; if you still hit it, try
  `pip install --upgrade --force-reinstall brotli zstandard`, and if that
  fails, isolate in a clean venv rather than continuing to patch a shared
  environment — this and two other dependency conflicts (aiohttp, the
  temperature param) all surfaced in the same anaconda base environment in
  a row, which is itself a signal to isolate rather than keep patching.
- **`LLMNotConfiguredError`** (in `trace_generation.py`) is deliberately
  distinct from transient API errors — it's raised for missing
  keys/packages and deliberately *not* retried by `pilot_batch_driver.py`'s
  retry-with-backoff wrapper, and causes the whole batch to fail fast
  rather than logging a confusing per-sample error for every single item.
  Keep this distinction if you add new failure modes to the LLM call.
- **The grounding checker needed real model output to find its real bugs.**
  Five bugs were found and fixed only after running actual GPT-5.5
  completions (not the synthetic test examples): markdown list markers
  ("1.", "2.") being parsed as fake claims, numeric tolerance far too loose
  (matched wrong tool-output fields by coincidence), Python's `bool`-is-an-
  `int` subclass landmine, span claims missed unless phrased with a
  "residue"/"position" keyword, and a terminology gap where "beta-sheet" in
  prose didn't match the tool's literal `beta_propensity` field name (fixed
  with a synonym-to-field-name mapping in `_STRUCTURAL_TERM_KEY_HINTS`).
  **Takeaway: don't trust a grounding score at face value until you've
  manually read a few of its "OK"s, not just its "FAIL"s** — the checker
  can be wrong in either direction, and was, twice, before real data
  exposed it. If you extend the checker further, re-run it against a real
  completion, not just synthetic examples, before trusting new logic.

## Known gaps / things that need your input

- `pilot_batch_driver.py`'s `_COLUMN_ALIASES` (dataset column-name
  matching) is a best guess, never validated against the real silkome
  dataset schema. Spot-check `load_dataset()`'s output against a few real
  rows before a full pilot run.
- `MODEL_NAME = "gpt-5.5"` in `trace_generation.py` may not match your
  provider's actual deployment/model string — confirm against your
  provider's dashboard.
- `call_prm_judge()` in `staged_grpo_rewards.py` is unimplemented. Same
  wiring pattern as `call_trace_generator_llm` applies — validate its
  judgments against held-out human/independent-model labels before
  enabling Stage 3 of the reward schedule; an unvalidated judge steering RL
  gradients is a silent reward-hacking vector.
- `MOTIF_PATTERNS` (`bio_tools.py`), `_KNOWN_MOTIF_TERMS`, and
  `_STRUCTURAL_TERM_KEY_HINTS` (`grounding_checker.py`) are starter
  vocabularies — extend together as tooling grows, and watch for the same
  naming-convention mismatch failure mode that was already found once
  between these two files.
- No automated test suite exists yet — each module's `__main__` block is a
  worked-example smoke test, not a pytest suite. Worth converting if this
  project grows much further; ask before assuming this is wanted.

## Code conventions used so far

- No hardcoded credentials, ever — environment variables only.
- A distinct `*NotConfiguredError` (not a generic exception) for
  missing-credential/missing-package failures, so callers can fail fast
  instead of retrying something that will never succeed.
- Every module is runnable standalone (`python3 <file>.py`) with a
  demonstrative `__main__` block — keep this pattern for new modules.
- Prefer real, tested, dependency-free implementations over stubs
  wherever the logic doesn't actually require an external API
  (`bio_tools.py` is the example — real Kyte-Doolittle/Chou-Fasman/Uversky
  algorithms, not placeholders). Reserve stubs for things that genuinely
  require credentials or infrastructure you don't have.
