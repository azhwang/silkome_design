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
- **Stage 4 (staged GRPO reward): SCAFFOLDED, not runnable yet.**
  `staged_grpo_rewards.py`'s reward functions are tested on synthetic
  examples. `call_prm_judge()` is an unimplemented stub — needed for Stage
  3 of the reward schedule (not to be confused with pipeline Stage 3/ORPO;
  the staged reward schedule's own stages 1-4 are internal to that file).
- **Thesis Aim 2 (self-assembling peptide networks): STARTED.**
  `aim2_peptide_networks/` holds the evaluation floor and dataset
  feasibility check — see that directory's README. Three dependency-free
  modules, all self-checking: `dataset_schema.py` (canonical records with
  mandatory measurement conditions + a heterogeneity audit that returns a
  go/marginal/no-go verdict on holdout feasibility), `identity_holdout.py`
  (identity-based partition with a verified no-leakage guarantee), and
  `baselines.py` (reports that refuse a bare score, plus a
  constant-guess audit that generalises the C.6 finding).
  **Real data is now wired in for link 2 only (aggregation propensity).**
  `ap_dataset.py` + `ap_model.py` run against 62,159 peptides (length 5-10)
  with coarse-grained AP labels from
  `github.com/Zihan-Liu-00/DL_for_Peptide` (Liu et al., Brief. Bioinform.
  2023). Ridge on 29 `bio_tools` descriptors reaches Spearman 0.910 on a
  random split and 0.907 on an identity split, beating the trivial and
  single-covariate baselines on both MAE and rho, with the correctness
  criterion passing the constant-guess audit (75% vs 42%). Two structural
  findings: all 10,450 pentapeptides form a *single* identity cluster at
  threshold 0.6 (combinatorial percolation) while lengths 6-10 are
  near-singletons; and because of that the identity split holds out zero
  length-5 peptides, so the two splits are not comparable (the module
  prints this warning itself). **The AP label is simulation output, not
  measurement** - it is the chain's intermediate, not the modulus.
  **Experimental data is now wired in for the gelation + morphology nodes.**
  `xu2023_gelation.py` loads Xu et al., Nat. Commun. 14, 3880 (2023)
  Supplementary Data 7: 165 synthesised tetrapeptides with observed state
  (60.6% gel), TEM morphology and the authors' APH/APHC ranks. Results
  (identity-grouped 5-fold OOF, logistic on the same 29 descriptors):
  gelation AUC 0.703 / balanced 0.680; fibrous morphology AUC 0.568 /
  balanced 0.544; majority baseline 0.500 on both. The informative
  contrast with the AP result above: descriptors reproduce the
  *simulator's* output almost perfectly (rho 0.910) but predict the
  *experiment* only modestly, so the coarse-grained step is not redundant
  with cheap descriptors where it matters.
  **Critical data trap found: `concentration_mM` is label-determined.**
  The protocol escalates 30 -> 60 -> 90 -> 120 mM until gelation, so every
  non-gel sits at 120 mM and "concentration < 120 => gel" holds 82/82 with
  zero false positives (89.1% accuracy vs 60.6% majority). Feeding it to
  the model lifts AUC 0.703 -> 0.902 while learning nothing.
  `protocol_escalation_report()` detects this generically (tests for
  one-sided precision, not high accuracy); escalation is standard in the
  gelation literature, so expect it in any compiled corpus. This qualifies
  `dataset_schema.py`'s treatment of concentration as a covariate - right
  for an independently set concentration, wrong for an escalated one.
  The authors' APHC baseline scores higher (AUC 0.926) but is NOT a fair
  comparator: the 165 peptides were selected by that very score.
  **A third corpus is wired in: literature-mined phase data.**
  `peptideminer_phases.py` loads `lamm-mit/PeptideMiner`'s
  `phase_data_clean.csv` (Yang et al., arXiv:2411.05421): 1,017 observations
  with peptide, terminal chemistry, concentration, pH, temperature,
  incubation and phase (hydrogel 219, fiber 96, none 407, ...). It produced
  the project's sharpest leakage result: **1,017 rows but only 119 unique
  sequences**, which at identity 0.6 form 23 clusters with the largest
  holding 90 (76%). Row-level folds inflate AUC by +0.209 (hydrogel) and
  +0.357 (fiber) over identity-grouped folds - and for fiber the grouped
  AUC is exactly 0.500, i.e. the row-level 0.857 was entirely sequence
  memorisation. Contrast Finding 6 on the Xu data, where grouping cost only
  0.016 AUC: near-duplicate structure is cheap when a corpus samples
  sequence space sparsely and ruinous when it re-measures few sequences
  many times, and only the cluster count distinguishes the two.
  Also: `protocol_escalation_report()` reports precision 0.272 here and
  does NOT flag concentration, versus 1.000 on Xu - it separates a genuine
  experimental variable from a label proxy without being told which is which.
  **STILL NO MODULUS DATA ANYWHERE.** Three experimental/simulated corpora
  now cover the chain's gelation and morphology nodes; none carries a
  mechanical property. Specific dead ends, so they are not re-chased:
  Simpson et al. 2025 (doi 10.1002/adma.202415031, the proposal's ref 18,
  the most on-point paper) is open access on PMC but PMC CAPTCHAs automated
  requests and Wiley is off the allowlist; the Buehler group's
  arXiv:2505.08850 computes Young's modulus for all 400 dipeptides and all
  8,000 tripeptides but its cited repo `lamm-mit/PeptideModulus` 404s on
  every branch; Leeds 10.5518/975 and Montazeri et al. are on blocked
  hosts; Zenodo/figshare searches return only single-system studies.
  Getting modulus data is now the binding constraint on Aim 2.
- **No training has been run.** No SFT, ORPO, or GRPO training has been
  executed. Everything above is data/reward pipeline scaffolding.

**Relation to the thesis proposal.** The three aims are: Aim 1 (FGF-receptor
binding, complete, manuscript under internal review), Aim 2 (peptide network
mechanics — `aim2_peptide_networks/`), and Aim 3 (silk molecule-to-fiber,
which the original pipeline in this repo serves). The agent/tooling layer
(`bio_tools.py`, `grounding_checker.py`, `trace_generation.py`) is shared
across Aims 2 and 3 — extend it in place rather than forking it per aim.

**Immediate next step (Aim 2):** compile the experimental peptide gelation +
rheology corpus and run `heterogeneity_report()` on it. That returns a
go/no-go on the aim's own stated top risk (data too sparse or heterogeneous
to support a holdout) and is a prerequisite for everything downstream of it.
The AP work above covers link 2 only and does not substitute for it. Known
leads not yet reachable from this session's network: PMC
(`pmc.ncbi.nlm.nih.gov`), Springer supplementary
(`static-content.springer.com`) and Zenodo are blocked; `arxiv.org`,
`figshare.com`, `pubs.rsc.org` and `raw.githubusercontent.com` are allowed.

**Immediate next step (Aim 3 / silk pipeline):** before scaling `orpo_trace_pool.py` further, make
the traces predictive and fix the correctness label (see the baseline check
above) — otherwise ORPO would reinforce faithful-but-uninformative
reasoning. Then: wire and validate `call_prm_judge()` (currently an unimplemented stub) in
`staged_grpo_rewards.py` against held-out human/independent-model labels,
then run the first real training pass (no SFT/ORPO/GRPO training has been
executed yet — everything above is a validated but untrained pipeline).

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
