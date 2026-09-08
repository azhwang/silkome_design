# Reasoning Pipeline Scaffold

Five modules implementing the staged roadmap discussed for the silk
reasoning project (tool validation → warm traces → grounded-vs-shortcut
ORPO → staged multi-component GRPO):

## Real dataset

The real silkome dataset is `d5ma00154d1_suppl.csv` (2,177 MaSp sequences
with fiber-level mechanical properties — sequence, force_vector, strength,
toughness, and normalized `strength_norm`/`toughness_norm` in `[0, 1]`),
supplementary data from the RSC paper at
https://pubs.rsc.org/ma/article/6/13/4267/896018. `d5ma00154d2_suppl.csv`
is the same set with added model predictions. `ALL_SILK_SEQ.csv` (10,449
sequences, no mechanical properties) is the raw sequence pool from
[SilkomeGPT](https://github.com/lamm-mit/SilkomeGPT). All three are
gitignored (large, externally sourced — not ours to redistribute here);
place them in the repo root to run against real data.

`pilot_batch_driver.py`'s `_COLUMN_ALIASES` has been validated against
`d5ma00154d1_suppl.csv`: the sequence column is `seq`, and ground truth
is pulled from `strength_norm`/`toughness_norm` in preference to the raw
`strength`/`toughness` columns, which are in arbitrary units (~211-1284
and ~776-908963 respectively) that would silently break
`outcome_accuracy_reward`'s `[0, 1]` error math if used directly.

A 20-sample real pilot run (real GPT-5.5 completions, not mocked) against
this dataset scored **mean grounding 0.98, 60% of traces fully grounded**
(343 numeric + 145 position + 81 motif claims checked; 0 API errors). That
run surfaced two more grounding-checker bugs — see "Bugs found and
fixed" — which, once fixed, pushed a second 20-sample run to **mean
grounding 1.00, 85% fully grounded** with 0 API errors. A full 200-sample
real pilot batch is the current in-progress step.

- **`bio_tools.py`** (Stage 0) — real, self-contained bioinformatics
  analyses (Kyte-Doolittle hydropathy, Chou-Fasman secondary-structure
  propensity, motif detection, repeat-periodicity analysis, Uversky
  charge-hydropathy disorder prediction) plus a validation harness that
  checks them against a synthetic positive control and the real PySp
  sequence from the handoff doc. Run it directly (`python3 bio_tools.py`)
  to see the pass/fail report — it also surfaces a genuine, documented tool
  limitation (classic Chou-Fasman under-scores poly-Ala as beta-rich)
  rather than hiding it.

- **`grounding_checker.py`** (Stage 0/1) — cross-checks specific claims
  (numbers, residue positions, motif names) in a reasoning trace against
  the tool outputs that preceded them, and audits a pilot batch for
  hallucination rate.

- **`trace_generation.py`** (Stage 1) — orchestrates
  `sequence → run bio_tools → build LLM prompt → LLM call → structured
  trace → grounding audit`. The LLM call itself (`call_trace_generator_llm`)
  is a stub — wire it to your GPT-5.5 client. A `mock_trace_generator_llm`
  with one genuine and one deliberately fabricated claim is included so you
  can run the full pipeline end to end today and confirm the grounding
  checker actually catches the plant.

- **`pilot_batch_driver.py`** (Stage 1) — the actual script you run: loads a
  real batch of sequences from a `.csv`/`.json`/`.jsonl` dataset file
  (flexible, case-insensitive column matching — see `_COLUMN_ALIASES`),
  runs each through `trace_generation.py` with retry-with-backoff for
  transient API failures, saves every trace + its grounding report to a
  JSON output file, and prints the aggregate audit summary. Run with
  `--use-mock` to test the driver itself today without a dataset file or
  API key.

- **`orpo_pair_construction.py`** (Stage 3) — builds ORPO preference pairs
  that isolate reasoning quality from answer correctness: same-answer
  contrasts (grounded vs. shortcut reasoning reaching the same conclusion),
  anti-shortcut contrasts (grounded-but-wrong preferred over
  shortcut-but-right, at a tunable minority ratio, to actually suppress
  lucky guessing), and standard correctness-dominant pairs for volume.

- **`staged_grpo_rewards.py`** (Stage 4) — the multi-component GRPO reward
  (outcome accuracy, step coherence, tool-use correctness, PRM-based
  reasoning quality, group-level calibration), gated behind a
  `StagedRewardScheduler(stage=1..4)` so components come online in order of
  signal reliability rather than all at once.

## What's real vs. stubbed

Everything runs standalone and has been tested (`python3 <file>.py` in each
file runs a worked example; all six pass a full regression check together).
`call_trace_generator_llm()` in `trace_generation.py` is wired to a real
OpenAI-compatible client (reads `OPENAI_API_KEY` from the environment) and
**has now been tested against a live API**: a 20-sample real pilot run
against `d5ma00154d1_suppl.csv` completed with 0 API errors and a mean
grounding score of 0.98 (see "Real dataset" above).

`call_prm_judge()` in `staged_grpo_rewards.py` is still an unimplemented
stub (Stage 3 of the reward schedule) — same wiring pattern applies there
once you're ready for it. Validate its judgments against held-out human or
independent-model labels before enabling Stage 3; an unvalidated judge
steering RL gradients is a silent reward-hacking vector.

## Bugs found and fixed while building this

Worth knowing about since they affected real behavior, not just style:

- `repeat_pattern_analyzer`'s default `max_period=20` couldn't detect the
  35-residue repeat unit in the synthetic MaSp-like test sequence — raised
  to 50.
- The original PySp-vs-MaSp turn-propensity validation check encoded a
  wrong assumption: Chou-Fasman's Pturn for glycine (1.56) is actually
  *higher* than proline's (1.52), so a glycine-rich sequence can out-score
  a genuinely proline-rich one on that aggregate metric. Replaced with a
  direct proline-content comparison, and kept the propensity discrepancy as
  an explicit "known limitation" check rather than deleting it.
- `grounding_checker.py`'s motif matching did exact substring comparison,
  so `bio_tools.py`'s `poly_alanine` (underscore) label never matched a
  trace's `poly-alanine` (hyphen) phrasing — a real grounded claim was
  being flagged as hallucinated purely due to naming-convention mismatch.
  Fixed with normalization (strip hyphens/underscores/spaces before
  comparing).
- `_split_steps` in `staged_grpo_rewards.py` split on every newline,
  including mid-sentence line wraps from fixed-width text, which broke
  coherence scoring into nonsense fragments. Fixed to only split on
  sentence-ending punctuation and blank-line paragraph breaks.
- A numeric-claim regex bug read `12-40` as the negative number `-40`
  rather than part of a range; a range-flattening bug then caused genuinely
  grounded position claims (matching an exact tool-reported span) to be
  misreported as ungrounded. Both fixed in `grounding_checker.py`.
- The first version of the claim-role heuristic (direct vs. contrastive
  mentions, added to reduce noise in the grounding score — see below)
  checked the whole enclosing sentence for contrast cues like "disrupt" or
  "rather than". Real LLM prose runs long, and a single sentence containing
  both a genuine direct claim ("identified a proline-rich turn motif") and
  an unrelated contrast later in the same sentence ("...rather than a
  dragline protein...") caused the genuine claim to be wrongly swept into
  the contrastive bucket. Fixed by narrowing to a local character window
  immediately around each claim instead of the full sentence.
- **A real GPT-5.5 completion exposed four more bugs the synthetic test
  examples never triggered** (numbered-list reasoning is exactly the shape
  that broke this):
  - Markdown-style list markers ("1. ", "2. ") were extracted as fabricated
    numeric claims, and — worse — could spuriously "ground" against
    unrelated small integers anywhere in the tool output (residue counts,
    list lengths), silently inflating the grounding score with meaningless
    matches. Fixed by masking list markers before claim extraction.
  - Numeric tolerance was a flat 0.5 absolute, which is enormous for
    propensity/hydropathy scores that mostly live in a 0–2 range — a claim
    of `0.954` matched against a completely unrelated field's `0.4851`
    purely because it happened to be checked first and was "close enough".
    Fixed with a much tighter, mostly-relative tolerance (3% of the value,
    0.02 absolute floor) and best-match (not first-match) selection.
  - Python's `bool` is a subclass of `int`, so a claim of `0` could
    spuriously ground against any `False` value anywhere in the tool
    output. Booleans are now explicitly excluded from numeric matching.
  - The position regex required a "residue"/"position" keyword immediately
    before a number range, so real spans phrased as "beta_rich_spans at
    16-23, 23-35..." or "the span at 123-146" — the single most specific,
    checkable claims in a real trace — were never even extracted. Broadened
    to also catch bare `N-M` ranges plus more keyword variants
    (spans/domains/regions/turns).
  - Even after those fixes, "beta-sheet" claims still failed, because the
    tool reports `mean_beta_propensity`/`beta_rich_spans` fields, never the
    literal string "beta-sheet" — a real terminology gap, not a
    hallucination. Added a synonym-to-field-name mapping
    (`_STRUCTURAL_TERM_KEY_HINTS`) so a term can ground against a
    semantically-related tool *field name*, with the evidence string
    explicitly noting when a match came from this path rather than a
    literal value/motif match.
  - On the real trace these came from, the grounding score moved from a
    misleading 0.75 (built on several coincidental wrong-field matches) to
    0.96 (44/45 direct claims, all exact `diff=0` matches) once fixed —
    worth internalizing that the *first* number a checker like this
    produces on real model output shouldn't be trusted at face value until
    you've spot-checked a few of its "OK"s, not just its "FAIL"s.
- A real 20-sample pilot batch against the actual silkome dataset surfaced
  a seventh bug: `_POSITION_RE`'s single-residue branch matched "residue 0"
  inside phrases like "net charge per residue 0.0194", because `\b` fires
  right before the decimal point — a per-residue decimal statistic isn't a
  residue index. Fixed with a `(?!\.\d)` lookahead so a number immediately
  followed by a decimal fraction is no longer parsed as a bare position
  claim.
- The same batch surfaced two more real terminology gaps, same shape as
  the "beta-sheet" gap above: `intrinsically disordered` never matched
  because `disorder_prediction_chplot` reports `predicted_disordered`, and
  `crystalline region` never matched because the tool's only crystalline
  signal was the `poly_alanine` motif. Fixing the latter also exposed a
  bug in the hint mechanism itself: `_STRUCTURAL_TERM_KEY_HINTS` only
  searched flattened dict *keys*, but `poly_alanine` appears as a *value*
  inside `motif_detector`'s `motifs` list, not as a key — a key-only
  search could never find it. Hints now check flattened values too.
- A later real trace inferred "nanocrystalline regions" directly from its
  own reported beta-rich spans ("beta-sheet-rich segments can form
  hydrogen-bonded nanocrystalline regions") — biologically correct, since
  silk crystallites are hydrogen-bonded beta-sheets, but `crystalline
  region`'s hints only pointed at `poly_alanine`. Added
  `beta_propensity`/`beta_rich` as additional hint targets. After both
  terminology-gap fixes, the 20-sample pilot's mean grounding score moved
  from 0.98 to 1.00 and fully-grounded fraction from 60% to 85%.
- Two remaining motif FAILs in that same run (`amorphous region`,
  mentioned twice) turned out to be a *different* failure mode, not
  another naming gap: the model used the term in generic, hedged
  textbook-style asides about silk mechanics ("toughness in silk depends
  strongly on extensible amorphous regions that dissipate energy...")
  rather than as a specific claim about the sequence at hand. A synonym
  hint would force these to "ground" while quietly breaking what
  "grounded" means — the trace never actually claims *this sequence* has
  that property. Left unfixed deliberately; see "Known limitations"
  below.

## Claim roles: direct vs. contrastive mentions

`grounding_checker.py` now classifies each claim as `"direct"` (asserted as
something the tools observed) or `"contrastive"` (mentioned for contrast or
negation, e.g. "X disrupts Y formation," "unlike X," "lacks Y"). The primary
`grounding_score` is computed from direct claims only; contrastive mentions
are tracked separately (`contrastive_grounding_score` on `GroundingReport`,
`n_contrastive_mentions`/`contrastive_ungrounded_fraction` on
`AuditSummary`) rather than folded into the same number. This matters
because a trace legitimately discussing "unlike MaSp's poly-Ala crystalline
domains, this sequence lacks beta-sheet propensity" was previously scored
identically to a trace fabricating a beta-sheet claim outright — same
FAIL, very different actual reliability. The classifier is deliberately
conservative (biased toward "direct" when in doubt), since under-flagging
just means some background reasoning counts against the direct score (the
prior status quo), while over-flagging would let real fabrications slip
into the unscored bucket, which is the worse failure mode.

### Known limitation: generic/hedged mechanistic language

The direct/contrastive split doesn't capture a third real pattern: generic,
hedged textbook-style reasoning ("turns **can** interrupt long crystals and
create... amorphous regions") that uses structural vocabulary as
explanatory scaffolding, not as a specific claim about the sequence at
hand. These currently score as ungrounded direct claims (correctly, in the
sense that nothing in the tool output specifically supports them — but
also somewhat unfairly, since the model never claimed the tools found
this). A proper fix would add a third claim role (e.g. `"generic"`,
detected via modal hedging like "can"/"may"/"tends to" plus non-specific
subjects like "silk" or "in general") tracked separately from the direct
score, the same way contrastive mentions already are. Not implemented —
deliberately deferred rather than papering over it with a loose hint
mapping that would blur what "grounded" means.

## What still needs your input to be load-bearing

- Wire `call_prm_judge` to your actual PRM/judge client (same pattern as
  `call_trace_generator_llm` — environment variable for the key, a custom
  `*NotConfiguredError` distinct from transient failures, fail fast rather
  than retrying a missing-credential error).
- `bio_tools.py` is explicitly a "does the harness work at all" toolset
  (classical 1970s-2000s heuristics) — swap in real DSSP/AlphaFold/learned
  disorder predictors for production trace generation once available.
- `MOTIF_PATTERNS` / `_KNOWN_MOTIF_TERMS` are starter vocabularies on both
  the tool side and the checker side — extend both together as you add
  tools, and keep an eye out for the same naming-mismatch failure mode
  fixed above when you do.
- `tool_use_reward` in `staged_grpo_rewards.py` needs your inference
  harness to emit structured `TraceStep` sequences, not just flat
  completion strings.
- Calibration reward is intentionally *not* in the flat `reward_funcs` list
  TRL expects — its group-level computation needs GRPO's k-samples-per-prompt
  structure. Call `calibration_reward_grouped` separately and fold it into
  the reward your training loop passes to the advantage computation.
- `pilot_batch_driver.py`'s `_COLUMN_ALIASES` have been validated against
  the real `d5ma00154d1_suppl.csv` schema (see "Real dataset" above). If
  you point the driver at a different dataset file, spot-check the first
  few loaded `SilkSample`s again — a wrong alias match on the wrong column
  fails silently rather than raising.

