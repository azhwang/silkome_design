"""
Stage 1: Warm-Trace Generation Orchestrator
==============================================

Wires together the pieces built so far into the actual pilot-batch pipeline:

    sequence + metadata
        -> run_all_tools()            (bio_tools.py, real computation)
        -> build_trace_prompt()        (embeds tool outputs in the LLM prompt)
        -> call_trace_generator_llm()  (STUB — wire to your GPT-5.5 client)
        -> assemble_structured_trace() (-> List[TraceStep])
        -> check_trace_grounding()     (grounding_checker.py, per-trace audit)
        -> audit_trace_batch()         (grounding_checker.py, batch summary)

The one piece that can't run without your credentials is the actual LLM
call. Everything else — tool execution, prompt construction, trace assembly,
grounding audit — is real and runs today. A `mock_trace_generator_llm` is
included so the full pipeline can be exercised end to end right now,
including a deliberately-planted hallucinated claim, to confirm the
grounding checker actually catches what it's supposed to before you spend
API budget on the real thing.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from bio_tools import TOOL_REGISTRY
from grounding_checker import TraceStep, check_trace_grounding, audit_trace_batch, GroundingReport


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

@dataclass
class SilkSample:
    sequence: str
    family: str
    genus: str
    species: str
    protein_category: str
    ground_truth: Optional[Dict[str, float]] = None  # e.g. {"strength": 0.55, "toughness": 0.6}


# ---------------------------------------------------------------------------
# Step 1: run tools, wrap as TraceSteps
# ---------------------------------------------------------------------------

def run_tools_as_trace_steps(sequence: str) -> List[TraceStep]:
    """Runs every registered bioinformatics tool and wraps each result as a tool_call TraceStep."""
    steps = []
    for tool_name, tool_fn in TOOL_REGISTRY.items():
        output = tool_fn(sequence)
        steps.append(TraceStep(
            step_type="tool_call",
            tool_name=tool_name,
            tool_input={"sequence_length": len(sequence)},
            tool_output=output,
        ))
    return steps


# ---------------------------------------------------------------------------
# Step 2: build the LLM prompt, embedding real tool outputs
# ---------------------------------------------------------------------------

TRACE_GENERATION_SYSTEM_PROMPT = """You are a biophysics reasoning engine analyzing spider silk protein \
sequences. You will be given a protein sequence, its taxonomic metadata, and the outputs of several \
bioinformatics analyses that have already been run on it.

Your job: write a step-by-step reasoning chain explaining how the sequence's features (as reported by \
the tool outputs — not features you infer independently) plausibly drive fiber-level strength and \
toughness, grounded explicitly in chemistry and biophysics principles (e.g. hydrogen bonding in \
beta-sheet nanocrystals, entropic elasticity of amorphous chains, crosslink density, disorder and \
extensibility).

Critical constraint: every specific number, residue position, or motif name you state MUST come \
directly from the tool outputs provided below. Do not invent values. If you want to make a claim you \
cannot support with a provided tool output, say so explicitly as an inference rather than stating it as \
an observed fact.

Then predict strength and toughness in [0, 1] and respond in this exact format:

<reasoning>
...your step-by-step reasoning...
</reasoning>
<answer>
{"strength": <float in [0,1]>, "toughness": <float in [0,1]>}
</answer>
"""


def build_trace_prompt(sample: SilkSample, tool_steps: List[TraceStep]) -> str:
    tool_output_block = json.dumps(
        {step.tool_name: step.tool_output for step in tool_steps}, indent=2
    )
    return f"""{TRACE_GENERATION_SYSTEM_PROMPT}

Protein sequence:
{sample.sequence}

Family: {sample.family}
Genus: {sample.genus}
Species: {sample.species}
Protein category: {sample.protein_category}

Tool outputs:
{tool_output_block}
"""


# ---------------------------------------------------------------------------
# Step 2b: the "shortcut" prompt variant — deliberately withholds tool
# outputs, for Stage 2/3 ORPO pair construction (orpo_trace_pool.py).
#
# A real 12-completion sanity run showed this reasoning-tier model grounds
# reliably (~0.97-1.00) at temperature=1 when given tool outputs — there's
# no natural "shortcut" (ungrounded) variance to sample for the
# grounded-vs-shortcut ORPO contrast the roadmap wants. This prompt
# manufactures a genuine shortcut trace instead of hoping one turns up: the
# model sees the same sequence and metadata but never the tool outputs, so
# any specific number/position/motif it states has nothing real to ground
# against (the grounding checker still runs the real tools independently
# and checks the completion's claims against them — see
# generate_shortcut_trace_for_sample below). This is a different reasoning
# *process* (pattern-matching from general priors, not tool-grounded
# analysis), not an instruction to fabricate — deliberately not "invent
# fake numbers and claim they're from analysis," which would teach the
# reward model the wrong thing (that grounding fails because of dishonesty
# specifically, not because the reasoning skipped verification).
# ---------------------------------------------------------------------------

SHORTCUT_SYSTEM_PROMPT = """You are a biophysics expert asked for a fast, intuitive estimate of a spider \
silk protein's fiber-level mechanical properties. You will be given only the protein sequence and its \
taxonomic metadata — no bioinformatics tool outputs are available for this request.

Your job: write a brief reasoning chain based on general domain knowledge and whatever you can judge by \
eye from the sequence and its taxonomy (e.g. typical composition/structure patterns for this protein \
family), then predict strength and toughness. This is a quick expert judgment call, not a rigorous \
per-residue analysis — don't claim to have measured or computed specific values you don't actually have \
access to here.

Respond in this exact format:

<reasoning>
...your step-by-step reasoning...
</reasoning>
<answer>
{"strength": <float in [0,1]>, "toughness": <float in [0,1]>}
</answer>
"""


def build_shortcut_prompt(sample: SilkSample) -> str:
    return f"""{SHORTCUT_SYSTEM_PROMPT}

Protein sequence:
{sample.sequence}

Family: {sample.family}
Genus: {sample.genus}
Species: {sample.species}
Protein category: {sample.protein_category}
"""


# ---------------------------------------------------------------------------
# Step 3: the actual LLM call (stub) + a mock for end-to-end testing
# ---------------------------------------------------------------------------

class LLMNotConfiguredError(RuntimeError):
    """Raised when the LLM client is missing required setup (API key, package) —
    distinct from a transient API failure so callers can fail fast instead of
    retrying or silently logging every sample as an unrelated error."""


def call_trace_generator_llm(prompt: str) -> str:
    """
    Calls the LLM via an OpenAI-compatible client. Reads the API key from
    the OPENAI_API_KEY environment variable — never hardcode it here.

    Adjust MODEL_NAME below to whatever your provider calls it (e.g. your
    org's actual deployment name for GPT-5.5) — I don't have that string
    confirmed against your provider account, so don't assume this literal
    value is correct without checking your provider's docs/dashboard.

    Requires: pip install openai --break-system-packages
    """
    import os
    try:
        from openai import OpenAI
    except Exception as e:
        # Broad on purpose: a broken openai/httpx/aiohttp version combination
        # in the local environment can surface as ImportError, AttributeError,
        # or other exception types buried in openai's own import chain, not
        # just a clean "package missing" ImportError. Whatever the underlying
        # cause, the actionable fix for the person running this is the same.
        raise LLMNotConfiguredError(
            f"Failed to import the openai package ({type(e).__name__}: {e}). "
            "This is usually a version mismatch between openai/httpx/aiohttp in "
            "your environment, not a problem with this script. Try: "
            "pip install --upgrade openai httpx aiohttp"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMNotConfiguredError(
            "OPENAI_API_KEY environment variable not set. "
            "Run `export OPENAI_API_KEY=\"sk-...\"` in your shell (or load it from a "
            ".env file) before calling this function."
        )

    client = OpenAI(
        api_key=api_key,
        # Accept-Encoding: identity tells the server not to compress the
        # response at all, which sidesteps a response-decompression bug seen
        # in some httpx/openai/brotli version combinations (a TypeError deep
        # in the decoder, surfaced by the SDK as a misleading
        # "APIConnectionError: Connection error" even though the request
        # itself succeeded). If your environment doesn't hit that bug this
        # header is harmless — it just costs a slightly larger response body.
        default_headers={"Accept-Encoding": "identity"},
    )
    MODEL_NAME = "gpt-5.5"  # confirm the exact model/deployment string with your provider

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        # No `temperature` override — reasoning-tier models (this one included,
        # per the BadRequestError this raised when temperature=0.7 was passed)
        # only accept the default value of 1 and reject any other override.
    )
    return response.choices[0].message.content


def mock_trace_generator_llm(prompt: str) -> str:
    """
    Deterministic stand-in so the full pipeline (tools -> prompt -> trace ->
    grounding audit) can be exercised today without an API key. Deliberately
    includes one grounded claim and one fabricated claim, so running this
    through the grounding checker below demonstrably catches the planted
    hallucination rather than just rubber-stamping everything.
    """
    return """<reasoning>
The motif detector identified a proline-rich turn motif, consistent with
this being a pyriform spidroin rather than a dragline (MaSp) protein —
proline-rich turns disrupt extended beta-sheet formation and instead
promote a more flexible, turn-containing backbone appropriate for cement
and attachment-disc function. It also reports a hydrophobicity peak of 9.9
near residue 900, which is outside anything the tools actually returned and
is likely fabricated.
</reasoning>
<answer>
{"strength": 0.6, "toughness": 0.5}
</answer>"""


# ---------------------------------------------------------------------------
# Step 4: assemble the full structured trace
# ---------------------------------------------------------------------------

def assemble_structured_trace(tool_steps: List[TraceStep], llm_completion: str) -> List[TraceStep]:
    import re
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", llm_completion, re.DOTALL)
    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else llm_completion.strip()
    return tool_steps + [TraceStep(step_type="reasoning", text=reasoning_text)]


# ---------------------------------------------------------------------------
# Step 5: end-to-end single-sample pipeline
# ---------------------------------------------------------------------------

@dataclass
class GeneratedTrace:
    sample: SilkSample
    llm_completion: str
    structured_trace: List[TraceStep]
    grounding_report: GroundingReport


def generate_trace_for_sample(
    sample: SilkSample,
    llm_fn: Callable[[str], str] = call_trace_generator_llm,
) -> GeneratedTrace:
    tool_steps = run_tools_as_trace_steps(sample.sequence)
    prompt = build_trace_prompt(sample, tool_steps)
    completion = llm_fn(prompt)
    structured_trace = assemble_structured_trace(tool_steps, completion)
    report = check_trace_grounding(structured_trace)
    return GeneratedTrace(
        sample=sample,
        llm_completion=completion,
        structured_trace=structured_trace,
        grounding_report=report,
    )


def generate_shortcut_trace_for_sample(
    sample: SilkSample,
    llm_fn: Callable[[str], str] = call_trace_generator_llm,
) -> GeneratedTrace:
    """
    Same output shape as generate_trace_for_sample, but the LLM never sees
    the tool outputs (build_shortcut_prompt instead of build_trace_prompt).
    The tools are still run and still assembled into structured_trace, so
    check_trace_grounding scores the completion against real tool output
    the model had no access to — an honest low grounding score, not a
    simulated one, since the model may still happen to state something
    that coincidentally matches (that's a real, legitimate "lucky" case
    the grounding checker should catch either way).
    """
    tool_steps = run_tools_as_trace_steps(sample.sequence)
    prompt = build_shortcut_prompt(sample)
    completion = llm_fn(prompt)
    structured_trace = assemble_structured_trace(tool_steps, completion)
    report = check_trace_grounding(structured_trace)
    return GeneratedTrace(
        sample=sample,
        llm_completion=completion,
        structured_trace=structured_trace,
        grounding_report=report,
    )


# ---------------------------------------------------------------------------
# Step 6: pilot batch generation + audit
# ---------------------------------------------------------------------------

def generate_pilot_batch(
    samples: List[SilkSample],
    llm_fn: Callable[[str], str] = call_trace_generator_llm,
) -> Dict[str, Any]:
    """
    Runs the full pipeline over a pilot batch and returns both the
    individual results and the aggregate audit summary — this is the
    Stage 1 deliverable: a hallucination-rate number to decide whether the
    trace-generation prompt needs iteration before scaling up.
    """
    generated = [generate_trace_for_sample(s, llm_fn) for s in samples]
    traces_only = [g.structured_trace for g in generated]
    summary = audit_trace_batch(traces_only)
    return {"generated": generated, "audit_summary": summary}


def print_pilot_batch_report(result: Dict[str, Any]) -> None:
    summary = result["audit_summary"]
    print(f"Pilot batch: {summary.n_traces} traces")
    print(f"Mean grounding score: {summary.mean_grounding_score:.2f}")
    print(f"Fully grounded fraction: {summary.fully_grounded_fraction:.0%}")
    print(f"Claims by kind (direct): {summary.claims_by_kind}")
    print(f"Ungrounded by kind (direct): {summary.ungrounded_by_kind}")
    print(f"Contrastive mentions (informational, not scored): {summary.n_contrastive_mentions}")
    if summary.contrastive_ungrounded_fraction is not None:
        print(f"Contrastive ungrounded fraction: {summary.contrastive_ungrounded_fraction:.0%}")
    print("Worst-scoring traces (indices into `generated`):", summary.worst_traces)


# ---------------------------------------------------------------------------
# Example usage (runs fully end-to-end with the mock LLM, no API key needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = SilkSample(
        sequence=(
            "SSSAYSGTSTGGSSVSQSQPIISSAPVYFNAQTLTSSLASSLQSDRALNFISSGQLSASDVS"
            "TSVSSAVAQTLGISQSSVQNIISQQMSSVRTGASSSSVSQAIANAVSSAVQASGAATPGQEQ"
            "SIAQRVYSSISTYLSQLISQRTAPAPAPAPRPAPMPAPAPRPAPMPAPAPRPRPAPAPRPAP"
        ),
        family="Araneidae",
        genus="Eriophora",
        species="pustulosa",
        protein_category="PySp",
        ground_truth={"strength": 0.5, "toughness": 0.6},
    )

    result = generate_pilot_batch([sample], llm_fn=mock_trace_generator_llm)
    print_pilot_batch_report(result)

    print("\n=== Per-claim detail for the one trace ===")
    g = result["generated"][0]
    for c in g.grounding_report.claims:
        status = "OK " if c.grounded else "FAIL"
        print(f"  [{status}] ({c.kind}, {c.role}) {c.raw_text!r} -> {c.evidence}")

    print(f"\nTrying the real LLM call (uses OPENAI_API_KEY if set, otherwise fails clearly):")
    try:
        real_result = generate_trace_for_sample(sample)
        print("  SUCCESS — real completion received:\n")
        print("  " + "-" * 70)
        print("\n".join(f"  {line}" for line in real_result.llm_completion.splitlines()))
        print("  " + "-" * 70)
        print(f"\n  Grounding score (direct claims): {real_result.grounding_report.grounding_score:.2f}")
        print(f"  Contrastive grounding score: {real_result.grounding_report.contrastive_grounding_score}")
        print("\n  Per-claim detail:")
        for c in real_result.grounding_report.claims:
            status = "OK " if c.grounded else "FAIL"
            print(f"    [{status}] ({c.kind}, {c.role}) {c.raw_text!r} -> {c.evidence}")
    except LLMNotConfiguredError as e:
        print(f"  LLMNotConfiguredError: {e}")
