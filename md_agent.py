"""
MD Simulation Agent: described material -> CG simulation -> molecular properties
==================================================================================

Takes a free-text description of a material ("MaSp1-like dragline repeat,
4 chains at 2 mM, 150 mM NaCl, 300 K", or "a 60-bead polymer in poor
solvent") and runs a tool-using agent loop that:

    description
      -> lookup_reference_sequence / analyze_sequence   (grounded inputs; bio_tools.py)
      -> setup_simulation      (validated MaterialSpec -> md_engine.ParticleSystem)
      -> run_simulation        (Langevin MD: OpenMM if installed, else numpy)
      -> compute_properties    (md_analysis.py, with block-averaged error bars)
      -> extend_simulation     (if diagnostics say "not equilibrated" and budget allows)
      -> final report          (LLM narrative + code-assembled numbers)
      -> grounding audit       (grounding_checker.py: are the narrative's numbers
                                actually in the tool outputs?)

Design decisions that matter:

  * **The LLM never supplies numbers or sequences.** Every reported property
    in the final JSON comes from `compute_properties`, not from the model's
    prose. A sequence passed to `setup_simulation` must appear verbatim in
    the user's description or the built-in reference library (optionally
    tandem-repeated) — otherwise the call is rejected. This is the same
    "don't let the generator invent the inputs" rule as the trace pipeline.
  * **Tool errors are returned to the LLM, not raised**, so it can fix a bad
    spec. `LLMNotConfiguredError` (missing key/package) still fails fast.
  * **Hard budgets** on total MD steps, beads, and agent turns — an agent
    that can launch simulations must not be able to launch unbounded ones.
  * `--use-mock` runs a deterministic rule-based policy through the *same*
    loop and the same OpenAI-format messages, so everything except the API
    call is exercised without a key.

Usage:
    python3 md_agent.py "<material description>" [--use-mock] [--backend auto|numpy|openmm]
                        [--max-md-steps N] [--out report.json]
    python3 md_agent.py                      # mock demo on two built-in examples
"""

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from bio_tools import PYSP_ERIOPHORA_PUSTULOSA, SYNTHETIC_MASP_LIKE, TOOL_REGISTRY
from grounding_checker import TraceStep, check_trace_grounding
from md_analysis import AVAILABLE_PROPERTIES, compute_properties
from md_engine import (HPS_PARAMS, BackendNotAvailableError, MaterialSpec, SpecError,
                       Trajectory, build_system, run_md)
from trace_generation import LLMNotConfiguredError

MODEL_NAME = "gpt-5.5"  # confirm the exact model/deployment string with your provider

# Each reference entry is a real or explicitly-synthetic sequence already used
# (and documented) in bio_tools.py. Extend with sequences you have provenance for.
REFERENCE_SEQUENCES = {
    "synthetic_masp_like": {
        "sequence": SYNTHETIC_MASP_LIKE,
        "description": "Synthetic MaSp-like dragline repeat (GGX glycine-rich + poly-Ala blocks), "
                       "3 repeats, 105 aa. Constructed positive control, not a natural sequence.",
    },
    "pysp_eriophora_pustulosa": {
        "sequence": PYSP_ERIOPHORA_PUSTULOSA,
        "description": "Eriophora pustulosa pyriform spidroin (PySp) repetitive region, 372 aa. "
                       "Real sequence; attachment-disc silk, proline-rich, not dragline.",
    },
}

MODEL_CAVEATS = {
    "protein": [
        "HPS coarse-grained model: one bead per residue, implicit solvent. It captures chain "
        "dimensions, compaction, and sticker-driven self-association of disordered chains.",
        "HPS with the Kapcha-Rossky scale is known to over-compact many IDPs relative to "
        "experiment (later variants such as HPS-Urry rescale this); trust trends between "
        "sequences/conditions more than absolute Rg values.",
        "It cannot represent secondary structure (beta-sheet nanocrystals, helices) or hydrogen "
        "bonding, so it says nothing direct about crystalline-domain mechanics or fiber strength.",
        "Dynamics are Langevin model time; diffusion/relaxation values depend on the chosen "
        "friction and are not real-time quantities.",
    ],
    "polymer": [
        "Generic bead-spring model in reduced LJ units (sigma, epsilon, tau); maps to a real "
        "polymer only through a coarse-graining choice of bead size and energy scale.",
        "Chemistry enters only through solvent quality (attraction strength); no tacticity, "
        "torsional stiffness, or specific interactions.",
    ],
}


class BudgetExceededError(RuntimeError):
    """Raised when a tool call would exceed the agent's MD-step budget."""


# ---------------------------------------------------------------------------
# Workspace: tool implementations (state lives here, not in the LLM)
# ---------------------------------------------------------------------------

_SEQ_RE = re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY]{15,}\b")


def extract_user_sequences(description: str) -> Dict[str, str]:
    """Amino-acid strings (>= 15 uppercase standard residues) the user wrote in the description."""
    return {f"user_sequence_{i + 1}": m for i, m in enumerate(_SEQ_RE.findall(description))}


@dataclass
class RunRecord:
    system_id: str
    trajectories: List[Trajectory] = field(default_factory=list)

    def merged(self) -> Trajectory:
        """Concatenates segments (extensions continue from the previous segment's end)."""
        if len(self.trajectories) == 1:
            return self.trajectories[0]
        t0 = self.trajectories[0]
        times, offset = [], 0.0
        for tr in self.trajectories:
            times.append(tr.times + offset)
            offset += tr.n_steps * tr.dt
        last = self.trajectories[-1]
        return Trajectory(
            frames=np.concatenate([t.frames for t in self.trajectories]),
            times=np.concatenate(times),
            potential_energy=np.concatenate([t.potential_energy for t in self.trajectories]),
            kinetic_temperature=np.concatenate([t.kinetic_temperature for t in self.trajectories]),
            final_positions=last.final_positions, final_velocities=last.final_velocities,
            backend=t0.backend, wall_seconds=sum(t.wall_seconds for t in self.trajectories),
            n_steps=sum(t.n_steps for t in self.trajectories), dt=t0.dt)


class MDWorkspace:
    def __init__(self, description: str, backend: str = "auto", max_md_steps: int = 400_000,
                 frames_per_run: int = 200):
        self.description = description
        self.backend = backend
        self.max_md_steps = max_md_steps
        self.frames_per_run = frames_per_run
        self.steps_used = 0
        self.user_sequences = extract_user_sequences(description)
        self.systems: Dict[str, Any] = {}
        self.runs: Dict[str, RunRecord] = {}
        self.last_spec: Optional[MaterialSpec] = None
        self.last_properties: Optional[Dict[str, Any]] = None
        self.last_run_id: Optional[str] = None

    # --- tools -------------------------------------------------------------

    def lookup_reference_sequence(self, name: Optional[str] = None) -> Dict[str, Any]:
        entries = {k: {"description": v["description"], "length": len(v["sequence"])}
                   for k, v in REFERENCE_SEQUENCES.items()}
        entries.update({k: {"description": "sequence given verbatim in the user's description",
                            "length": len(v)} for k, v in self.user_sequences.items()})
        if not name:
            return {"available": entries}
        seq = self._resolve_named(name)
        if seq is None:
            return {"error": f"unknown reference {name!r}", "available": entries}
        return {"name": name, "sequence": seq, **entries[name]}

    def analyze_sequence(self, sequence_name: str) -> Dict[str, Any]:
        seq = self._resolve_named(sequence_name)
        if seq is None:
            return {"error": f"unknown sequence {sequence_name!r}; use lookup_reference_sequence()"}
        out: Dict[str, Any] = {"sequence_name": sequence_name, "length": len(seq)}
        for tool, fn in TOOL_REGISTRY.items():
            res = fn(seq)
            # drop per-residue arrays: too long for the context window, not needed for setup
            out[tool] = {k: v for k, v in res.items()
                         if not (isinstance(v, list) and len(v) > 20)}
        return out

    def setup_simulation(self, model: str, sequence_name: Optional[str] = None,
                         sequence: Optional[str] = None, repeat_count: int = 1,
                         **spec_fields) -> Dict[str, Any]:
        if model == "protein":
            unit = self._resolve_named(sequence_name) if sequence_name else None
            if sequence_name and unit is None:
                raise SpecError(f"unknown sequence_name {sequence_name!r}")
            if unit is None and sequence:
                unit = self._ground_raw_sequence(sequence)
            if unit is None:
                raise SpecError("protein model needs sequence_name (preferred) or a sequence "
                                "that appears verbatim in the description/reference library")
            if not (1 <= int(repeat_count) <= 50):
                raise SpecError("repeat_count must be in [1, 50]")
            spec_fields["sequence"] = unit * int(repeat_count)
        allowed = set(MaterialSpec.__dataclass_fields__) - {"model", "sequence"}
        unknown = sorted(set(spec_fields) - allowed - {"sequence"})
        if unknown:
            raise SpecError(f"unknown spec fields {unknown}; allowed: {sorted(allowed)}")
        spec = MaterialSpec(model=model, **spec_fields)
        system = build_system(spec)   # validates
        sid = f"sys_{uuid.uuid4().hex[:6]}"
        self.systems[sid] = system
        self.last_spec = spec
        summary = {
            "system_id": sid, "model": model, "units": system.units, "n_beads": system.n,
            "n_chains": spec.n_chains,
            "chain_length": len(spec.sequence) if model == "protein" else spec.chain_length,
            "box_length": system.box_length, "periodic": system.box_length is not None,
            "timestep": system.dt, "friction": system.friction,
        }
        if model == "protein":
            summary.update(temperature_K=spec.temperature_K, ionic_strength_M=spec.ionic_strength_M,
                           debye_length_nm=system.debye_length,
                           net_charge_per_chain=float(system.charge[system.chain_ids == 0].sum()))
        else:
            summary.update(attraction_lambda=float(system.lam[0]), kT=system.kT)
        summary["recommended_n_steps"] = 100_000 if model == "protein" else 50_000
        summary["md_steps_remaining"] = self.max_md_steps - self.steps_used
        return summary

    def run_simulation(self, system_id: str, n_steps: Optional[int] = None,
                       seed: int = 0) -> Dict[str, Any]:
        system = self._system(system_id)
        n_steps = int(n_steps or (100_000 if system.length_unit == "nm" else 50_000))
        self._charge_budget(n_steps)
        interval = max(1, n_steps // self.frames_per_run)
        traj = run_md(system, n_steps, report_interval=interval, backend=self.backend, seed=seed)
        rid = f"run_{uuid.uuid4().hex[:6]}"
        self.runs[rid] = RunRecord(system_id=system_id, trajectories=[traj])
        self.last_run_id = rid
        return self._run_summary(rid)

    def extend_simulation(self, run_id: str, n_steps: int) -> Dict[str, Any]:
        rec = self._run(run_id)
        system = self.systems[rec.system_id]
        n_steps = int(n_steps)
        self._charge_budget(n_steps)
        last = rec.trajectories[-1]
        interval = max(1, n_steps // self.frames_per_run)
        traj = run_md(system, n_steps, report_interval=interval, backend=self.backend,
                      seed=len(rec.trajectories), positions=last.final_positions,
                      velocities=last.final_velocities, minimize=False)
        rec.trajectories.append(traj)
        return self._run_summary(run_id)

    def compute_properties(self, run_id: str, properties: Optional[List[str]] = None,
                           burn_in_fraction: float = 0.2) -> Dict[str, Any]:
        rec = self._run(run_id)
        system = self.systems[rec.system_id]
        props = compute_properties(rec.merged(), system, properties, burn_in_fraction)
        props["run_id"] = run_id
        self.last_properties = props
        self.last_run_id = run_id
        return props

    # --- helpers -----------------------------------------------------------

    def _resolve_named(self, name: str) -> Optional[str]:
        if name in self.user_sequences:
            return self.user_sequences[name]
        if name in REFERENCE_SEQUENCES:
            return REFERENCE_SEQUENCES[name]["sequence"]
        return None

    def _ground_raw_sequence(self, seq: str) -> str:
        """A raw sequence is accepted only if it's verbatim in the description or the library."""
        s = "".join(seq.split()).upper()
        haystacks = [self.description.upper()] + [v["sequence"] for v in REFERENCE_SEQUENCES.values()]
        if any(s in h for h in haystacks):
            return s
        raise SpecError("sequence not found verbatim in the user's description or the reference "
                        "library. Do not compose or recall sequences from memory; use "
                        "sequence_name, or a repeat unit the user gave plus repeat_count")

    def _system(self, sid: str):
        if sid not in self.systems:
            raise SpecError(f"unknown system_id {sid!r}; call setup_simulation first")
        return self.systems[sid]

    def _run(self, rid: str) -> RunRecord:
        if rid not in self.runs:
            raise SpecError(f"unknown run_id {rid!r}")
        return self.runs[rid]

    def _charge_budget(self, n_steps: int) -> None:
        if n_steps < 1000:
            raise SpecError("n_steps must be >= 1000")
        if self.steps_used + n_steps > self.max_md_steps:
            raise BudgetExceededError(
                f"would exceed the MD-step budget ({self.steps_used} used + {n_steps} requested "
                f"> {self.max_md_steps}); report with what you have")
        self.steps_used += n_steps

    def _run_summary(self, rid: str) -> Dict[str, Any]:
        rec = self.runs[rid]
        tr = rec.merged()
        return {"run_id": rid, "system_id": rec.system_id, "backend": tr.backend,
                "total_steps": tr.n_steps, "simulated_time": round(tr.n_steps * tr.dt, 3),
                "n_frames": len(tr.frames), "wall_seconds": round(tr.wall_seconds, 1),
                "mean_kinetic_temperature_ratio": round(float(np.mean(tr.kinetic_temperature)), 3),
                "md_steps_remaining": self.max_md_steps - self.steps_used}

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = TOOLS.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(self, **args)
        except (SpecError, BudgetExceededError, BackendNotAvailableError, ValueError, TypeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "lookup_reference_sequence": MDWorkspace.lookup_reference_sequence,
    "analyze_sequence": MDWorkspace.analyze_sequence,
    "setup_simulation": MDWorkspace.setup_simulation,
    "run_simulation": MDWorkspace.run_simulation,
    "extend_simulation": MDWorkspace.extend_simulation,
    "compute_properties": MDWorkspace.compute_properties,
}

TOOL_SCHEMAS = [
    {"name": "lookup_reference_sequence",
     "description": "List available sequences (built-in library + sequences the user wrote in the "
                    "description, named user_sequence_N), or fetch one by name.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}},
    {"name": "analyze_sequence",
     "description": "Run bioinformatics analyses (motifs, hydropathy, charge, beta propensity, "
                    "repeats, disorder) on a named sequence.",
     "parameters": {"type": "object", "properties": {"sequence_name": {"type": "string"}},
                    "required": ["sequence_name"]}},
    {"name": "setup_simulation",
     "description": "Build a coarse-grained MD system. model='protein' (HPS, one bead/residue, real "
                    "units) needs sequence_name; model='polymer' (bead-spring, reduced LJ units) "
                    "needs chain_length and solvent_quality.",
     "parameters": {"type": "object", "properties": {
         "model": {"type": "string", "enum": ["protein", "polymer"]},
         "name": {"type": "string", "description": "short label for the material"},
         "sequence_name": {"type": "string"},
         "repeat_count": {"type": "integer", "description": "tandem repeats of the named sequence"},
         "chain_length": {"type": "integer"},
         "solvent_quality": {"type": "string", "enum": ["good", "theta", "poor"]},
         "attraction_lambda": {"type": "number", "description": "polymer only, 0..1, overrides solvent_quality"},
         "n_chains": {"type": "integer"},
         "temperature_K": {"type": "number"},
         "ionic_strength_M": {"type": "number"},
         "histidine_charge": {"type": "number"},
         "concentration_mM": {"type": "number"},
         "box_length": {"type": "number"},
         "seed": {"type": "integer"}},
         "required": ["model"]}},
    {"name": "run_simulation",
     "description": "Energy-minimize then run Langevin MD on a system. Returns a run_id.",
     "parameters": {"type": "object", "properties": {
         "system_id": {"type": "string"}, "n_steps": {"type": "integer"}, "seed": {"type": "integer"}},
         "required": ["system_id"]}},
    {"name": "extend_simulation",
     "description": "Continue an existing run for more steps (use when diagnostics.looks_equilibrated is false).",
     "parameters": {"type": "object", "properties": {
         "run_id": {"type": "string"}, "n_steps": {"type": "integer"}},
         "required": ["run_id", "n_steps"]}},
    {"name": "compute_properties",
     "description": "Compute molecular-level properties with block-averaged standard errors, plus "
                    f"equilibration diagnostics. Available: {AVAILABLE_PROPERTIES}.",
     "parameters": {"type": "object", "properties": {
         "run_id": {"type": "string"},
         "properties": {"type": "array", "items": {"type": "string", "enum": AVAILABLE_PROPERTIES}},
         "burn_in_fraction": {"type": "number"}},
         "required": ["run_id"]}},
]

AGENT_SYSTEM_PROMPT = f"""You are a molecular simulation agent. Given a described material, you set up \
and run coarse-grained molecular dynamics with the provided tools and report molecular-level properties.

Models available:
- protein: HPS model (Dignon et al. 2018), one bead per residue, implicit solvent, real units \
(nm, ps, K, M). Good for chain dimensions, compaction, and self-association of disordered/repetitive \
proteins (e.g. spidroin repeats). CANNOT represent secondary structure, beta-sheet crystals, or H-bonds.
- polymer: generic bead-spring homopolymer in reduced LJ units; solvent quality good/theta/poor.

Rules:
1. Never write a sequence from memory. Use lookup_reference_sequence to see what is available \
(including sequences the user typed, named user_sequence_N) and pass sequence_name. If the user gave a \
repeat unit and a count, use repeat_count. If no grounded sequence exists for a protein the user names, \
say so and use the closest available option or the polymer model, explaining the substitution.
2. Map the description onto spec fields (temperature, salt, number of chains, concentration). State any \
assumption you had to make.
3. After compute_properties, check diagnostics.looks_equilibrated. If false and budget remains, call \
extend_simulation (e.g. same number of steps again) and recompute. Do not loop more than twice.
4. Final answer (no tool call): a concise report. Quote numbers exactly as the tool returned them, with \
units and +/- sem. Interpret the Flory exponent (about 0.33 collapsed, 0.5 theta/ideal, 0.59 expanded), \
contacts, and clustering. Close with the model's limitations relevant to the user's question. \
Do not state any number that is not in a tool output.
Total MD-step budget and per-call costs are reported in tool outputs as md_steps_remaining."""


# ---------------------------------------------------------------------------
# Policies: decide the next assistant message given the conversation
# ---------------------------------------------------------------------------

Policy = Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]]


def openai_policy(messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Real LLM policy via an OpenAI-compatible client (same setup and gotchas as
    trace_generation.call_trace_generator_llm — see CLAUDE.md)."""
    import os
    try:
        from openai import OpenAI
    except Exception as e:
        raise LLMNotConfiguredError(
            f"Failed to import the openai package ({type(e).__name__}: {e}). "
            "Try: pip install --upgrade openai httpx") from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMNotConfiguredError(
            "OPENAI_API_KEY environment variable not set (or run with --use-mock).")
    client = OpenAI(api_key=api_key, default_headers={"Accept-Encoding": "identity"})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=[{"type": "function", "function": s} for s in tool_schemas],
        # No temperature override: this model class rejects anything but the default.
    )
    msg = response.choices[0].message
    out: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [{"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name,
                                           "arguments": tc.function.arguments}}
                             for tc in msg.tool_calls]
    return out


def parse_description_rules(description: str) -> Dict[str, Any]:
    """Keyword/regex parser used by the mock policy. Deliberately simple —
    the real policy is the LLM; this exists so the loop runs without a key."""
    text = description.lower()
    spec: Dict[str, Any] = {}
    seqs = extract_user_sequences(description)
    polymer_words = ("polymer", "bead-spring", "homopolymer", "polyethylene", "polystyrene", "-mer")
    if seqs:
        spec.update(model="protein", sequence_name=next(iter(seqs)))
    elif any(w in text for w in ("pysp", "pyriform")):
        spec.update(model="protein", sequence_name="pysp_eriophora_pustulosa")
    elif any(w in text for w in ("silk", "spidroin", "masp", "dragline", "protein")):
        spec.update(model="protein", sequence_name="synthetic_masp_like")
    elif any(w in text for w in polymer_words):
        spec.update(model="polymer")
    else:
        spec.update(model="polymer")
    if m := re.search(r"(\d+)\s*(?:x|×|repeats?|copies of the repeat)", text):
        spec["repeat_count"] = int(m.group(1))
    if m := re.search(r"(\d+)\s*(?:chains|molecules|copies)", text):
        spec["n_chains"] = int(m.group(1))
    if spec["model"] == "protein":
        if m := re.search(r"(\d+(?:\.\d+)?)\s*k\b", text):
            spec["temperature_K"] = float(m.group(1))
        elif m := re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*c|deg(?:rees)?\s*c|c\b)", text):
            spec["temperature_K"] = float(m.group(1)) + 273.15
        if m := re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:nacl|kcl|salt|ionic)", text):
            spec["ionic_strength_M"] = float(m.group(1)) / 1000.0
        elif m := re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:nacl|kcl|salt|ionic)", text):
            spec["ionic_strength_M"] = float(m.group(1))
        if m := re.search(r"(\d+(?:\.\d+)?)\s*mm\b(?!\s*(?:nacl|kcl|salt|ionic))", text):
            spec["concentration_mM"] = float(m.group(1))
    else:
        m = re.search(r"(\d+)\s*-?\s*(?:mer|bead|monomer|segment)", text)
        spec["chain_length"] = int(m.group(1)) if m else 50
        spec["solvent_quality"] = ("poor" if any(w in text for w in ("poor", "bad", "collapse", "melt"))
                                   else "theta" if "theta" in text else "good")
    return spec


def _fmt(d: Optional[Dict[str, Any]], key: str = "mean") -> str:
    if not d:
        return "n/a"
    return f"{d[key]} ± {d['sem']}" if "sem" in d else str(d[key])


def template_report(props: Dict[str, Any], setup: Dict[str, Any], spec_note: str) -> str:
    """Deterministic narrative for the mock policy, quoting tool outputs verbatim."""
    L = props["length_unit"]
    lines = [f"Simulated {setup['n_chains']} chain(s) of {setup['chain_length']} beads "
             f"({setup['model']} model, {setup['units']}). {spec_note}"]
    lines.append(f"Radius of gyration: {_fmt(props.get('radius_of_gyration'))} {L}; "
                 f"end-to-end distance: {_fmt(props.get('end_to_end_distance'))} {L} "
                 f"(Ree^2/Rg^2 = {props.get('ree2_over_rg2')}).")
    fl = props.get("flory_exponent")
    if fl:
        nu = fl["nu"]
        regime = ("collapsed/globular" if nu < 0.42 else "near-ideal (theta-like)" if nu < 0.54
                  else "expanded (good-solvent-like)")
        lines.append(f"Flory scaling exponent nu = {nu}, i.e. {regime}.")
    if props.get("asphericity"):
        lines.append(f"Shape anisotropy kappa^2 = {_fmt(props['asphericity'])}; "
                     f"persistence length = {props.get('persistence_length')} {L}.")
    c = props.get("contacts")
    if c:
        lines.append(f"Non-local intrachain contact fraction = {c['intrachain_contact_fraction']}.")
        if "largest_cluster_fraction" in c:
            lines.append(f"Interchain contacts per chain = {c['interchain_contacts_per_chain']}; "
                         f"largest cluster holds {c['largest_cluster_fraction']} of chains.")
        top = c.get("top_interchain_contact_pairs") or c.get("top_intrachain_contact_pairs")
        if top:
            lines.append("Most frequent contacting residue pairs: "
                         + ", ".join(f"{p['pair']} ({p['share']})" for p in top[:3]) + ".")
    d = props["diagnostics"]
    lines.append("Equilibration: " + ("looks equilibrated" if d["looks_equilibrated"] else
                                      "NOT clearly equilibrated — treat values as provisional")
                 + f" (Rg drift/sem = {d['rg_drift_over_sem']}, relative sem = {d['rg_relative_sem']}).")
    return "\n".join(lines)


def make_mock_policy(description: str) -> Policy:
    """Rule-based stand-in for the LLM that walks the same tool sequence an
    LLM would (and emits OpenAI-format messages, so the loop is identical)."""
    plan = parse_description_rules(description)
    state = {"phase": "start", "extensions": 0, "setup": None, "run_id": None}
    counter = iter(range(1, 10_000))

    def call(name: str, **args) -> Dict[str, Any]:
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{next(counter)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}

    def last_result(messages) -> Dict[str, Any]:
        return json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else {}

    def policy(messages, _schemas):
        res = last_result(messages)
        ph = state["phase"]
        if "error" in res:
            return {"role": "assistant", "content": f"Stopping: tool error: {res['error']}"}
        if ph == "start":
            state["phase"] = "analyzed"
            if plan["model"] == "protein":
                return call("analyze_sequence", sequence_name=plan["sequence_name"])
            ph = "analyzed"
        if ph == "analyzed":
            state["phase"] = "setup"
            return call("setup_simulation", **plan)
        if ph == "setup":
            state["setup"] = res
            state["phase"] = "ran"
            return call("run_simulation", system_id=res["system_id"],
                        n_steps=res["recommended_n_steps"])
        if ph == "ran":
            state["run_id"] = res["run_id"]
            state["phase"] = "computed"
            return call("compute_properties", run_id=res["run_id"])
        if ph == "computed":
            remaining = state["setup"]["recommended_n_steps"]
            if (not res["diagnostics"]["looks_equilibrated"] and state["extensions"] < 2):
                state["extensions"] += 1
                state["phase"] = "ran"
                return call("extend_simulation", run_id=state["run_id"], n_steps=remaining)
            note = ("Spec parsed by the rule-based mock policy: "
                    + ", ".join(f"{k}={v}" for k, v in plan.items()) + ".")
            return {"role": "assistant", "content": template_report(res, state["setup"], note)}
        return {"role": "assistant", "content": "Done."}

    # an extension that hits the budget returns an error; report what we have instead
    def guarded(messages, schemas):
        res = last_result(messages)
        if "error" in res and "budget" in res["error"] and state["run_id"]:
            state["phase"] = "computed"
            state["extensions"] = 99
            return call("compute_properties", run_id=state["run_id"])
        return policy(messages, schemas)
    return guarded


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    description: str
    spec: Optional[Dict[str, Any]]
    properties: Optional[Dict[str, Any]]
    report: str
    model_caveats: List[str]
    grounding: Dict[str, Any]
    tool_log: List[Dict[str, Any]]
    md_steps_used: int
    wall_seconds: float


def _grounding_audit(tool_log: List[Dict[str, Any]], report: str) -> Dict[str, Any]:
    """Reuses grounding_checker: every number in the final narrative should
    trace back to some tool output. Not yet validated on real LLM narratives
    of MD results (see CLAUDE.md's warning about trusting this score)."""
    steps = [TraceStep(step_type="tool_call", tool_name=e["tool"], tool_input=e["args"],
                       tool_output=e["result"]) for e in tool_log]
    steps.append(TraceStep(step_type="reasoning", text=report))
    gr = check_trace_grounding(steps)
    return {"grounding_score": gr.grounding_score, "n_direct_claims": gr.total_claims,
            "ungrounded_claims": [c.raw_text for c in gr.direct_claims if not c.grounded]}


def run_agent(description: str, policy: Policy, workspace: MDWorkspace,
              max_turns: int = 14, verbose: bool = True) -> AgentResult:
    t0 = time.time()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Material description:\n{description}"},
    ]
    tool_log: List[Dict[str, Any]] = []
    report = ""
    for _ in range(max_turns):
        msg = policy(messages, TOOL_SCHEMAS)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            report = msg.get("content", "")
            break
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                args, result = {}, {"error": f"arguments were not valid JSON: {e}"}
            else:
                if verbose:
                    shown = {k: (v[:20] + "..." if isinstance(v, str) and len(v) > 24 else v)
                             for k, v in args.items()}
                    print(f"  -> {name}({shown})", flush=True)
                result = workspace.dispatch(name, args)
            if verbose and "error" in result:
                print(f"     error: {result['error']}", flush=True)
            tool_log.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, default=str)})
    else:
        report = "(agent hit max_turns without a final report)"

    spec = asdict(workspace.last_spec) if workspace.last_spec else None
    return AgentResult(
        description=description, spec=spec, properties=workspace.last_properties, report=report,
        model_caveats=MODEL_CAVEATS.get(spec["model"], []) if spec else [],
        grounding=_grounding_audit(tool_log, report), tool_log=tool_log,
        md_steps_used=workspace.steps_used, wall_seconds=round(time.time() - t0, 1))


def simulate_material(description: str, use_mock: bool = False, backend: str = "auto",
                      max_md_steps: int = 400_000, verbose: bool = True) -> AgentResult:
    ws = MDWorkspace(description, backend=backend, max_md_steps=max_md_steps)
    policy = make_mock_policy(description) if use_mock else openai_policy
    return run_agent(description, policy, ws, verbose=verbose)


def _print_result(r: AgentResult) -> None:
    print("\n=== Report ===\n" + r.report)
    print("\n=== Model caveats ===")
    for c in r.model_caveats:
        print(" - " + c)
    g = r.grounding
    print(f"\nGrounding audit: {g['grounding_score']:.2f} over {g['n_direct_claims']} numeric/motif "
          f"claims; ungrounded: {g['ungrounded_claims'] or 'none'}")
    print(f"MD steps used: {r.md_steps_used}; wall time {r.wall_seconds}s")


DEMO_DESCRIPTIONS = [
    "Single chain of a MaSp-like spider dragline silk repeat in water at 300 K with 150 mM NaCl.",
    "A 60-mer flexible homopolymer in poor solvent.",
]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("description", nargs="?", help="free-text material description")
    ap.add_argument("--use-mock", action="store_true", help="rule-based policy, no API key needed")
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "openmm"])
    ap.add_argument("--max-md-steps", type=int, default=400_000)
    ap.add_argument("--out", help="write the full result (incl. tool log) as JSON")
    a = ap.parse_args(argv)

    descriptions = [a.description] if a.description else DEMO_DESCRIPTIONS
    use_mock = a.use_mock or not a.description
    results = []
    for desc in descriptions:
        print(f"\n### {desc}\n(policy: {'mock' if use_mock else MODEL_NAME})")
        try:
            r = simulate_material(desc, use_mock=use_mock, backend=a.backend,
                                  max_md_steps=a.max_md_steps)
        except LLMNotConfiguredError as e:
            print(f"LLM not configured: {e}", file=sys.stderr)
            return 2
        _print_result(r)
        results.append(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump([asdict(r) for r in results] if len(results) > 1 else asdict(results[0]),
                      f, indent=2, default=str)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
