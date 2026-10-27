"""
Stage A curriculum: sequence reading with exact, checkable answers
==================================================================

The end goal is a model that predicts silk properties from sequence alone,
with no tools at inference. That model first has to *read* a spidroin
correctly: find poly-Ala blocks, segment repeat units, count motifs.
Language models are unreliable at exactly this (counting and indexing over
hundreds of residues), and every later stage depends on it.

This module turns real spidroin sequences (Spider Silkome, via
`silkome_download.py`) into question/answer tasks whose gold answers are
computed deterministically, plus a checker that scores a model's answer
field by field. The checker's score is a verifiable reward usable for SFT
filtering or GRPO — no judge model involved.

Task types (each asks for a JSON object with fixed keys):
  motifs       : length + overlapping counts of silk motifs (GGX, GPGXX, ...)
  composition  : percent composition of selected residues
  polyala      : poly-Ala blocks (A{4,}): count, longest, 1-indexed spans
  repeats      : crystalline blocks ([SAV]{6,}, as in Arakawa et al. 2022),
                 repeat-region length, amorphous:crystalline length ratio
  compare      : which of two sequences has more of a property

Difficulty levels by sequence length: L1 <=150, L2 <=400, L3 <=1000, L4 >1000.
Splits are by genus (hash-based, deterministic), so test sequences come
from genera never seen in training.

Usage:
  python3 sequence_reading_curriculum.py                  # build + self-test
  python3 sequence_reading_curriculum.py --eval-llm 20    # score 20 test tasks with
                                                          # trace_generation's LLM client
                                                          # (makes real API calls)
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
GENES_PATH = os.path.join(HERE, "silkome_data", "genes.jsonl")
OUT_DIR = os.path.join(HERE, "curriculum")

MOTIFS = {  # name -> regex for overlapping occurrences (lookahead)
    "GGX": r"(?=GG[A-Z])",
    "GPGXX": r"(?=GPG[A-Z]{2})",
    "GGY": r"(?=GGY)",
    "QQ": r"(?=QQ)",
    "YGQGG": r"(?=YGQGG)",
    "AGQG": r"(?=AGQG)",
    "GA": r"(?=GA)",
}
COMP_RESIDUES = ["A", "G", "S", "Q", "P", "Y"]
LEVELS = [(150, "L1"), (400, "L2"), (1000, "L3"), (10 ** 9, "L4")]
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
FLOAT_TOL = {"pct": 0.5, "ratio": 0.05}  # absolute tolerances


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _spans(pattern: str, seq: str) -> List[List[int]]:
    """1-indexed inclusive [start, end] spans of non-overlapping regex matches."""
    return [[m.start() + 1, m.end()] for m in re.finditer(pattern, seq)]


def analyze_sequence(seq: str) -> Dict[str, Any]:
    """Every quantity a task can ask about, computed exactly."""
    n = len(seq)
    polya = _spans(r"A{4,}", seq)
    crys = _spans(r"[SAV]{6,}", seq)
    region_hits = _spans(r"[SAV]{5,}", seq)
    if len(region_hits) >= 2:
        r0, r1 = region_hits[0][0], region_hits[-1][1]
        region_len = r1 - r0 + 1
        crys_in_region = sum(e - s + 1 for s, e in crys if s >= r0 and e <= r1)
        ratio = round((region_len - crys_in_region) / crys_in_region, 2) if crys_in_region else None
    else:
        region_len, ratio = 0, None
    return {
        "length": n,
        "motif_counts": {k: len(re.findall(p, seq)) for k, p in MOTIFS.items()},
        "composition_pct": {a: round(100 * seq.count(a) / n, 1) for a in COMP_RESIDUES},
        "polyala_count": len(polya),
        "polyala_longest": max((e - s + 1 for s, e in polya), default=0),
        "polyala_spans": polya,
        "crystalline_count": len(crys),
        "crystalline_residues": sum(e - s + 1 for s, e in crys),
        "repeat_region_length": region_len,
        "amorphous_to_crystalline": ratio,
    }


def level_of(seq: str) -> str:
    return next(name for cutoff, name in LEVELS if len(seq) <= cutoff)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

DEFINITIONS = (
    "Definitions: positions are 1-indexed and inclusive. Motif counts include "
    "overlapping occurrences (X = any residue). A poly-Ala block is a maximal run "
    "of 4 or more consecutive A. A crystalline block is a maximal run of 6 or more "
    "residues drawn only from S, A and V. The repeat region runs from the start of "
    "the first to the end of the last maximal run of 5 or more S/A/V residues."
)


def _prompt(seq: str, question: str, keys: Dict[str, str]) -> str:
    schema = ", ".join(f'"{k}": {v}' for k, v in keys.items())
    return (f"Spidroin sequence ({len(seq)} residues):\n{seq}\n\n{DEFINITIONS}\n\n{question}\n"
            f"Answer with only a JSON object: {{{schema}}}")


def make_task(seq: str, kind: str, rng: random.Random, other: Optional[str] = None) -> Dict[str, Any]:
    g = analyze_sequence(seq)
    if kind == "motifs":
        chosen = rng.sample(sorted(MOTIFS), 3)
        gold = {"length": g["length"], **{f"count_{m}": g["motif_counts"][m] for m in chosen}}
        types = {"length": "int", **{f"count_{m}": "int" for m in chosen}}
        q = f"Report the sequence length and the number of occurrences of {', '.join(chosen)}."
    elif kind == "composition":
        chosen = rng.sample(COMP_RESIDUES, 3)
        gold = {f"pct_{a}": g["composition_pct"][a] for a in chosen}
        types = {f"pct_{a}": "pct" for a in chosen}
        q = f"Report the percentage of residues that are {', '.join(chosen)} (one decimal place)."
    elif kind == "polyala":
        gold = {"count": g["polyala_count"], "longest": g["polyala_longest"], "spans": g["polyala_spans"]}
        types = {"count": "int", "longest": "int", "spans": "spans"}
        q = "Report the number of poly-Ala blocks, the length of the longest, and every block's [start, end]."
    elif kind == "repeats":
        gold = {"crystalline_count": g["crystalline_count"], "crystalline_residues": g["crystalline_residues"],
                "repeat_region_length": g["repeat_region_length"], "amorphous_to_crystalline": g["amorphous_to_crystalline"]}
        types = {"crystalline_count": "int", "crystalline_residues": "int",
                 "repeat_region_length": "int", "amorphous_to_crystalline": "ratio"}
        q = ("Report the number of crystalline blocks, the total residues in them, the repeat-region "
             "length (0 if fewer than two S/A/V runs), and the ratio of non-crystalline to crystalline "
             "residues within the repeat region (two decimals; null if undefined).")
    elif kind == "compare":
        assert other is not None
        h = analyze_sequence(other)
        prop, label = rng.choice([("polyala_longest", "the longer longest poly-Ala block"),
                                  ("crystalline_count", "more crystalline blocks"),
                                  ("G", "the higher percentage of glycine")])
        va = g["composition_pct"]["G"] if prop == "G" else g[prop]
        vb = h["composition_pct"]["G"] if prop == "G" else h[prop]
        gold = {"answer": "A" if va > vb else "B" if vb > va else "tie"}
        types = {"answer": "choice"}
        prompt = (f"Sequence A ({len(seq)} residues):\n{seq}\n\nSequence B ({len(other)} residues):\n{other}\n\n"
                  f"{DEFINITIONS}\n\nWhich sequence has {label}?\n"
                  'Answer with only a JSON object: {"answer": "A" | "B" | "tie"}')
        return {"kind": kind, "prompt": prompt, "gold": gold, "types": types,
                "level": level_of(seq if len(seq) >= len(other) else other)}
    else:
        raise ValueError(kind)
    keys = {k: {"int": "integer", "pct": "number", "ratio": "number or null",
                "spans": "list of [start, end]", "choice": "string"}[t] for k, t in types.items()}
    return {"kind": kind, "prompt": _prompt(seq, q, keys), "gold": gold, "types": types, "level": level_of(seq)}


# ---------------------------------------------------------------------------
# Checker (the reward)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """Last parseable JSON object in a model's output (tolerates prose / code fences)."""
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in reversed(starts):
        depth = 0
        for j in range(s, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[s:j + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break
    return None


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or v is None:  # bool is an int subclass — reject it explicitly
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_field(kind: str, gold, pred) -> float:
    if kind == "int":
        p = _num(pred)
        return float(p is not None and p == gold)
    if kind in ("pct", "ratio"):
        if gold is None:
            return float(pred is None)
        p = _num(pred)
        return float(p is not None and abs(p - gold) <= FLOAT_TOL[kind])
    if kind == "spans":  # F1 over exact spans
        try:
            ps = {(int(a), int(b)) for a, b in pred}
        except (TypeError, ValueError):
            return 0.0
        gs = {tuple(s) for s in gold}
        if not gs and not ps:
            return 1.0
        tp = len(gs & ps)
        return 0.0 if tp == 0 else 2 * tp / (len(gs) + len(ps))
    if kind == "choice":
        return float(isinstance(pred, str) and pred.strip().lower() == gold.lower())
    raise ValueError(kind)


def score(task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """Per-field credit, mean score in [0, 1], and an all-correct flag."""
    obj = extract_json(model_output)
    if obj is None:
        return {"parse_ok": False, "fields": {}, "score": 0.0, "all_correct": False}
    fields = {k: score_field(t, task["gold"][k], obj.get(k)) for k, t in task["types"].items()}
    s = sum(fields.values()) / len(fields)
    return {"parse_ok": True, "fields": fields, "score": s, "all_correct": s == 1.0}


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def load_sequences(min_len: int = 30) -> List[Dict[str, str]]:
    seen, out = set(), []
    with open(GENES_PATH) as f:
        for line in f:
            r = json.loads(line)
            for g in r["genes"]:
                s = (g.get("amino_acid_sequence") or "").upper().rstrip("*")
                if len(s) >= min_len and s not in seen and re.fullmatch(r"[A-Z]+", s):
                    seen.add(s)
                    out.append({"seq": s, "genus": r["genus"], "type": g["protein_name"]})
    return out


def split_of(genus: str) -> str:
    u = int(hashlib.sha256(genus.encode()).hexdigest(), 16) % 10000 / 10000
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if u < acc:
            return name
    return SPLITS[-1][0]


def build(tasks_per_seq: int = 2, seed: int = 0) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    seqs = load_sequences()
    by_split = defaultdict(list)
    for s in seqs:
        by_split[split_of(s["genus"])].append(s)
    kinds = ["motifs", "composition", "polyala", "repeats", "compare"]
    data = {}
    for split, pool in by_split.items():
        tasks = []
        for s in pool:
            g = analyze_sequence(s["seq"])
            # Block-finding tasks only on sequences that have blocks; otherwise
            # "0 blocks, no spans" is right without reading (many entries are
            # terminal-domain fragments with no poly-Ala).
            eligible = [k for k in kinds
                        if not (k == "polyala" and g["polyala_count"] == 0)
                        and not (k == "repeats" and g["crystalline_count"] < 2)]
            for kind in rng.sample(eligible, min(tasks_per_seq, len(eligible))):
                other = rng.choice(pool)["seq"] if kind == "compare" else None
                t = make_task(s["seq"], kind, rng, other)
                t.update({"split": split, "genus": s["genus"], "spidroin_type": s["type"]})
                tasks.append(t)
        data[split] = tasks
    return data


def write(data: Dict[str, List[dict]]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, tasks in data.items():
        with open(os.path.join(OUT_DIR, f"stage_a_{split}.jsonl"), "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")


# ---------------------------------------------------------------------------
# Self-tests and optional LLM baseline
# ---------------------------------------------------------------------------

def _naive_answer(task: dict, train_medians: Dict[str, float]) -> str:
    """A non-reading baseline: typical values from training, no spans, always 'A'."""
    ans = {}
    for k, t in task["types"].items():
        ans[k] = ([] if t == "spans" else "A" if t == "choice" else
                  round(train_medians.get(k, 0)) if t == "int" else train_medians.get(k, 0))
    return json.dumps(ans)


def self_test(data: Dict[str, List[dict]]) -> None:
    # 1. hand-checked example
    ex = "GGAAAAGSAVSAVGGYAAAAAAQQ"
    g = analyze_sequence(ex)
    assert g["polyala_spans"] == [[3, 6], [17, 22]], g["polyala_spans"]
    assert g["motif_counts"]["GGX"] == 2 and g["motif_counts"]["QQ"] == 1 and g["motif_counts"]["GGY"] == 1
    # crystalline: SAVSAV (8-13) and AAAAAA (17-22); AAAA at 3-6 is too short
    assert g["crystalline_count"] == 2 and g["crystalline_residues"] == 12, g
    # repeat region 8..22 = 15 residues, 3 non-crystalline -> 3/12
    assert g["repeat_region_length"] == 15 and g["amorphous_to_crystalline"] == 0.25, g
    # 2. gold scores 1.0; a single off-by-one is caught; prose around JSON still parses
    for t in data["test"][:500]:
        assert score(t, "Here you go:\n```json\n" + json.dumps(t["gold"]) + "\n```")["score"] == 1.0
    t = next(t for t in data["test"] if t["kind"] == "polyala" and t["gold"]["count"] > 0)
    bad = dict(t["gold"], longest=t["gold"]["longest"] + 1)
    assert score(t, json.dumps(bad))["score"] < 1.0
    assert score(t, '{"count": true}')["fields"]["count"] == 0.0  # booleans are not numbers
    print("self-test passed: hand example, gold = 1.0, off-by-one caught, bool rejected, prose tolerated")

    # 3. a non-reading baseline should score poorly
    med = defaultdict(list)
    for t in data["train"]:
        for k, v in t["gold"].items():
            if t["types"][k] in ("int", "pct", "ratio") and v is not None:
                med[k].append(v)
    medians = {k: sorted(v)[len(v) // 2] for k, v in med.items()}
    by_kind = defaultdict(list)
    for t in data["test"]:
        by_kind[t["kind"]].append(score(t, _naive_answer(t, medians))["score"])
    print("non-reading baseline (typical values) on test, mean score by task: "
          + ", ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in sorted(by_kind.items())))


def eval_llm(data: Dict[str, List[dict]], n: int, seed: int = 0) -> None:
    from trace_generation import LLMNotConfiguredError, call_trace_generator_llm
    rng = random.Random(seed)
    tasks = rng.sample(data["test"], min(n, len(data["test"])))
    res = defaultdict(list)
    for i, t in enumerate(tasks):
        try:
            out = call_trace_generator_llm(t["prompt"])
        except LLMNotConfiguredError:
            raise
        except Exception as e:  # transient API failure: record as unscored
            print(f"  task {i}: API error {type(e).__name__}: {e}")
            continue
        s = score(t, out)
        res[(t["kind"], t["level"])].append(s["score"])
    print("LLM baseline (mean score, n):")
    for k in sorted(res):
        print(f"  {k[0]:<12}{k[1]}  {sum(res[k]) / len(res[k]):.2f}  (n={len(res[k])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tasks-per-seq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-llm", type=int, default=0, help="score N test tasks with the configured LLM (API calls)")
    args = ap.parse_args()

    data = build(args.tasks_per_seq, args.seed)
    write(data)
    genera = {s: len({t["genus"] for t in v}) for s, v in data.items()}
    print(f"wrote {OUT_DIR}/stage_a_{{train,val,test}}.jsonl")
    for split in ("train", "val", "test"):
        v = data[split]
        print(f"  {split:<5} {len(v):>6} tasks, {genera[split]:>3} genera | "
              f"kinds {dict(sorted(Counter(t['kind'] for t in v).items()))} | "
              f"levels {dict(sorted(Counter(t['level'] for t in v).items()))}")
    self_test(data)
    if args.eval_llm:
        eval_llm(data, args.eval_llm, args.seed)


if __name__ == "__main__":
    main()
