"""Provenance of the MD fragments, using the authors' input and extraction files
(silkome_data/md_input_sequences.csv, silkome_data/md_fragment_extraction.csv; from Wei Lu).

1. Natural vs. SilkomeGPT-generated fragments, and how often generated fragments
   reproduce natural Silkome sequence verbatim.
2. Where fragments sit in their source protein: repeat region vs. terminal regions.
3. MD vs. measured fiber properties using the paper's own property assignment for
   natural sequences (no substring matching needed), overall and for fragments that
   lie mostly inside the repeat region.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

D = os.path.join(REPO, "silkome_data")
md = pd.read_csv(os.path.join(REPO, "d5ma00154d1_suppl.csv"))
ex = pd.read_csv(os.path.join(D, "md_fragment_extraction.csv"))
inp = pd.read_csv(os.path.join(D, "md_input_sequences.csv"))
smap = pd.read_csv(os.path.join(D, "md_fragment_species_map.csv"))  # from md_vs_fiber.py

df = md.merge(ex[["idv", "section_sequence", "section_start_index", "section_end_index", "Novel"]],
              left_on=["prot_id", "seq"], right_on=["idv", "section_sequence"], how="inner")
assert len(df) == len(md), "MD rows must align one-to-one with the extraction file"
df["origin"] = np.where(df.Novel.fillna(False).astype(bool), "SilkomeGPT", "natural")
src = inp.set_index("idv")

# --- 1. origin, and verbatim natural sequence inside generated fragments ---------------
matched = set(smap.loc[smap.n_species > 0, "fragment"])
df["in_silkome_genes"] = df.seq.str.upper().isin(matched)
print("== 1. Origin of MD fragments ==")
print(df.groupby("origin").agg(rows=("seq", "size"), unique_seqs=("seq", "nunique"),
                               source_proteins=("prot_id", "nunique"),
                               verbatim_in_silkome=("in_silkome_genes", "mean")).round(2).to_string())
print("(verbatim_in_silkome: fraction found as an exact substring of downloaded Silkome genes, "
      "which cover only species with measured fibers)")

# --- 2. position of each fragment in its source protein ----------------------------------
def repeat_span(seq):
    hits = [m.span() for m in re.finditer(r"[SAV]{5,}", seq)]
    return (hits[0][0], hits[-1][1]) if len(hits) >= 2 else None

rows = []
for r in df.itertuples():
    full = src.loc[r.prot_id, "sequence"]
    s, e = int(r.section_start_index), int(r.section_end_index) + 1   # 0-based, end exclusive
    span = repeat_span(full)
    inside = 0 if span is None else max(0, min(e, span[1]) - max(s, span[0]))
    rows.append({"frac_in_repeat_region": inside / (e - s),
                 "rel_start": s / len(full), "rel_end": e / len(full),
                 "touches_c_end": e >= len(full) - 5, "touches_n_end": s <= 5,
                 "full_len": len(full)})
df = pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
df["location"] = np.select(
    [df.frac_in_repeat_region >= 0.8, df.frac_in_repeat_region <= 0.2],
    ["mostly repeat region", "mostly outside repeats"], "mixed")
print("\n== 2. Where fragments sit in their source protein ==")
print(pd.crosstab(df.origin, df.location, normalize="index").round(2).to_string())
print(f"touches the protein's C-terminal end: natural {df[df.origin == 'natural'].touches_c_end.mean():.0%}, "
      f"generated {df[df.origin == 'SilkomeGPT'].touches_c_end.mean():.0%}")
print(f"median source length: natural {int(df[df.origin == 'natural'].full_len.median())}, "
      f"generated {int(df[df.origin == 'SilkomeGPT'].full_len.median())}; "
      f"median fragment length {int(df.len.median())}")

# --- 3. MD vs measured fiber properties (natural only; paper's property assignment) ------
nat = df[df.origin == "natural"].copy()
for col in ["strength", "toughness"]:   # length-adjusted MD values (log residual on log length)
    b = np.polyfit(np.log(md.len), np.log(md[col]), 2)
    nat[col + "_adj"] = np.log(nat[col]) - np.polyval(b, np.log(nat.len))
fiber = src.loc[src.Novel.isna(), ["toughness", "E", "strength", "strain"]].add_prefix("fiber_")
P = (nat.groupby("prot_id")[["strength", "toughness", "strength_adj", "toughness_adj", "frac_in_repeat_region"]]
     .mean().join(fiber, how="inner"))
# proteins from the same sample share an identical property tuple; collapse to avoid
# counting one measurement many times
P["sample_key"] = P[[c for c in P.columns if c.startswith("fiber_")]].round(4).astype(str).agg("|".join, axis=1)

def table(frame, label):
    S = frame.groupby("sample_key").mean(numeric_only=True)
    n = len(S)
    print(f"\n{label}: {len(frame)} proteins, {n} distinct measured samples "
          f"(chance |ρ| ≈ {1.96 / np.sqrt(max(n - 1, 1)):.2f})")
    md_cols = ["strength", "toughness", "strength_adj", "toughness_adj"]
    fib_cols = ["fiber_strength", "fiber_toughness", "fiber_E", "fiber_strain"]
    print(f"{'MD (mean per sample)':<22}" + "".join(f"{c:>17}" for c in fib_cols))
    for m in md_cols:
        print(f"{'md_' + m:<22}" + "".join(f"{spearmanr(S[m], S[f]).correlation:>+17.2f}" for f in fib_cols))

print("\n== 3. MD vs measured fiber properties (natural sequences, paper's assignment) ==")
table(P, "All natural fragments")
table(P[P.frac_in_repeat_region >= 0.8], "Fragments mostly inside the repeat region")
