"""Calibrate our repeat-region processing against Arakawa et al. Fig. S9 / Table S1:
amorphous/poly-Ala (SAV-block) length ratio per paralog vs. silk properties."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import json, re, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import silkome_fiber_model as m

PARALOGS = {"MaSp1": "MaSp1", "MaSp2": "MaSp2", "MaSp2B": "MaSp2", "MaSp3": "MaSp3"}
PROPS = ["toughness", "youngs_modulus", "tensile_strength", "strain_at_break", "supercontraction"]
PAPER = {  # Table S1
    "MaSp1": [-0.083, 0.068, 0.019, -0.045, 0.526],
    "MaSp2": [-0.009, -0.072, -0.106, 0.092, 0.394],
    "MaSp3": [-0.007, -0.061, 0.048, -0.071, 0.056],
}

def ratio(seq, min_block):
    region_hits = [h.span() for h in re.finditer(r"[SAV]{5,}", seq)]
    if len(region_hits) < 2:
        return None
    region = seq[region_hits[0][0]:region_hits[-1][1]]
    crys = sum(len(b) for b in re.findall(r"[SAV]{%d,}" % min_block, region))
    return (len(region) - crys) / crys if crys else None

genes_by_ind = {}
with open(m.DATA + "/genes.jsonl") as f:
    for line in f:
        r = json.loads(line)
        genes_by_ind[r["individual_id"]] = r["genes"]
P = pd.DataFrame(json.load(open(m.DATA + "/mechanical_properties.json")))
for c in PROPS:
    P[c] = P[c].map(m._num)

for min_block in (6, 5):  # paper: crystalline = SAV motif "length greater than five"
    feats = defaultdict(list)
    for _, s in P.iterrows():
        per = defaultdict(list)
        for g in genes_by_ind.get(s["individual_id"], []):
            par = PARALOGS.get(g["protein_name"])
            seq = (g.get("amino_acid_sequence") or "").upper()
            if par and seq:
                v = ratio(seq, min_block)
                if v is not None:
                    per[par].append(v)
        for par in PAPER:
            feats[par].append(np.mean(per[par]) if per[par] else np.nan)
    print(f"\n== crystalline block = [SAV]{{{min_block},}}: Pearson r (ours / paper Table S1), n samples ==")
    print(f"{'':8}" + "".join(f"{p:>22}" for p in PROPS))
    for par in PAPER:
        x = pd.Series(feats[par], index=P.index)
        cells = []
        for prop, paper_r in zip(PROPS, PAPER[par]):
            ok = x.notna() & P[prop].notna()
            r = pearsonr(x[ok], P.loc[ok, prop])[0] if ok.sum() > 5 else np.nan
            cells.append(f"{r:+.2f} / {paper_r:+.2f} (n={ok.sum()})")
        print(f"{par:<8}" + "".join(f"{c:>22}" for c in cells))
