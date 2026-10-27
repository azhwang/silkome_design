"""Compare Arakawa et al. data S4 motif correlations with the same correlations on the public download."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import json, re, warnings
warnings.filterwarnings("ignore")
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import silkome_fiber_model as m

S4 = os.environ.get("SILKOME_S4", os.path.join(os.path.dirname(REPO), "sciadv.abo6043_data_s1_to_s4", "sciadv.abo6043_data_s4.xlsx"))
PROPS = {"toughness": "toughness", "young's_modulus": "youngs_modulus", "tensile_strength": "tensile_strength",
         "strain_at_break": "strain_at_break", "crystallinity": "crystallinity",
         "birefringence": "birefringence", "diameter": "diameter", "supercontraction": "supercontraction"}
PARALOGS = {"MaSp1": "MaSp1", "MaSp2": "MaSp2", "MaSp2B": "MaSp2", "MaSp3": "MaSp3"}

theirs = pd.read_excel(S4, sheet_name="variableSelection-N>30", index_col=0)
motifs = list(theirs.index)
by_par = defaultdict(set)
for mo in motifs:
    p, k = mo.split("-", 1)
    by_par[p].add(k)

def repeat_region(seq):
    hits = [h.span() for h in re.finditer(r"[SAV]{5,}", seq)]
    if len(hits) < 2:
        return None, 0
    region = seq[hits[0][0]:hits[-1][1]]
    return region, max(len(re.findall(r"[SAV]{6,}", region)), 1)

def rates(region, n_rep, wanted):
    c = Counter()
    for k in {len(w) for w in wanted}:
        for i in range(len(region) - k + 1):
            s = region[i:i + k]
            if s in wanted:
                c[s] += 1
    return {w: c[w] / n_rep for w in wanted}

genes_by_ind = {}
with open(m.DATA + "/genes.jsonl") as f:
    for line in f:
        r = json.loads(line)
        genes_by_ind[r["individual_id"]] = r["genes"]

P = pd.DataFrame(json.load(open(m.DATA + "/mechanical_properties.json")))
for c in PROPS.values():
    P[c] = P[c].map(m._num)

feat_rows = []
for _, s in P.iterrows():
    per = defaultdict(list)
    for g in genes_by_ind.get(s["individual_id"], []):
        par = PARALOGS.get(g["protein_name"])
        seq = (g.get("amino_acid_sequence") or "").upper()
        if par in by_par and seq:
            region, n_rep = repeat_region(seq)
            if region:
                per[par].append(rates(region, n_rep, by_par[par]))
    row = {}
    for par, lst in per.items():
        for w in by_par[par]:
            row[f"{par}-{w}"] = np.mean([d[w] for d in lst])
    feat_rows.append(row)
F = pd.DataFrame(feat_rows, index=P.index).reindex(columns=motifs)
print(f"{len(P)} measured samples; with MaSp1 repeats: {F.filter(like='MaSp1-').notna().any(axis=1).sum()}, "
      f"MaSp2: {F.filter(like='MaSp2-').notna().any(axis=1).sum()}, MaSp3: {F.filter(like='MaSp3-').notna().any(axis=1).sum()}")

ours = pd.DataFrame(index=motifs, columns=list(PROPS), dtype=float)
nobs = pd.DataFrame(index=motifs, columns=list(PROPS), dtype=float)
for col, pcol in PROPS.items():
    y = P[pcol]
    for mo in motifs:
        x = F[mo]
        ok = x.notna() & y.notna()
        if ok.sum() >= 30 and x[ok].std() > 0:
            ours.loc[mo, col] = pearsonr(x[ok], y[ok])[0]
            nobs.loc[mo, col] = ok.sum()

print("\n== Do our motif correlations track theirs? (agreement across motifs) ==")
print(f"{'property':<18}{'motifs':>7}{'n samples':>11}{'Pearson(theirs, ours)':>23}{'same sign':>11}{'their max|r|':>14}{'our max|r|':>12}")
for col in PROPS:
    both = theirs[col].notna() & ours[col].notna()
    if both.sum() < 10:
        print(f"{col:<18}{both.sum():>7}  (too few motifs to compare)")
        continue
    a, b = theirs.loc[both, col].astype(float), ours.loc[both, col].astype(float)
    print(f"{col:<18}{both.sum():>7}{int(nobs.loc[both, col].median()):>11}{pearsonr(a, b)[0]:>+23.2f}"
          f"{np.mean(np.sign(a) == np.sign(b)):>11.0%}{a.abs().max():>14.2f}{b.abs().max():>12.2f}")

print("\n== Their top-5 motifs per property: their r vs ours ==")
for col in ["toughness", "tensile_strength", "strain_at_break", "young's_modulus", "supercontraction"]:
    top = theirs[col].abs().sort_values(ascending=False).index[:5]
    print(f"  {col}: " + "  ".join(f"{mo} {theirs.loc[mo, col]:+.2f}/{ours.loc[mo, col]:+.2f}" for mo in top))

# Null: largest |r| the screen finds by chance, per paralog (MaSp3 has ~33 samples, which
# inflates a pooled null, so MaSp1 and MaSp2 are tested separately).
rng = np.random.default_rng(0)
print("\n== Chance level per paralog: 95th pct of top |r| with property shuffled (100 perms) ==")
for col in ["toughness", "tensile_strength", "strain_at_break", "young's_modulus", "supercontraction"]:
    y = P[PROPS[col]]
    for par in ["MaSp1", "MaSp2"]:
        cols = [mo for mo in motifs if mo.startswith(par + "-") and not np.isnan(ours.loc[mo, col])]
        if not cols:
            continue
        rows = F[cols].notna().all(axis=1) & y.notna()
        Xz = F.loc[rows, cols]
        Xz = (Xz - Xz.mean()) / Xz.std().replace(0, np.nan)
        Xz = Xz.loc[:, Xz.notna().all()]
        yv = y[rows].values
        n = len(yv)
        def top(v):
            v = (v - v.mean()) / v.std()
            return np.nanmax(np.abs(Xz.values.T @ v / (n - 1)))
        null = [top(rng.permutation(yv)) for _ in range(100)]
        th = theirs.loc[[c for c in cols if c in theirs.index], col].abs().max()
        print(f"  {col:<17}{par}: {len(cols):>4} motifs, n={n:<4} chance 95th pct {np.percentile(null, 95):.2f}"
              f"   their top {th:.2f}   our top {top(yv):.2f}")

ours.to_csv(os.path.join(REPO, "analyses", "s4_ours_vs_theirs.csv"))
