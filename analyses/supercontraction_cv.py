"""Held-out prediction of supercontraction from Arakawa et al.'s prespecified motifs."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import json, re, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import silkome_fiber_model as m

PARALOGS = {"MaSp1": "MaSp1", "MaSp2": "MaSp2", "MaSp2B": "MaSp2", "MaSp3": "MaSp3"}
MOTIFS = [("MaSp1", "AAAAAA"), ("MaSp2", "AAAAAA"), ("MaSp2", "AAAAAAAA"), ("MaSp1", "AGQG")]

def rate(seq, motif):
    hits = [h.span() for h in re.finditer(r"[SAV]{5,}", seq)]
    if len(hits) < 2:
        return None
    region = seq[hits[0][0]:hits[-1][1]]
    n_rep = max(len(re.findall(r"[SAV]{6,}", region)), 1)
    return sum(1 for i in range(len(region) - len(motif) + 1) if region.startswith(motif, i)) / n_rep

df = m.load_species_table()
P = pd.DataFrame(json.load(open(m.DATA + "/mechanical_properties.json")))
P["sc"] = P["supercontraction"].map(m._num)
P["taxon"] = P.genus + " " + P.species
df["sc"] = df.taxon.map(P.groupby("taxon").sc.mean())

rows = []
for _, r in df.iterrows():
    vals = defaultdict(list)
    for seq, g in r["genes"].items():
        par = PARALOGS.get(g["type"])
        if par:
            for p, mo in MOTIFS:
                if p == par:
                    v = rate(seq, mo)
                    if v is not None:
                        vals[(p, mo)].append(v)
    feats = [np.mean(vals[k]) if vals[k] else np.nan for k in MOTIFS]
    rows.append(feats)
X = np.array(rows)
ok = df.sc.notna().to_numpy() & ~np.isnan(X).any(1)
X, y = X[ok], df.sc.to_numpy()[ok]
gen, fam = df.genus.to_numpy()[ok], df.family.to_numpy()[ok]
F = pd.get_dummies(fam).to_numpy(float)
print(f"n={ok.sum()} species with supercontraction + MaSp1 & MaSp2 repeats; "
      f"{len(set(gen))} genera, {len(set(fam))} families {pd.Series(fam).value_counts().head(4).to_dict()}")

def cv(Z, groups):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(Z, y, groups):
        oof[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 11))).fit(Z[tr], y[tr]).predict(Z[te])
    return r2_score(y, oof), spearmanr(y, oof).correlation

for name, Z, fam_cv in [("family only", F, False),
                        ("4 paper motifs (poly-Ala, AGQG)", X, True),
                        ("motifs + family", np.hstack([X, F]), False)]:
    r2g, rg = cv(Z, gen)
    line = f"  {name:<34} leave-one-genus-out R² {r2g:+.2f} (ρ {rg:+.2f})"
    if fam_cv:
        r2f, rf = cv(Z, fam)
        line += f"   leave-one-family-out R² {r2f:+.2f} (ρ {rf:+.2f})"
    print(line)
