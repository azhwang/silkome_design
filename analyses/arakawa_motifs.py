"""Replicate Arakawa et al. 2022's motif screen and test whether its motifs generalize.

Their method (Methods, 'For motif extraction'): repetitive region = longest segment
spanning [SAV]{5,}; split into repeat units at [SAV]{6,} crystalline blocks; count every
1-8 aa motif in the repetitive region / number of repeats; average per paralog
(MaSp1/2/3); Pearson r with each property; drop motifs seen in < 30 samples.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import re, warnings
warnings.filterwarnings("ignore")
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import silkome_fiber_model as m

PARALOGS = {"MaSp1": "MaSp1", "MaSp2": "MaSp2", "MaSp2B": "MaSp2", "MaSp3": "MaSp3"}
PROPS = ["toughness", "tensile_strength", "strain_at_break"]

def repeat_region(seq):
    hits = [mm.span() for mm in re.finditer(r"[SAV]{5,}", seq)]
    if len(hits) < 2:
        return None, 0
    region = seq[hits[0][0]:hits[-1][1]]
    n_rep = len(re.findall(r"[SAV]{6,}", region))
    return region, max(n_rep, 1)

def kmer_rates(region, n_rep, kmax=8):
    c = Counter()
    for k in range(1, kmax + 1):
        for i in range(len(region) - k + 1):
            c[region[i:i + k]] += 1
    return {km: v / n_rep for km, v in c.items()}

df = m.load_species_table()
rows = []
for _, r in df.iterrows():
    per_par = defaultdict(list)
    for seq, g in r["genes"].items():
        par = PARALOGS.get(g["type"])
        if not par:
            continue
        region, n_rep = repeat_region(seq)
        if region:
            per_par[par].append(kmer_rates(region, n_rep))
    feats = {}
    for par, lst in per_par.items():
        keys = set().union(*lst)
        for km in keys:
            feats[f"{par}:{km}"] = np.mean([d.get(km, 0.0) for d in lst])
    if feats:
        rows.append({"taxon": r["taxon"], "family": r["family"], "genus": r["genus"],
                     **{p: np.log(r[p]) if r[p] > 0 else np.nan for p in PROPS}, "feats": feats})
sp = pd.DataFrame(rows)
print(f"{len(sp)} species with MaSp1/2/3 repeat regions; families: {sp.family.value_counts().head(4).to_dict()}")

# Motif matrix: keep motifs present (nonzero) in >= 30 species, as in the paper
cnt = Counter(k for f in sp.feats for k, v in f.items() if v > 0)
keep = sorted(k for k, c in cnt.items() if c >= 30)
X = np.array([[f.get(k, 0.0) for k in keep] for f in sp.feats])
print(f"{len(keep)} motifs tested (present in >=30 species)")

# --- 1. Their headline motifs: global r vs within-family r -------------------
TABLE1 = {  # (property, motif feature, paper's sign)
    ("toughness", "MaSp1:YGQGG", "+"), ("toughness", "MaSp1:GYGQGG", "+"),
    ("toughness", "MaSp1:GGGQ", "+"), ("toughness", "MaSp2:P", "-"), ("toughness", "MaSp2:SQGP", "-"),
    ("toughness", "MaSp1:SY", "-"), ("toughness", "MaSp1:SV", "-"),
    ("tensile_strength", "MaSp1:GYGQGG", "+"), ("tensile_strength", "MaSp1:QGGS", "+"),
    ("tensile_strength", "MaSp2:PQ", "-"), ("tensile_strength", "MaSp1:AGQG", "-"),
    ("strain_at_break", "MaSp1:GYGQGG", "+"), ("strain_at_break", "MaSp1:QGP", "+"),
    ("strain_at_break", "MaSp1:PGA", "+"),
}
idx = {k: i for i, k in enumerate(keep)}
def r_global_within(x, y, fam):
    ok = ~np.isnan(y)
    rg = pearsonr(x[ok], y[ok])[0]
    d = pd.DataFrame({"x": x[ok], "y": y[ok], "f": fam[ok]})
    d = d[d.f.map(d.f.value_counts()) >= 3]
    dm = d.groupby("f")[["x", "y"]].transform(lambda c: c - c.mean())
    return rg, pearsonr(dm.x, dm.y)[0], len(dm)

print("\n== Table 1 motifs: paper sign, r across species, r within families ==")
fam = sp.family.to_numpy()
for prop, motif, sign in sorted(TABLE1):
    if motif not in idx:
        print(f"  {prop:<17}{motif:<14} (not present in >=30 species)")
        continue
    rg, rw, n = r_global_within(X[:, idx[motif]], sp[prop].to_numpy(), fam)
    print(f"  {prop:<17}{motif:<14} paper {sign}   global r {rg:+.2f}   within-family r {rw:+.2f} (n={n})")

araneid = (sp.family == "Araneidae").to_numpy()
g = X[:, idx["MaSp1:GYGQGG"]] if "MaSp1:GYGQGG" in idx else None
if g is not None:
    print(f"\nGYGQGG in MaSp1 present: Araneidae {np.mean(g[araneid] > 0):.0%} of species, "
          f"other families {np.mean(g[~araneid] > 0):.0%}")

# --- 2. How extreme is the top screened r under the null? --------------------
rng = np.random.default_rng(0)
print("\n== Screen: top |r| observed vs. top |r| with property shuffled (100 perms) ==")
Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
for prop in PROPS:
    y = sp[prop].to_numpy(); ok = ~np.isnan(y)
    def top_r(yv):
        yz = (yv - yv.mean()) / yv.std()
        return np.max(np.abs(Xs[ok].T @ yz / ok.sum()))
    obs = top_r(y[ok]); null = [top_r(rng.permutation(y[ok])) for _ in range(100)]
    # within-family version
    d = pd.DataFrame(X[ok]); d["f"] = fam[ok]; yy = pd.Series(y[ok])
    big = d.f.map(d.f.value_counts()).to_numpy() >= 3
    Xw = d[big].groupby("f").transform(lambda c: c - c.mean()).to_numpy()
    yw = (yy[big] - yy[big].groupby(d.f[big].values).transform("mean")).to_numpy()
    Xw = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-12); yw = (yw - yw.mean()) / yw.std()
    obs_w = np.nanmax(np.abs(Xw.T @ yw / len(yw)))
    print(f"  {prop:<17} top |r| {obs:.2f} (null 95th pct {np.percentile(null, 95):.2f})   "
          f"within-family top |r| {obs_w:.2f}")

# --- 3. Does the screen generalize? Re-run it on training genera only ----------
print("\n== Screen-then-predict on held-out genera (5 random genus splits) ==")
def screen_predict(prop, top_n=20, family_too=False):
    y = sp[prop].to_numpy(); ok = ~np.isnan(y)
    Xo, yo, gen = X[ok], y[ok], sp.genus.to_numpy()[ok]
    F = pd.get_dummies(sp.family[ok]).to_numpy(float)
    r2s = []
    for rep in range(5):
        folds = m.random_group_folds(gen, 5, rep); oof = np.full(len(yo), np.nan)
        for k in range(5):
            tr, te = folds != k, folds == k
            Xt = (Xo[tr] - Xo[tr].mean(0)) / (Xo[tr].std(0) + 1e-12)
            r = np.abs(Xt.T @ ((yo[tr] - yo[tr].mean()) / yo[tr].std()) / tr.sum())
            sel = np.argsort(-np.nan_to_num(r))[:top_n]
            Z = np.hstack([Xo[:, sel], F]) if family_too else Xo[:, sel]
            gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                              {"ridge__alpha": np.logspace(-1, 4, 11)}, cv=GroupKFold(4), scoring="r2")
            gs.fit(Z[tr], yo[tr], groups=gen[tr]); oof[te] = gs.predict(Z[te])
        r2s.append(r2_score(yo, oof))
    return np.mean(r2s), np.std(r2s)

def family_only(prop):
    y = sp[prop].to_numpy(); ok = ~np.isnan(y); gen = sp.genus.to_numpy()[ok]
    F = pd.get_dummies(sp.family[ok]).to_numpy(float); yo = y[ok]; r2s = []
    for rep in range(5):
        folds = m.random_group_folds(gen, 5, rep); oof = np.full(len(yo), np.nan)
        for k in range(5):
            tr, te = folds != k, folds == k
            oof[te] = Ridge(alpha=1.0).fit(F[tr], yo[tr]).predict(F[te])
        r2s.append(r2_score(yo, oof))
    return np.mean(r2s)

for prop in PROPS:
    a = screen_predict(prop); b = screen_predict(prop, family_too=True)
    print(f"  {prop:<17} family only {family_only(prop):+.3f}   screened top-20 motifs {a[0]:+.3f} ± {a[1]:.3f}"
          f"   motifs + family {b[0]:+.3f} ± {b[1]:.3f}")
