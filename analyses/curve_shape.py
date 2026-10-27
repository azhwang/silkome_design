"""Is the length-independent shape of the MD force curves learnable from sequence?"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
import silk_regressor as sr

d = pd.read_csv(REPO + "/d5ma00154d1_suppl.csv")
d["F"] = [np.array(v.strip("[]").split(), float) for v in d.force_vector]
grid = np.linspace(0, 1, 21)

def shape_targets(f):
    prof = np.interp(grid, np.linspace(0, 1, len(f)), f)
    body = f[: int(0.7 * len(f))]                      # before the final-stretch regime
    pk, props = find_peaks(body, prominence=0.1 * f.max())
    return {
        "unfold_peak_force": body.max(),                       # strongest rupture before full extension
        "unfold_mean_force": body.mean(),
        "n_rupture_peaks_per_100res": 100 * len(pk) / len(f),
        "final_stretch_share": (f[int(0.7 * len(f)):].sum()) / f.sum(),  # share of work in last 30%
        "profile_curvature": np.polyfit(grid, prof / prof.max(), 2)[0],
    }

T = pd.DataFrame([shape_targets(f) for f in d.F])
T["seq"] = d.seq.str.upper().str.replace(r"[^A-Z]", "", regex=True)
T = T.groupby("seq", as_index=False).mean()   # average replicate sequences, as in silk_regressor
seqs = T.seq.tolist()
groups = sr.cluster_sequences(seqs)
L = np.array([[len(s)] for s in seqs], float)
comp = np.vstack([sr.composition(s) for s in seqs])
bio = np.vstack([sr.biotools_features(s) for s in seqs])

def model(n_extra):
    ct = ColumnTransformer([("len", SplineTransformer(n_knots=6), [0])] +
                           ([("x", StandardScaler(), list(range(1, 1 + n_extra)))] if n_extra else []))
    return Pipeline([("ct", ct), ("ridge", Ridge())])

def grouped_r2(X, y):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        gs = GridSearchCV(model(X.shape[1] - 1), {"ridge__alpha": np.logspace(-2, 5, 15)},
                          cv=GroupKFold(4), scoring="r2").fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict(X[te])
    return r2_score(y, oof)

print(f"{len(seqs)} unique sequences, {len(set(groups))} clusters; R² on held-out clusters\n")
print(f"{'curve-shape target':<30}{'ρ with length':>14}{'length only':>13}{'+ composition':>15}{'+ bio_tools':>13}")
for t in ["unfold_peak_force", "unfold_mean_force", "n_rupture_peaks_per_100res", "final_stretch_share", "profile_curvature"]:
    y = T[t].to_numpy()
    base = grouped_r2(L, y)
    c = grouped_r2(np.hstack([L, comp]), y)
    b = grouped_r2(np.hstack([L, comp, bio]), y)
    print(f"{t:<30}{spearmanr(y, L[:, 0]).correlation:>+14.2f}{base:>13.2f}{c:>15.2f}{b:>13.2f}")
