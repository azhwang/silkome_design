"""Can anything beyond length predict MD strength/toughness on held-out sequence clusters?
Spline(length) baseline + feature blocks, ridge alpha tuned by *grouped* inner CV, plus
learning curves over the fraction of training clusters. Needs cached ESM embeddings
(run silk_regressor.py --esm-model ... first)."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
import silk_regressor as sr

D = REPO + "/"
df, noise = sr.load_data(D + sr.DEFAULT_DATA_PATH)
df = df.dropna(subset=["strength"]).reset_index(drop=True)
seqs = df.seq.tolist()
groups = sr.cluster_sequences(seqs)
L = np.array([[len(s)] for s in seqs], float)

def term_comp(s, k=30):
    return np.concatenate([sr.composition(s[:k]), sr.composition(s[-k:])])

def load_esm(name):
    z = np.load(D + f"embeddings/facebook__{name}.npz", allow_pickle=True)
    idx = {s: i for i, s in enumerate(z["seqs"])}
    return z["emb"][[idx[s] for s in seqs]]

comp = np.vstack([sr.composition(s) for s in seqs])
blocks = {
    "comp": comp,
    "comp+termini": np.hstack([comp, np.vstack([term_comp(s) for s in seqs])]),
    "biotools": np.hstack([comp, np.vstack([sr.biotools_features(s) for s in seqs])]),
    "dipep": np.vstack([sr.dipeptide_composition(s) for s in seqs]),
    "esm35M": load_esm("esm2_t12_35M_UR50D"),
    "esm650M": load_esm("esm2_t33_650M_UR50D"),
}

def model(n_extra, pca=None):
    """spline(length) + scaled extra features [-> PCA] -> ridge."""
    extra = [StandardScaler()] + ([PCA(pca, random_state=0)] if pca else [])
    ct = ColumnTransformer([("len", SplineTransformer(n_knots=6, degree=3), [0])] +
                           ([("x", make_pipeline(*extra), list(range(1, 1 + n_extra)))] if n_extra else []))
    return Pipeline([("ct", ct), ("ridge", Ridge())])

ALPHAS = {"ridge__alpha": np.logspace(-2, 5, 15)}

def grouped_r2(X, y, pca=None, train_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        if train_frac < 1:
            g = np.unique(groups[tr]); keep = rng.choice(g, int(len(g) * train_frac), replace=False)
            tr = tr[np.isin(groups[tr], keep)]
        gs = GridSearchCV(model(X.shape[1] - 1, pca), ALPHAS, cv=GroupKFold(4), scoring="r2")
        gs.fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict(X[te])
    return r2_score(y, oof), oof

if __name__ == "__main__":
    for target in ["strength", "log10_toughness"]:
        y = df[target].to_numpy()
        base, _ = grouped_r2(L, y)
        print(f"\n=== {target}: spline(length) grouped R² = {base:.3f}  (noise ceiling ≈ {noise[target]:.2f}) ===")
        print(f"{'features':<22}{'R²':>7}{'ΔR² vs len':>12}")
        best = (base, "length", None)
        for name, B in blocks.items():
            for pca in ([None, 16, 64] if name.startswith("esm") or name == "dipep" else [None]):
                r2, _ = grouped_r2(np.hstack([L, B]), y, pca)
                label = name + (f" (PCA{pca})" if pca else "")
                print(f"{label:<22}{r2:>7.3f}{r2 - base:>+12.3f}")
                if r2 > best[0]:
                    best = (r2, label, (B, pca))
        print(f"best: {best[1]}")
        if best[2] is not None:
            B, pca = best[2]
            print("learning curve (fraction of training clusters -> grouped R², mean of 3 seeds):")
            for f in [0.25, 0.5, 0.75, 1.0]:
                rs = [grouped_r2(np.hstack([L, B]), y, pca, f, s)[0] for s in range(3 if f < 1 else 1)]
                rl = [grouped_r2(L, y, None, f, s)[0] for s in range(3 if f < 1 else 1)]
                print(f"  {f:.2f}: {best[1]} {np.mean(rs):.3f}   length-only {np.mean(rl):.3f}   gap {np.mean(rs)-np.mean(rl):+.3f}")
