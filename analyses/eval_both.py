"""(A) Do contact-map fold-proxy features help MD strength/toughness beyond length?
(B) How learnable are the d2 pre_* columns (SilkomeGPT predictions made from the fragments)?
Needs embeddings/esm650M_contact_features.npz (contact_features.py) and cached ESM-2 650M
embeddings (silk_regressor.py --esm-model facebook/esm2_t33_650M_UR50D)."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
import silk_regressor as sr

D = REPO + "/"
df, _ = sr.load_data(D + sr.DEFAULT_DATA_PATH)
fib = pd.read_csv(D + "d5ma00154d2_suppl.csv")
fib["seq"] = fib["seq"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
FIBER = ["pre_strength", "pre_toughness", "pre_E", "pre_strain"]
df = df.merge(fib.groupby("seq", as_index=False)[FIBER].mean(), on="seq", how="left")
df = df.dropna(subset=["strength"]).reset_index(drop=True)
seqs = df.seq.tolist()
groups = sr.cluster_sequences(seqs)
L = np.array([[len(s)] for s in seqs], float)

def cached(path, key):
    z = np.load(D + path, allow_pickle=True)
    idx = {s: i for i, s in enumerate(z["seqs"])}
    return z[key][[idx[s] for s in seqs]]

comp = np.vstack([sr.composition(s) for s in seqs])
contact = cached("embeddings/esm650M_contact_features.npz", "F")
esm = cached("embeddings/facebook__esm2_t33_650M_UR50D.npz", "emb")
BLOCKS = {  # name -> (features, pca)
    "comp": (comp, None),
    "contact (fold)": (contact, None),
    "comp+contact": (np.hstack([comp, contact]), None),
    "esm650M PCA16": (esm, 16),
    "esm650M full": (esm, None),
}

def model(n_extra, pca=None):
    extra = [StandardScaler()] + ([PCA(pca, random_state=0)] if pca else [])
    ct = ColumnTransformer([("len", SplineTransformer(n_knots=6, degree=3), [0])] +
                           ([("x", make_pipeline(*extra), list(range(1, 1 + n_extra)))] if n_extra else []))
    return Pipeline([("ct", ct), ("ridge", Ridge())])

def grouped_r2(X, y, pca=None):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        gs = GridSearchCV(model(X.shape[1] - 1, pca), {"ridge__alpha": np.logspace(-2, 5, 15)},
                          cv=GroupKFold(4), scoring="r2")
        gs.fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict(X[te])
    return r2_score(y, oof), spearmanr(y, oof).correlation

if __name__ == "__main__":
    print("fiber-level pre_* vs MD labels (Spearman):")
    for f in FIBER:
        print(f"  {f:<14} vs MD strength {spearmanr(df[f], df.strength).correlation:+.3f}  "
              f"vs MD toughness {spearmanr(df[f], df.log10_toughness).correlation:+.3f}  "
              f"vs length {spearmanr(df[f], L[:, 0]).correlation:+.3f}  distinct values {df[f].round(5).nunique()}")
    for target in ["strength", "log10_toughness"] + FIBER:
        y = df[target].to_numpy()
        base, base_rho = grouped_r2(L, y)
        print(f"\n=== {target}: spline(length) grouped R² {base:.3f} (ρ {base_rho:.2f}) ===")
        for name, (B, pca) in BLOCKS.items():
            r2, rho = grouped_r2(np.hstack([L, B]), y, pca)
            print(f"  + {name:<18} R² {r2:.3f}  ΔR² {r2 - base:+.3f}  ρ {rho:.2f}")
