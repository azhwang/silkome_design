"""
Attribution check for the tool-free strength regressor
=======================================================

Question: when `silk_regressor.py`'s best strength model (length + amino-acid
composition, ridge; grouped-CV R² ≈ 0.32) predicts strength, is it using
sequence features that plausibly *cause* strength, or is composition just a
fingerprint for which protein family a fragment came from?

Four checks, each designed to separate those two explanations:

  1. Coefficient stability — refit the ridge on each grouped-CV training
     fold; a residue whose coefficient changes sign across folds is not a
     learned rule.
  2. Grouped permutation importance — shuffle one feature in the *held-out*
     cluster fold and measure the R² drop. Only features that help on
     unseen families count.
  3. Global vs within-cluster association — Spearman ρ of each residue's
     frequency with strength across the whole dataset, vs after subtracting
     cluster means (and partialling out length). A real per-residue effect
     should keep its sign within families; a family-identity proxy won't.
  4. Counterfactual edits — apply biologically motivated edits to real
     sequences (insert a poly-Ala block, convert GGX -> GAX, etc.) and see
     whether the model's predicted strength moves the way silk biology
     says it should.

Usage:
  python3 silk_attribution.py            # prints report, writes JSON
"""

import argparse
import json
import re
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import silk_regressor as sr

FEATURE_NAMES = ["length"] + [f"frac_{a}" for a in sr.AMINO_ACIDS]
SEED = 0

# What spider-silk biology would lead you to expect for *strength*.
# (+) = more of it should raise strength, (-) = lower, (0) = no strong prior.
# Poly-Ala blocks form the β-sheet nanocrystals that carry load; Gly-rich
# GGX / GPG regions are amorphous and mainly govern extensibility.
BIOLOGY_PRIOR = {"A": "+", "G": "-", "P": "-", "S": "0"}


def features(seqs: List[str]) -> np.ndarray:
    return np.array([[len(s)] + list(sr.composition(s)) for s in seqs], dtype=float)


# ---------------------------------------------------------------------------
# 1 + 2: fold-wise coefficients and grouped permutation importance
# ---------------------------------------------------------------------------

def fold_coefficients_and_importance(X, y, groups, n_folds=5, n_repeats=10):
    rng = np.random.default_rng(SEED)
    coefs, importances = [], []
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        model = sr.make_models()["ridge"].fit(X[tr], y[tr])
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        # Back to original units: MPa per unit feature. For composition
        # features, report per +1 percentage point (0.01 fraction).
        raw = ridge.coef_ / scaler.scale_
        raw[1:] *= 0.01
        coefs.append(raw)

        base = r2_score(y[te], model.predict(X[te]))
        imp = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            drops = []
            for _ in range(n_repeats):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - r2_score(y[te], model.predict(Xp)))
            imp[j] = np.mean(drops)
        importances.append(imp)
    return np.array(coefs), np.array(importances)


# ---------------------------------------------------------------------------
# 3: global vs within-cluster association
# ---------------------------------------------------------------------------

def _residualize(v: np.ndarray, on: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(on), on])
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def within_cluster_association(X, y, groups) -> pd.DataFrame:
    """ρ(feature, strength): global, within-cluster, within-cluster | length."""
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["y"], df["g"] = y, groups
    multi = (df.g.map(df.g.value_counts()) > 1).to_numpy()  # singletons have no within-variance
    demeaned = df.groupby("g")[FEATURE_NAMES + ["y"]].transform(lambda c: c - c.mean())[multi]
    len_w = demeaned["length"].to_numpy()
    y_w = demeaned["y"].to_numpy()
    y_wl = _residualize(y_w, len_w)
    rows = []
    for f in FEATURE_NAMES:
        x_w = demeaned[f].to_numpy()
        rows.append({
            "feature": f,
            "rho_global": spearmanr(df[f], df["y"]).correlation,
            "rho_within_cluster": spearmanr(x_w, y_w).correlation,
            "rho_within_cluster_given_length": (
                np.nan if f == "length" else
                spearmanr(_residualize(x_w, len_w), y_wl).correlation),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4: counterfactual edits
# ---------------------------------------------------------------------------

def _insert_poly_ala(seq: str, block: int = 8) -> str:
    """Replace a length-`block` window in the middle with poly-Ala (length kept)."""
    if len(seq) <= block:
        return "A" * len(seq)
    mid = (len(seq) - block) // 2
    return seq[:mid] + "A" * block + seq[mid + block:]


EDITS: Dict[str, Callable[[str], str]] = {
    # expected ↑ strength: more crystalline-forming Ala
    "insert poly-Ala block (8 aa, length-preserving)": _insert_poly_ala,
    "GGX -> GAX (Gly to Ala in amorphous repeats)": lambda s: re.sub(r"GG(?=[A-Z])", "GA", s),
    # expected ↓ strength: dilute crystalline blocks with Gly
    "poly-Ala -> poly-Gly (A{4,} blocks to G)": lambda s: re.sub(r"A{4,}", lambda m: "G" * len(m.group()), s),
    # control: length-changing edit, should move prediction via length only
    "duplicate first 20 residues (length +20)": lambda s: s[:20] + s,
}
EXPECTED = {
    "insert poly-Ala block (8 aa, length-preserving)": "+",
    "GGX -> GAX (Gly to Ala in amorphous repeats)": "+",
    "poly-Ala -> poly-Gly (A{4,} blocks to G)": "-",
    "duplicate first 20 residues (length +20)": "length",
}


def counterfactual_edits(seqs, X, y, groups, n_folds=5) -> pd.DataFrame:
    """Apply each edit to held-out sequences and predict with the fold model."""
    seqs = np.array(seqs, dtype=object)
    rows = []
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        model = sr.make_models()["ridge"].fit(X[tr], y[tr])
        base = model.predict(X[te])
        for name, edit in EDITS.items():
            edited = [edit(s) for s in seqs[te]]
            changed = np.array([e != s for e, s in zip(edited, seqs[te])])
            if not changed.any():
                continue
            delta = model.predict(features(edited)) - base
            rows += [{"edit": name, "delta": d} for d in delta[changed]]
    out = pd.DataFrame(rows).groupby("edit")["delta"].agg(
        n_edited="size", mean_delta_MPa="mean",
        frac_increase=lambda d: float((d > 0).mean())).reset_index()
    out["expected"] = out["edit"].map(EXPECTED)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(data_path: str, out_path: str) -> dict:
    df, _ = sr.load_data(data_path)
    df = df.dropna(subset=["strength"]).reset_index(drop=True)
    seqs = df["seq"].tolist()
    y = df["strength"].to_numpy()
    groups = sr.cluster_sequences(seqs)
    X = features(seqs)
    print(f"{len(seqs)} sequences, {len(set(groups))} clusters; "
          f"target = strength (mean {y.mean():.0f}, sd {y.std():.0f} MPa)\n")

    coefs, imps = fold_coefficients_and_importance(X, y, groups)
    assoc = within_cluster_association(X, y, groups)
    table = assoc.assign(
        coef_mean=coefs.mean(0), coef_sd=coefs.std(0),
        sign_stable=[bool(np.all(c > 0) or np.all(c < 0)) for c in coefs.T],
        perm_importance_R2=imps.mean(0),
    )
    table["within_cluster_sign_flip"] = (
        np.sign(table.rho_global) != np.sign(table.rho_within_cluster_given_length)
    ) & (table.feature != "length")
    table = table.sort_values("perm_importance_R2", ascending=False)

    pd.set_option("display.width", 200)
    print("== 1-3. Per-feature attribution (ridge, length + composition) ==")
    print("coef units: MPa per residue of length, or MPa per +1 percentage point of that residue")
    print(table.round(3).to_string(index=False))

    edits = counterfactual_edits(seqs, X, y, groups)
    print("\n== 4. Counterfactual edits on held-out sequences ==")
    print(edits.round(3).to_string(index=False))

    top = table[table.perm_importance_R2 > 0.005]
    robust = top[top.sign_stable & ~top.within_cluster_sign_flip]
    print("\n== Summary ==")
    print(f"features with grouped permutation importance > 0.005 R²: {top.feature.tolist()}")
    print(f"...of which sign-stable across folds AND same sign within clusters: "
          f"{robust.feature.tolist()}")
    for aa, prior in BIOLOGY_PRIOR.items():
        row = table[table.feature == f"frac_{aa}"].iloc[0]
        got = "+" if row.coef_mean > 0 else "-"
        print(f"  {aa}: biology prior {prior}, model coef {got} "
              f"({row.coef_mean:+.2f} MPa/pp), within-cluster ρ|len "
              f"{row.rho_within_cluster_given_length:+.2f}")

    report = {"attribution": table.to_dict(orient="records"),
              "counterfactual_edits": edits.to_dict(orient="records")}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-path", default=sr.DEFAULT_DATA_PATH)
    ap.add_argument("--out", default="silk_attribution_results.json")
    args = ap.parse_args()
    run(args.data_path, args.out)
