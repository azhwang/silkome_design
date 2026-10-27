"""
Tool-free sequence -> mechanical-property regressor (silk-specific baseline)
===========================================================================

The *control* for the tool-augmented reasoning pipeline: how much of
strength/toughness is predictable directly from sequence, with no tools
and no reasoning? If the reasoning model can't beat this, it isn't adding
accuracy (only, at best, interpretability).

Data: d5ma00154d1_suppl.csv — ~2.2k spidroin fragments (median ~120 aa)
with per-fragment `strength` / `toughness` (these carry a `force_vector`,
i.e. they look like simulated pulling values, not fiber-level lab
measurements — keep that in mind when interpreting results).

Design decisions (each one guards against a specific way to fool yourself):
  - Identical sequences appear multiple times with slightly different
    labels (~5% CV). They are averaged into one row, and their spread is
    reported as a noise ceiling on achievable R².
  - Near-duplicate fragments (homologs across species) are clustered on
    3-mer Jaccard distance and CV folds are split by cluster
    (GroupKFold). A random-KFold score is reported alongside purely to
    show how much a naive split would inflate results.
  - Toughness is heavy-tailed (CV ~0.6, max ~15x median) -> modeled as
    log10(toughness).
  - Trivial baselines (predict mean, length-only) are always reported: a
    fancy model that can't beat length-only has learned nothing.

Feature sets:
  length     : sequence length only
  comp       : length + 20 amino-acid frequencies
  biotools   : comp + scalar outputs of bio_tools.py (hydropathy, charge,
               Chou-Fasman, repeat score, motif counts, disorder)
  dipep      : comp + 400 dipeptide frequencies
  esm        : mean-pooled ESM-2 embeddings (needs `--esm-model`; weights
               are downloaded from Hugging Face on first use and the
               embeddings cached to .npy)

Usage:
  python3 silk_regressor.py                      # baselines, no downloads
  python3 silk_regressor.py --esm-model facebook/esm2_t12_35M_UR50D
"""

import argparse
import json
import os
import re
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bio_tools

DEFAULT_DATA_PATH = "d5ma00154d1_suppl.csv"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Toughness in this data is ~entirely explained by chain length (length-only
# grouped ρ≈0.97 — longer simulated chains extend further; normalizing per
# residue doesn't remove it, length-only still gets R²≈0.80). So every model
# is also reported as ΔR² over the best length-only model: that
# column, not raw R², says whether sequence *content* was learned.
TARGETS = ["strength", "log10_toughness"]
SEED = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load fragments, average replicate sequences, return (df, noise stats)."""
    raw = pd.read_csv(path)
    raw = raw.dropna(subset=["seq"])
    raw["seq"] = raw["seq"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    raw = raw[raw["seq"].str.len() > 0]
    raw["log10_toughness"] = np.log10(raw["toughness"])

    # Replicate noise: variance of labels among identical sequences, as a
    # fraction of total label variance -> ceiling on explainable variance.
    noise = {}
    for t in TARGETS:
        sub = raw.dropna(subset=[t])
        dup = sub[sub.duplicated("seq", keep=False)]
        within_var = dup.groupby("seq")[t].var(ddof=1).mean()
        noise[t] = float(1 - within_var / sub[t].var(ddof=1))

    agg = raw.groupby("seq", as_index=False).agg(
        strength=("strength", "mean"),
        log10_toughness=("log10_toughness", "mean"),
        n_replicates=("seq", "size"),
    )
    return agg, noise


def kmer_set(seq: str, k: int = 3) -> set:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def cluster_sequences(seqs: List[str], cut: float = 0.7, k: int = 3) -> np.ndarray:
    """Average-linkage clustering on k-mer Jaccard distance.

    `cut` is a Jaccard *distance*; 3-mer Jaccard distance 0.7 corresponds
    roughly to ~50-60% sequence identity for fragments this length, i.e.
    homologs from related species land in the same cluster.
    """
    from scipy.sparse import csr_matrix
    vocab: Dict[str, int] = {}
    rows, cols = [], []
    for i, seq in enumerate(seqs):
        for km in kmer_set(seq, k):
            rows.append(i)
            cols.append(vocab.setdefault(km, len(vocab)))
    m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(seqs), len(vocab)))
    inter = (m @ m.T).toarray()
    size = np.asarray(m.sum(1)).ravel()
    union = size[:, None] + size[None, :] - inter
    dist = 1 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, t=cut, criterion="distance")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def composition(seq: str) -> np.ndarray:
    n = max(len(seq), 1)
    return np.array([seq.count(a) / n for a in AMINO_ACIDS])


_DIPEPTIDES = ["".join(p) for p in product(AMINO_ACIDS, repeat=2)]
_DIPEP_INDEX = {d: i for i, d in enumerate(_DIPEPTIDES)}


def dipeptide_composition(seq: str) -> np.ndarray:
    v = np.zeros(len(_DIPEPTIDES))
    for i in range(len(seq) - 1):
        idx = _DIPEP_INDEX.get(seq[i:i + 2])
        if idx is not None:
            v[idx] += 1
    return v / max(len(seq) - 1, 1)


def biotools_features(seq: str) -> np.ndarray:
    hyd = bio_tools.hydropathy_profile(seq)
    chg = bio_tools.charge_profile(seq)
    beta = bio_tools.beta_sheet_propensity(seq)
    rep = bio_tools.repeat_pattern_analyzer(seq)
    dis = bio_tools.disorder_prediction_chplot(seq)
    mot = bio_tools.motif_detector(seq)
    n = max(len(seq), 1)
    poly_a = [len(m) for m in re.findall(r"A{4,}", seq)]
    return np.array([
        hyd["mean_hydropathy"], hyd["max_window_value"], hyd["min_window_value"],
        chg["net_charge"], chg["mean_net_charge_per_residue"],
        beta["mean_alpha_propensity"], beta["mean_beta_propensity"],
        beta["mean_turn_propensity"], len(beta["beta_rich_spans"]) / n,
        rep["repeat_score"], rep["dominant_period"] or 0,
        dis["mean_hydropathy_normalized"], float(dis["predicted_disordered"]),
        len(mot["motifs"]),
        sum(len(v) for v in mot["spans"].values()) / n,
        len(re.findall(r"GG[A-Z]", seq)) / n,
        len(re.findall(r"GPG", seq)) / n,
        sum(poly_a) / n, max(poly_a, default=0),
    ], dtype=float)


def esm_embeddings(seqs: List[str], model_name: str, cache_dir: str = "embeddings",
                   batch_size: int = 8) -> np.ndarray:
    """Mean-pooled last-layer ESM-2 embeddings, cached per model to .npy."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, model_name.replace("/", "__") + ".npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["seqs"]) == list(seqs):
            return z["emb"]

    import torch
    from transformers import AutoTokenizer, EsmModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    out = []
    order = np.argsort([len(s) for s in seqs])  # length-sorted batches = less padding
    with torch.no_grad():
        for b in range(0, len(seqs), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([seqs[i] for i in idx], return_tensors="pt", padding=True).to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0  # drop <cls>
            lengths = enc["attention_mask"].sum(1) - 1
            mask[torch.arange(len(idx)), lengths] = 0  # drop <eos>
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            out.append((idx, pooled.float().cpu().numpy()))
            if (b // batch_size) % 25 == 0:
                print(f"  embedded {min(b + batch_size, len(seqs))}/{len(seqs)}")
    emb = np.zeros((len(seqs), out[0][1].shape[1]), dtype=np.float32)
    for idx, e in out:
        emb[idx] = e
    np.savez(cache, emb=emb, seqs=np.array(seqs, dtype=object))
    return emb


def build_feature_sets(seqs: List[str], esm_model: str = None,
                       only: List[str] = None) -> Dict[str, np.ndarray]:
    """Feature matrices by name. `length` is always included (ΔR² reference)."""
    wanted = lambda name: only is None or name in only
    length = np.array([[len(s)] for s in seqs], dtype=float)
    comp = np.hstack([length, np.vstack([composition(s) for s in seqs])])
    sets = {"length": length}
    if wanted("comp"):
        sets["comp"] = comp
    biotools = None
    if wanted("biotools") or (esm_model and wanted("esm+biotools")):
        biotools = np.hstack([comp, np.vstack([biotools_features(s) for s in seqs])])
        if wanted("biotools"):
            sets["biotools"] = biotools
    if wanted("dipep"):
        sets["dipep"] = np.hstack([comp, np.vstack([dipeptide_composition(s) for s in seqs])])
    if esm_model:
        emb = esm_embeddings(seqs, esm_model)
        if wanted("esm"):
            sets["esm"] = np.hstack([length, emb])
        if wanted("esm+biotools"):
            sets["esm+biotools"] = np.hstack([biotools, emb])
    return sets


# ---------------------------------------------------------------------------
# Models + evaluation
# ---------------------------------------------------------------------------

def make_models() -> Dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
        "hgb": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED),
    }


def cross_validate(X: np.ndarray, y: np.ndarray, splitter, groups=None) -> Dict[str, float]:
    """Out-of-fold predictions for each model -> pooled Spearman / R²."""
    results = {}
    for name, model in make_models().items():
        oof = np.full_like(y, np.nan, dtype=float)
        for tr, te in splitter.split(X, y, groups):
            model.fit(X[tr], y[tr])
            oof[te] = model.predict(X[te])
        results[name] = {
            "spearman": float(spearmanr(y, oof).correlation),
            "r2": float(r2_score(y, oof)),
            "oof": oof,
        }
    return results


def run(data_path: str, esm_model: str = None, cluster_cut: float = 0.7,
        n_folds: int = 5, out_path: str = "silk_regressor_results.json",
        feature_sets: List[str] = None) -> dict:
    df, noise = load_data(data_path)
    seqs = df["seq"].tolist()
    print(f"{len(df)} unique sequences "
          f"({int(df.n_replicates.sum())} rows before replicate averaging)")

    groups = cluster_sequences(seqs, cut=cluster_cut)
    sizes = np.bincount(groups)[1:]
    print(f"{len(sizes)} sequence clusters at 3-mer Jaccard distance {cluster_cut} "
          f"(largest {sizes.max()}, singletons {(sizes == 1).sum()})")

    features = build_feature_sets(seqs, esm_model, feature_sets)
    group_cv = GroupKFold(n_splits=n_folds)
    random_cv = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    report = {"n_sequences": len(df), "n_clusters": int(len(sizes)),
              "cluster_cut": cluster_cut, "esm_model": esm_model,
              "noise_ceiling_r2": noise, "results": {}}
    best_oof = {}
    for target in TARGETS:
        mask = df[target].notna().to_numpy()
        y = df.loc[mask, target].to_numpy()
        g = groups[mask]
        print(f"\n=== {target}  (n={mask.sum()}, replicate-noise R² ceiling "
              f"≈ {noise[target]:.2f}) ===")
        print(f"{'features':<14}{'model':<7}{'grouped ρ':>10}{'grouped R²':>12}"
              f"{'ΔR² vs len':>12}{'random ρ':>10}{'random R²':>11}")
        report["results"][target] = {}
        best = (-np.inf, None)
        length_r2 = {}
        for fname, X in features.items():
            Xm = X[mask]
            grp = cross_validate(Xm, y, group_cv, g)
            rnd = cross_validate(Xm, y, random_cv)
            for m in grp:
                if fname == "length":
                    length_r2[m] = grp[m]["r2"]
                # vs the *best* length-only model: length->toughness is
                # nonlinear, so a linear length fit is too easy to "beat".
                gain = grp[m]["r2"] - max(length_r2.values())
                print(f"{fname:<14}{m:<7}{grp[m]['spearman']:>10.3f}{grp[m]['r2']:>12.3f}"
                      f"{gain:>+12.3f}{rnd[m]['spearman']:>10.3f}{rnd[m]['r2']:>11.3f}")
                report["results"][target][f"{fname}/{m}"] = {
                    "delta_r2_vs_length": gain,
                    "grouped": {k: v for k, v in grp[m].items() if k != "oof"},
                    "random": {k: v for k, v in rnd[m].items() if k != "oof"},
                }
                if grp[m]["r2"] > best[0]:
                    best = (grp[m]["r2"], f"{fname}/{m}", grp[m]["oof"])
        report["results"][target]["best_grouped"] = best[1]
        best_oof[target] = (mask, best[2])
        print(f"best (grouped CV): {best[1]}  R²={best[0]:.3f}")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    oof_df = df[["seq"] + TARGETS].copy()
    oof_df["cluster"] = groups
    for target, (mask, oof) in best_oof.items():
        oof_df.loc[mask, f"pred_{target}"] = oof
    oof_df.to_csv(out_path.replace(".json", "_oof.csv"), index=False)
    print(f"\nwrote {out_path} and {out_path.replace('.json', '_oof.csv')}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--esm-model", default=None,
                    help="e.g. facebook/esm2_t12_35M_UR50D, facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--cluster-cut", type=float, default=0.7)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", default="silk_regressor_results.json")
    ap.add_argument("--feature-sets", nargs="+", default=None,
                    help="subset of: comp biotools dipep esm esm+biotools "
                         "(length is always included)")
    args = ap.parse_args()
    run(args.data_path, args.esm_model, args.cluster_cut, args.n_folds, args.out,
        args.feature_sets)
