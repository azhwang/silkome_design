"""
Fiber-level property model on the *measured* Spider Silkome data
=================================================================

Predicts dragline-silk toughness / Young's modulus / tensile strength /
strain at break (real lab measurements, Arakawa et al. 2022) from each
species' spidroin sequences. Data comes from `silkome_download.py`.

Why this is set up the way it is:
  - Unit of prediction = species. Properties are averaged over the measured
    individuals of a species; sequences are pooled over all sequenced
    individuals of that species (transcriptomes and silk samples are often
    from different individuals).
  - Sequences are partial: N-terminal and C-terminal halves of spidroin
    genes (terminal domain + some adjacent repeats). The repeat part is
    approximated by trimming the terminal domains (~150 aa NTD side,
    ~110 aa CTD side) — that's where poly-Ala / GGX motifs live.
  - Related species share both sequence and silk properties. A random split
    would mostly test "recognize the genus". Folds are split by GENUS (main
    result) and by FAMILY (hardest test), repeated over several random genus
    -> fold assignments because n is small.
  - The baseline that matters is phylogeny: "predict the mean of the
    species' family". Sequence features are only interesting if they beat
    or add to it.

Feature sets:
  family      : one-hot family (phylogeny baseline)
  genes       : which spidroin types were found (MaSp1/2/2B/3, MiSp, ...)
  motifs      : MaSp repeat-region composition + silk motif statistics
                (poly-Ala runs, GA, GGX, GPGXX, QQ, ...)
  esm         : ESM-2 mean embedding over the species' MaSp sequences
                (needs cached HF weights; --esm-model)

Usage:
  python3 silkome_fiber_model.py
  python3 silkome_fiber_model.py --esm-model facebook/esm2_t33_650M_UR50D
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "silkome_data")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TARGETS = {  # silkome field -> log-transformed target name
    "toughness": "log_toughness",
    "youngs_modulus": "log_modulus",
    "tensile_strength": "log_strength",
    "strain_at_break": "log_strain",
}
GENE_TYPES = ["MaSp1", "MaSp2", "MaSp2B", "MaSp3", "MaSp", "Ampullate spidroin", "MiSp",
              "Flag", "AcSp", "AgSp1", "AgSp2", "PySp", "CySp", "Spidroin"]


def is_dragline(gene_type: str) -> bool:
    """Major-ampullate (dragline) spidroins. Some species' records only carry
    the generic label "Ampullate spidroin" — same gland, so count them."""
    return gene_type.startswith("MaSp") or gene_type == "Ampullate spidroin"
NTD_TRIM, CTD_TRIM = 150, 110
N_REPEATS = 5
SEED = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _num(v):
    if v is None:
        return np.nan
    v = str(v).replace("±", "").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan  # "ND", ""


def load_species_table() -> pd.DataFrame:
    with open(os.path.join(DATA, "mechanical_properties.json")) as f:
        props = pd.DataFrame(json.load(f))
    for col in list(TARGETS) + ["crystallinity"]:
        props[col] = props[col].map(_num)
    props["taxon"] = props["genus"] + " " + props["species"]

    genes_by_taxon: Dict[str, Dict[str, dict]] = defaultdict(dict)
    with open(os.path.join(DATA, "genes.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            taxon = rec["genus"] + " " + rec["species"]
            for g in rec["genes"]:
                seq = (g.get("amino_acid_sequence") or "").upper().rstrip("*")
                if seq:  # dedupe identical sequences across individuals
                    genes_by_taxon[taxon][seq] = {"type": g["protein_name"], "side": g["side"]}

    agg = props.groupby("taxon").agg(
        family=("family", "first"), genus=("genus", "first"),
        n_samples=("id", "size"), crystallinity=("crystallinity", "mean"),
        **{f: (f, "mean") for f in TARGETS})
    for f, t in TARGETS.items():
        agg[t] = np.log(agg[f].where(agg[f] > 0))
    agg["genes"] = [genes_by_taxon.get(t, {}) for t in agg.index]
    agg["n_masp"] = [sum(is_dragline(v["type"]) for v in g.values()) for g in agg["genes"]]
    return agg.reset_index()


def masp_repeat_regions(genes: Dict[str, dict]) -> List[str]:
    """Repeat-region parts of MaSp sequences, terminal domains trimmed off."""
    out = []
    for seq, g in genes.items():
        if not is_dragline(g["type"]):
            continue
        rep = seq[NTD_TRIM:] if g["side"] == "nterm" else seq[:-CTD_TRIM]
        if len(rep) >= 20:
            out.append(rep)
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def gene_presence(genes: Dict[str, dict]) -> np.ndarray:
    types = {g["type"] for g in genes.values()}
    return np.array([float(t in types) for t in GENE_TYPES])


MOTIF_NAMES = ([f"frac_{a}" for a in AMINO_ACIDS] +
               ["polyA_frac", "polyA_mean_run", "polyA_max_run", "GA_frac", "GGX_per_res",
                "GPGXX_per_res", "QQ_per_res", "SS_per_res", "GGY_per_res", "repeat_len_log"])


def motif_features(regions: List[str]) -> np.ndarray:
    if not regions:
        return np.full(len(MOTIF_NAMES), np.nan)
    s = "".join(regions)
    n = len(s)
    comp = [s.count(a) / n for a in AMINO_ACIDS]
    runs = [len(m) for m in re.findall(r"A{4,}", s)]
    ga = sum(len(m) for m in re.findall(r"(?:GA){2,}", s))
    return np.array(comp + [
        sum(runs) / n, np.mean(runs) if runs else 0.0, max(runs, default=0),
        ga / n,
        len(re.findall(r"GG[A-Z]", s)) / n,
        len(re.findall(r"GPG[A-Z]{2}", s)) / n,
        len(re.findall(r"QQ", s)) / n,
        len(re.findall(r"SS", s)) / n,
        len(re.findall(r"GGY", s)) / n,
        np.log(n),
    ])


def esm_species_embedding(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Mean ESM-2 embedding over each species' MaSp sequences (cached)."""
    cache = os.path.join(HERE, "embeddings", "silkome_" + model_name.replace("/", "__") + ".npz")
    seqs = sorted({s for g in df["genes"] for s, v in g.items() if is_dragline(v["type"])})
    emb = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        emb = dict(zip(z["seqs"], z["emb"]))
    missing = [s for s in seqs if s not in emb]
    if missing:
        import torch
        from transformers import AutoTokenizer, EsmModel
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name).to(dev).eval()
        # Pad to multiples of 64 so MPS sees a handful of tensor shapes, not
        # ~1,000 distinct lengths (per-shape graph caching made an unpadded
        # run hang for hours).
        for k, s in enumerate(missing):
            s_trunc = s[:1022]
            pad_to = min(1024, -(-(len(s_trunc) + 2) // 64) * 64)
            enc = tok([s_trunc], return_tensors="pt", padding="max_length",
                      max_length=pad_to).to(dev)
            with torch.no_grad():
                h = model(**enc).last_hidden_state[0, 1:len(s_trunc) + 1]
            emb[s] = h.mean(0).float().cpu().numpy()
            if k % 100 == 0:
                print(f"  embedded {k}/{len(missing)}", flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, seqs=np.array(list(emb), dtype=object), emb=np.array(list(emb.values())))
    dim = len(next(iter(emb.values())))
    out = []
    for g in df["genes"]:
        vs = [emb[s] for s, v in g.items() if is_dragline(v["type"])]
        out.append(np.mean(vs, 0) if vs else np.full(dim, np.nan))
    return np.array(out)


def build_features(df: pd.DataFrame, esm_model: str = None) -> Dict[str, np.ndarray]:
    family = pd.get_dummies(df["family"]).to_numpy(float)
    genes = np.vstack([gene_presence(g) for g in df["genes"]])
    motifs = np.vstack([motif_features(masp_repeat_regions(g)) for g in df["genes"]])
    sets = {
        "family": family,
        "genes": genes,
        "motifs": motifs,
        "genes+motifs": np.hstack([genes, motifs]),
        "family+genes+motifs": np.hstack([family, genes, motifs]),
    }
    if esm_model:
        esm = esm_species_embedding(df, esm_model)
        sets["esm"] = esm
        sets["family+genes+motifs+esm"] = np.hstack([family, genes, motifs, esm])
    return sets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_model(n_features: int):
    steps = [StandardScaler()]
    if n_features > 64:  # embeddings: compress before ridge, n is only a few hundred
        steps.append(PCA(16, random_state=SEED))
    return make_pipeline(*steps, Ridge())


def random_group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole groups (genera / families) to folds at random."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % n_folds))
    return np.array([fold_of[g] for g in groups])


def evaluate(X, y, groups, n_folds=5, repeats=N_REPEATS):
    r2s, rhos = [], []
    for rep in range(repeats):
        folds = random_group_folds(groups, n_folds, SEED + rep)
        oof = np.full(len(y), np.nan)
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            pipe = make_model(X.shape[1])
            inner = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-1, 4, 11)},
                                 cv=GroupKFold(4), scoring="r2")
            inner.fit(X[tr], y[tr], groups=groups[tr])
            oof[te] = inner.predict(X[te])
        r2s.append(r2_score(y, oof))
        rhos.append(spearmanr(y, oof).correlation)
    return np.mean(r2s), np.std(r2s), np.mean(rhos)


def run(esm_model: str = None, out_path: str = "silkome_fiber_results.json") -> dict:
    df = load_species_table()
    df = df[df["n_masp"] > 0].reset_index(drop=True)  # need at least one MaSp sequence
    print(f"{len(df)} species with measured properties and ≥1 MaSp sequence "
          f"({df.genus.nunique()} genera, {df.family.nunique()} families; "
          f"median {int(df.n_masp.median())} MaSp seqs/species)")

    features = build_features(df, esm_model)
    report = {"n_species": len(df), "results": {}}
    for split in ["genus", "family"]:
        groups = df[split].to_numpy()
        print(f"\n######## held-out {split.upper()} ({N_REPEATS} random fold assignments) ########")
        for target in TARGETS.values():
            mask = df[target].notna().to_numpy()
            y = df.loc[mask, target].to_numpy()
            print(f"\n=== {target} (n={mask.sum()}) ===")
            print(f"{'features':<26}{'R² mean':>9}{'± sd':>7}{'ρ':>7}")
            for name, X in features.items():
                Xm = X[mask]
                Xm = np.where(np.isnan(Xm), np.nanmean(Xm, 0), Xm)  # rare: no repeat region
                if split == "family" and name.startswith("family"):
                    continue  # family one-hot is meaningless when the family is held out
                r2, sd, rho = evaluate(Xm, y, groups[mask])
                print(f"{name:<26}{r2:>9.3f}{sd:>7.3f}{rho:>7.2f}")
                report["results"].setdefault(split, {}).setdefault(target, {})[name] = {
                    "r2": r2, "r2_sd": sd, "spearman": rho}

    # Interpretability: univariate associations of motif features with each
    # property, within families (family means removed) so they aren't
    # just phylogeny.
    motifs = features["motifs"]
    print("\n######## motif ↔ property, within-family Spearman ρ (top 6 per target) ########")
    report["within_family_motif_rho"] = {}
    for target in TARGETS.values():
        sub = pd.DataFrame(motifs, columns=MOTIF_NAMES)
        sub["y"], sub["family"] = df[target], df["family"]
        sub = sub.dropna()
        sub = sub[sub.family.map(sub.family.value_counts()) > 2]
        dem = sub.groupby("family")[MOTIF_NAMES + ["y"]].transform(lambda c: c - c.mean())
        rho = {m: spearmanr(dem[m], dem["y"]).correlation for m in MOTIF_NAMES}
        top = sorted(rho.items(), key=lambda kv: -abs(np.nan_to_num(kv[1])))[:6]
        print(f"{target:<14} " + "  ".join(f"{m} {r:+.2f}" for m, r in top))
        report["within_family_motif_rho"][target] = rho

    with open(os.path.join(HERE, out_path), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--out", default="silkome_fiber_results.json")
    args = ap.parse_args()
    run(args.esm_model, args.out)
