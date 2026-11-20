"""Is the supercontraction signal in analyses/other_properties.py real?

other_properties.py found held-out-genus R^2 ~ +0.3 for supercontraction from
gene presence + repeat-region composition (family alone ~ +0.05), contradicting
the earlier supercontraction_cv.py result (R^2 -0.09 with 4 hand-picked motifs).
n is only 84 species, 68% Araneidae, so this script stress-tests it:

  1. estimate + cluster-bootstrap (over genera) 95% CI, repeat-averaged OOF predictions
  2. leave-one-genus-out R^2 (the criterion supercontraction_cv.py used)
  3. permutation null that shuffles y WITHIN families (does sequence add beyond family?)
  4. Araneidae only (removes the Araneidae-vs-rest contrast)
  5. feature ablation + within-family univariate correlations
  6. where the signal lives (Araneidae vs. other families) and whether it is just
     "how many spidroin types were recovered" (a transcriptome-completeness/lineage proxy)

Result: the signal is carried by non-dragline gene presence; one number (# spidroin
types found) matches the whole feature set, dragline motifs add nothing to it, and it
vanishes within Araneidae. Not a dragline sequence-mechanics relationship.

Usage: python3 analyses/supercontraction_validation.py [--perms 100]
"""
import argparse
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analyses"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

import other_properties as op
import silkome_fiber_model as m

TARGET = "supercontraction"
GRID = {"ridge__alpha": np.logspace(-1, 4, 11)}


def fit_predict(Xtr, ytr, gtr, Xte):
    inner = GridSearchCV(m.make_model(Xtr.shape[1]), GRID, cv=GroupKFold(min(4, len(np.unique(gtr)))), scoring="r2")
    inner.fit(Xtr, ytr, groups=gtr)
    return inner.predict(Xte)


def cv_oof(X, y, g, repeats, n_folds=5):
    """Held-out-genus OOF predictions, one row per repeat."""
    oofs = []
    for rep in range(repeats):
        folds = m.random_group_folds(g, n_folds, m.SEED + rep)
        oof = np.full(len(y), np.nan)
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            oof[te] = fit_predict(X[tr], y[tr], g[tr], X[te])
        oofs.append(oof)
    return np.array(oofs)


def logo_oof(X, y, g):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        oof[te] = fit_predict(X[tr], y[tr], g[tr], X[te])
    return oof


def mean_r2(oofs, y):
    return float(np.mean([r2_score(y, o) for o in oofs]))


def cluster_bootstrap_ci(pred, y, g, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    genera = np.unique(g)
    idx_by = {k: np.where(g == k)[0] for k in genera}
    vals = []
    for _ in range(reps):
        pick = rng.choice(genera, len(genera), replace=True)
        ix = np.concatenate([idx_by[k] for k in pick])
        vals.append(r2_score(y[ix], pred[ix]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def impute(X):
    return np.where(np.isnan(X), np.nanmean(X, 0), X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    p = op.load_samples()
    df = op.species_table(p)
    feats = m.build_features(df)
    mk = df[TARGET].notna().to_numpy()
    d = df[mk].reset_index(drop=True)
    y = d[TARGET].to_numpy()
    g = d.genus.to_numpy()
    fam = d.family.to_numpy()
    F = {k: impute(v[mk]) for k, v in feats.items()}
    print(f"{len(d)} species, {d.genus.nunique()} genera, {d.family.nunique()} families "
          f"({(fam == 'Araneidae').sum()} Araneidae); sd(y)={y.std():.1f}\n")

    print("## 1. Held-out-genus R^2 (5-fold, %d fold assignments) with cluster-bootstrap 95%% CI over genera" % args.repeats)
    oof = {}
    for name in ["family", "genes", "motifs", "genes+motifs", "family+genes+motifs"]:
        oof[name] = cv_oof(F[name], y, g, args.repeats)
        avg = oof[name].mean(0)
        lo, hi = cluster_bootstrap_ci(avg, y, g)
        print(f"  {name:<22} R2={mean_r2(oof[name], y):+.3f}   rho={spearmanr(y, avg).correlation:+.2f}   "
              f"[CI of repeat-averaged prediction: {lo:+.2f}, {hi:+.2f}]", flush=True)

    print("\n## 2. Leave-one-genus-out R^2 (criterion used by supercontraction_cv.py: -0.09 with 4 motifs)")
    for name in ["family", "genes+motifs", "family+genes+motifs"]:
        o = logo_oof(F[name], y, g)
        print(f"  {name:<22} R2={r2_score(y, o):+.3f}  rho={spearmanr(y, o).correlation:+.2f}", flush=True)

    print("\n## 3. Permutation null: shuffle y within families, rerun the same CV (%d perms x 3 repeats)" % args.perms)
    rng = np.random.default_rng(0)
    obs = {n: mean_r2(oof[n][:3], y) for n in ["genes+motifs", "family+genes+motifs"]}
    args.perms = max(args.perms, 0)
    null = {n: [] for n in obs}
    fam_idx = [np.where(fam == f)[0] for f in np.unique(fam)]
    for i in range(args.perms):  # --perms 0 skips this (slow) section
        yp = y.copy()
        for ix in fam_idx:
            yp[ix] = y[rng.permutation(ix)]
        for n in obs:
            null[n].append(mean_r2(cv_oof(F[n], yp, g, 3), yp))
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{args.perms}", flush=True)
    for n in obs if args.perms else []:
        a = np.array(null[n])
        pval = (1 + (a >= obs[n]).sum()) / (1 + len(a))
        print(f"  {n:<22} observed R2={obs[n]:+.3f} | null mean={a.mean():+.3f}, 95th pct={np.percentile(a, 95):+.3f}, "
              f"max={a.max():+.3f} | permutation p={pval:.3f}")

    print("\n## 4. Araneidae only (57 species): held-out-genus R^2, no Araneidae-vs-rest contrast available")
    ar = fam == "Araneidae"
    for name in ["genes", "motifs", "genes+motifs"]:
        o = cv_oof(F[name][ar], y[ar], g[ar], args.repeats)
        print(f"  {name:<22} R2={mean_r2(o, y[ar]):+.3f}  (n={ar.sum()}, genera={len(np.unique(g[ar]))})", flush=True)
    others = ~ar
    print(f"  (non-Araneidae n={others.sum()} too small to evaluate separately)")

    print("\n## 5. Ablations on genes+motifs and within-family correlations")
    names = [f"gene:{t}" for t in m.GENE_TYPES] + list(m.MOTIF_NAMES)
    X = F["genes+motifs"]
    keep = {
        "all genes+motifs": np.ones(X.shape[1], bool),
        "drop poly-Ala features": np.array([not n.startswith(("polyA", "frac_A")) for n in names]),
        "drop gene presence": np.array([not n.startswith("gene:") for n in names]),
        "gene presence only": np.array([n.startswith("gene:") for n in names]),
        "composition fractions only": np.array([n.startswith("frac_") for n in names]),
    }
    for label, cols in keep.items():
        o = cv_oof(X[:, cols], y, g, args.repeats)
        print(f"  {label:<28} R2={mean_r2(o, y):+.3f}  ({cols.sum()} features)", flush=True)

    dem = pd.DataFrame(X, columns=names)
    dem["y"], dem["fam"] = y, fam
    big = dem.fam.map(dem.fam.value_counts()) > 2
    dem = dem[big]
    dm = dem.groupby("fam")[names + ["y"]].transform(lambda c: c - c.mean())
    rho = {n: spearmanr(dm[n], dm["y"]).correlation for n in names if dm[n].std() > 0}
    top = sorted(rho.items(), key=lambda kv: -abs(np.nan_to_num(kv[1])))[:8]
    print("  strongest within-family Spearman rho (family means removed):")
    print("   " + "  ".join(f"{n} {r:+.2f}" for n, r in top))

    print("\n## 6. Where does the signal live, and is it just 'how many spidroin types were recovered'?")
    for name in ["genes+motifs", "family+genes+motifs"]:
        avg = cv_oof(F[name], y, g, args.repeats).mean(0)
        print(f"  {name:<22} rho all={spearmanr(y, avg).correlation:+.2f} | Araneidae={spearmanr(y[ar], avg[ar]).correlation:+.2f} "
              f"| non-Araneidae={spearmanr(y[~ar], avg[~ar]).correlation:+.2f}")
    n_types = np.array([[len({v["type"] for v in gg.values()})] for gg in d.genes], float)
    print(f"  # spidroin types found vs supercontraction: Spearman all={spearmanr(n_types[:, 0], y).correlation:+.2f}, "
          f"Araneidae={spearmanr(n_types[ar, 0], y[ar]).correlation:+.2f}")
    drag = [i for i, t in enumerate(m.GENE_TYPES) if t.startswith("MaSp") or t in ("Ampullate spidroin", "MiSp")]
    G = F["genes"]
    nondrag = [i for i in range(G.shape[1]) if i not in drag]
    for label, X, rows in [
        ("dragline/MiSp gene presence only", G[:, drag], slice(None)),
        ("non-dragline gene presence only", G[:, nondrag], slice(None)),
        ("# spidroin types found (1 number)", n_types, slice(None)),
        ("dragline repeat motifs only", F["motifs"], slice(None)),
        ("motifs + # types", np.hstack([F["motifs"], n_types]), slice(None)),
    ]:
        print(f"  {label:<38} R2={mean_r2(cv_oof(X, y, g, args.repeats), y):+.3f}", flush=True)
    for label, X in [("# types, Araneidae only", n_types[ar]), ("motifs + # types, Araneidae only", np.hstack([F["motifs"], n_types])[ar])]:
        print(f"  {label:<38} R2={mean_r2(cv_oof(X, y[ar], g[ar], args.repeats), y[ar]):+.3f}", flush=True)


if __name__ == "__main__":
    main()
