"""Which of the *other* measured Silkome properties can sequence features predict on held-out genera?

silkome_fiber_model.py only models the four mechanical properties. The database
also has crystallinity, birefringence, diameter, body size, thermal weight loss,
water content and supercontraction for the same silk samples. For each property
this reports (1) the noise ceiling: how much of its variance is species-level
signal at all (one-way ICC from replicate samples, adjusted for the number of
samples behind each species mean), and (2) held-out-genus R^2 of the same
feature sets silkome_fiber_model.py uses. Section 2 tests whether diameter-
adjusting strength changes anything; section 3 asks whether measured structure
(diameter, birefringence, crystallinity) explains mechanics at all.

Usage: python3 analyses/other_properties.py [--quick]
"""
import argparse
import json
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import silkome_fiber_model as m

# field -> (target name, log-transform?)
TARGETS = {
    "toughness": ("log_toughness", True),
    "youngs_modulus": ("log_modulus", True),
    "tensile_strength": ("log_strength", True),
    "strain_at_break": ("log_strain", True),
    "crystallinity": ("crystallinity", False),
    "birefringence": ("birefringence", False),
    "diameter": ("log_diameter", True),
    "body_size": ("log_body_size", True),
    "water_content": ("water_content", False),
    "one_percent_weightloss": ("T_1pct_loss", False),
    "five_percent_weightloss": ("T_5pct_loss", False),
    "ten_percent_weightloss": ("T_10pct_loss", False),
    "supercontraction": ("supercontraction", False),
}
MECHANICAL = ["log_toughness", "log_modulus", "log_strength", "log_strain"]
FEATURE_SETS = ["family", "genes", "motifs", "genes+motifs", "family+genes+motifs"]


def load_samples() -> pd.DataFrame:
    p = pd.DataFrame(json.load(open(os.path.join(m.DATA, "mechanical_properties.json"))))
    p["taxon"] = p.genus + " " + p.species
    for field, (name, log) in TARGETS.items():
        v = p[field].map(m._num)
        p[name] = np.log(v.where(v > 0)) if log else v
    return p


def icc_species(p: pd.DataFrame, col: str):
    """One-way ICC(1) at species level from replicate samples; returns (icc, n_rep_species)."""
    d = p[["taxon", col]].dropna()
    n_per = d.groupby("taxon")[col].transform("size")
    d = d[n_per >= 2]
    k = d.taxon.nunique()
    if k < 8:
        return np.nan, k
    g = d.groupby("taxon")[col]
    n = len(d)
    sizes = g.size()
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    msb = (sizes * (g.mean() - d[col].mean()) ** 2).sum() / (k - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (n - k)
    return float((msb - msw) / (msb + (n0 - 1) * msw)), k


def ceiling(icc: float, k_per_species: np.ndarray) -> float:
    """Mean reliability of species means given how many samples back each one."""
    if np.isnan(icc):
        return np.nan
    icc = max(icc, 0.0)
    return float(np.mean(k_per_species * icc / (1 + (k_per_species - 1) * icc)))


def species_table(p: pd.DataFrame) -> pd.DataFrame:
    df = m.load_species_table()
    df = df[df["n_masp"] > 0].reset_index(drop=True)
    cols = [t[0] for t in TARGETS.values()]
    sp = p.groupby("taxon")[cols].mean()
    for c in cols:
        if c not in MECHANICAL:  # mechanical log targets already on df from load_species_table
            df[c] = df.taxon.map(sp[c])
    df["k_total"] = df.taxon.map(p.groupby("taxon").size())
    return df


def sweep(df, p, features, n_rep):
    m.N_REPEATS = n_rep
    rows = []
    for name, _ in [(t[0], 0) for t in TARGETS.values()]:
        mask = df[name].notna().to_numpy()
        y = df.loc[mask, name].to_numpy()
        icc, k_rep = icc_species(p, name)
        k_i = p.dropna(subset=[name]).groupby("taxon").size().reindex(df.taxon[mask]).to_numpy()
        row = {"target": name, "n_species": int(mask.sum()), "ICC": icc, "n_rep_species": k_rep,
               "ceiling_R2": ceiling(icc, k_i)}
        for fs in FEATURE_SETS:
            X = features[fs][mask]
            X = np.where(np.isnan(X), np.nanmean(X, 0), X)
            r2, sd, rho = m.evaluate(X, y, df.genus.to_numpy()[mask], repeats=n_rep)
            row[fs] = r2
            row[fs + "_sd"] = sd
        rows.append(row)
        print(f"  {name:<18}n={row['n_species']:<4}ICC={icc:5.2f} ceiling={row['ceiling_R2']:5.2f}  "
              + "  ".join(f"{fs}={row[fs]:+.2f}" for fs in FEATURE_SETS), flush=True)
    return pd.DataFrame(rows)


def adjusted_strength(p: pd.DataFrame):
    """Remove the within-species diameter effect from log strength; returns (adjusted col, beta)."""
    d = p.dropna(subset=["log_strength", "log_diameter"]).copy()
    d = d[d.groupby("taxon").taxon.transform("size") >= 2]
    cols = ["log_strength", "log_diameter"]
    dem = d[cols] - d.groupby("taxon")[cols].transform("mean")
    fit = sm.OLS(dem["log_strength"], dem[["log_diameter"]]).fit(
        cov_type="cluster", cov_kwds={"groups": d.taxon.astype("category").cat.codes})
    beta = float(fit.params["log_diameter"])
    p = p.copy()
    p["log_strength_adj"] = p["log_strength"] - beta * (p["log_diameter"] - p["log_diameter"].mean())
    return p, beta, float(fit.pvalues["log_diameter"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 repeats instead of 5")
    ap.add_argument("--out", default=os.path.join(REPO, "analyses", "other_properties_results.json"))
    args = ap.parse_args()
    n_rep = 2 if args.quick else m.N_REPEATS

    p = load_samples()
    df = species_table(p)
    print(f"{len(p)} samples; {len(df)} species with >=1 MaSp sequence ({df.genus.nunique()} genera)\n")
    features = m.build_features(df)

    print("## 1. Sequence-feature R^2 on HELD-OUT GENERA, every measured property")
    res = sweep(df, p, features, n_rep)
    res["best"] = res[FEATURE_SETS].max(axis=1)
    res["best_set"] = res[FEATURE_SETS].idxmax(axis=1)
    res["gain_over_family"] = res["best"] - res["family"]
    print("\n" + res[["target", "n_species", "ICC", "ceiling_R2", "family", "best", "best_set", "gain_over_family"]]
          .round(2).to_string(index=False))

    print("\n## 2. Does removing the diameter effect make strength more predictable?")
    p2, beta, pval = adjusted_strength(p)
    sp = p2.groupby("taxon")["log_strength_adj"].mean()
    df["log_strength_adj"] = df.taxon.map(sp)
    mask = df["log_strength_adj"].notna().to_numpy()
    print(f"within-species beta(log diameter -> log strength) = {beta:+.2f} (p={pval:.3f})")
    adj = {}
    for label, col in [("raw", "log_strength"), ("diameter-adjusted", "log_strength_adj")]:
        mk = (df["log_strength"].notna() & df["log_strength_adj"].notna()).to_numpy()
        y = df.loc[mk, col].to_numpy()
        adj[label] = {}
        for fs in ["family", "motifs", "family+genes+motifs"]:
            X = features[fs][mk]
            X = np.where(np.isnan(X), np.nanmean(X, 0), X)
            r2, sd, _ = m.evaluate(X, y, df.genus.to_numpy()[mk], repeats=n_rep)
            adj[label][fs] = (r2, sd)
        icc, _ = icc_species(p2, col)
        print(f"  {label:<18} ICC={icc:.2f}  " + "  ".join(f"{fs}={v[0]:+.3f}±{v[1]:.3f}" for fs, v in adj[label].items()))

    print("\n## 3. Could structure be a bridge? Measured structure -> mechanics, held-out genera")
    print("   (diagnostic only: these are lab measurements, not available from sequence)")
    structure = df[["log_diameter", "birefringence", "crystallinity", "log_body_size", "water_content"]]
    bridge = {}
    for label, cols in [("diameter+birefringence", ["log_diameter", "birefringence"]),
                        ("+crystallinity", ["log_diameter", "birefringence", "crystallinity"])]:
        bridge[label] = {}
        for tgt in MECHANICAL:
            mk = (structure[cols].notna().all(axis=1) & df[tgt].notna()).to_numpy()
            if mk.sum() < 40:
                continue
            r2, sd, _ = m.evaluate(structure.loc[mk, cols].to_numpy(), df.loc[mk, tgt].to_numpy(),
                                   df.genus.to_numpy()[mk], repeats=n_rep)
            bridge[label][tgt] = (int(mk.sum()), r2, sd)
        print(f"  {label:<24}" + "  ".join(f"{t.replace('log_', '')}: R2={v[1]:+.2f} (n={v[0]})"
                                           for t, v in bridge[label].items()))

    out = {"n_species": len(df), "sweep": res.to_dict(orient="records"),
           "diameter_adjustment": {"beta": beta, "p": pval, "results": adj},
           "structure_bridge": bridge}
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
