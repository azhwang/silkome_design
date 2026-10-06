"""
LLM reasoning traces vs. tool-free regressors, on the same sequences
=====================================================================

Question: do the tool-grounded reasoning traces (orpo_trace_pool.py output)
predict strength/toughness better than cheap baselines? Compared on the
sequences present in BOTH the ORPO run and the regressors' out-of-fold
(cluster-grouped CV) predictions from silk_regressor.py.

Currency is Spearman rho: strength/toughness are predicted in different
units by different systems (LLM: [0,1] normalized; regressors: raw strength
and log10 toughness), and rank correlation is invariant to those monotone
transforms. LLM predictions are averaged over a sequence's samples
(ensembling), which flatters the LLM, so conclusions are conservative.

Absolute error is reported only for LLM vs. predict-the-mean, in normalized
units, with the mean taken from dataset rows NOT in the evaluation set.

Usage:
  python3 llm_vs_regressor_comparison.py --orpo orpo_pairs_n500_backfilled.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REGRESSORS = {
    "tool-free regressor (best of length/comp/biotools/dipep)": "silk_regressor_results",
    "ESM-2 35M regressor": "silk_regressor_results_esm35M",
    "ESM-2 650M regressor": "silk_regressor_results_esm650M",
}
TARGETS = {"strength": "strength", "toughness": "log10_toughness"}


def rho(a, b):
    return float(spearmanr(a, b).correlation)


def partial_rho(x, y, z):
    """Spearman of x,y after regressing both (rank-transformed) on z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    res = lambda r: r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def bootstrap_ci(fn, n, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orpo", default="orpo_pairs_n500_backfilled.json")
    ap.add_argument("--data-path", default="d5ma00154d1_suppl.csv")
    ap.add_argument("--out", default="llm_vs_regressor_results.json")
    args = ap.parse_args()

    traces = json.load(open(args.orpo))["scored_traces"]
    rows = [
        {"seq": t["metadata"]["sequence"], "kind": t["metadata"]["trace_kind"],
         "strength": t["final_answer"]["strength"], "toughness": t["final_answer"]["toughness"]}
        for t in traces if t["final_answer"]
    ]
    n_dropped = len(traces) - len(rows)
    llm = pd.DataFrame(rows).groupby(["seq", "kind"], as_index=False)[["strength", "toughness"]].mean()

    oof = {name: pd.read_csv(stem + "_oof.csv").set_index("seq") for name, stem in REGRESSORS.items()}
    best_label = {name: json.load(open(stem + ".json"))["results"] for name, stem in REGRESSORS.items()}
    base = oof[next(iter(REGRESSORS))]
    seqs = sorted(set(llm.seq) & set(base.index))
    print(f"{len(set(llm.seq))} unique LLM sequences, {len(seqs)} also in regressor OOF "
          f"({n_dropped} unparseable traces dropped)")

    truth = base.loc[seqs]
    length = np.array([len(s) for s in seqs], dtype=float)
    n = len(seqs)
    preds = {}
    for kind, label in [("grounded_prompt", "LLM traces, tools given"), ("shortcut_prompt", "LLM traces, tools withheld")]:
        sub = llm[llm.kind == kind].set_index("seq").loc[seqs]
        preds[label] = {t: sub[t].to_numpy() for t in TARGETS}
    for name in REGRESSORS:
        o = oof[name].loc[seqs]
        preds[name] = {"strength": o["pred_strength"].to_numpy(), "toughness": o["pred_log10_toughness"].to_numpy()}
    preds["length only (rank by length)"] = {t: length for t in TARGETS}

    y = {t: truth[col].to_numpy() for t, col in TARGETS.items()}
    out = {"n_sequences": n, "n_unparseable_traces_dropped": n_dropped, "spearman": {}, "length_diagnostics": {}}

    print(f"\nSpearman rho vs. truth (n={n} sequences; 95% bootstrap CI)")
    print(f"{'predictor':<58}{'strength':>20}{'toughness':>20}")
    for label, p in preds.items():
        cells, out["spearman"][label] = [], {}
        for t in TARGETS:
            v = rho(p[t], y[t])
            ci = bootstrap_ci(lambda i: rho(p[t][i], y[t][i]), n)
            out["spearman"][label][t] = {"rho": v, "ci95": ci}
            cells.append(f"{v:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
        print(f"{label:<58}{cells[0]:>20}{cells[1]:>20}")

    print("\nDoes the LLM track chain length? (rho with length; partial rho vs truth controlling for length)")
    for label in ("LLM traces, tools given", "LLM traces, tools withheld"):
        out["length_diagnostics"][label] = {}
        for t in TARGETS:
            p = preds[label][t]
            d = {"rho_pred_vs_length": rho(p, length), "rho_truth_vs_length": rho(y[t], length),
                 "partial_rho_vs_truth_given_length": partial_rho(p, y[t], length),
                 "partial_ci95": bootstrap_ci(lambda i: partial_rho(p[i], y[t][i], length[i]), n)}
            out["length_diagnostics"][label][t] = d
            print(f"  {label:<28}{t:<10} pred~length {d['rho_pred_vs_length']:+.2f} | truth~length "
                  f"{d['rho_truth_vs_length']:+.2f} | partial vs truth {d['partial_rho_vs_truth_given_length']:+.2f} "
                  f"[{d['partial_ci95'][0]:+.2f},{d['partial_ci95'][1]:+.2f}]")

    full = pd.read_csv(args.data_path)
    full["seq"] = full["seq"].str.upper()
    held_in = set(seqs)
    rest = full[~full.seq.isin(held_in)]
    in_set = full[full.seq.isin(held_in)]
    out["absolute_error_normalized"] = {}
    print(f"\nMean absolute error, normalized [0,1] units (mean baseline from {len(rest)} rows outside the eval set)")
    for t, ncol in [("strength", "strength_norm"), ("toughness", "toughness_norm")]:
        tn = in_set.groupby("seq")[ncol].mean().loc[seqs].to_numpy()
        mean_pred = rest[ncol].mean()
        res = {"predict_mean": float(np.mean(np.abs(tn - mean_pred))), "mean_value_used": float(mean_pred)}
        for label in ("LLM traces, tools given", "LLM traces, tools withheld"):
            res[label] = float(np.mean(np.abs(tn - preds[label][t])))
        out["absolute_error_normalized"][t] = res
        print(f"  {t:<10}" + "  ".join(f"{k}: {v:.3f}" for k, v in res.items() if k != "mean_value_used"))

    # Same "correct" definition as orpo_trace_pool.py: mean abs error over both properties <= tolerance.
    tol = 0.1
    tn_both = np.column_stack([in_set.groupby("seq")[c].mean().loc[seqs].to_numpy()
                               for c in ("strength_norm", "toughness_norm")])
    const = np.array([rest["strength_norm"].mean(), rest["toughness_norm"].mean()])
    out["correct_within_tolerance"] = {"tolerance": tol,
        "constant_mean_guess": float(np.mean(np.abs(tn_both - const).mean(1) <= tol))}
    for label in ("LLM traces, tools given", "LLM traces, tools withheld"):
        guess = np.column_stack([preds[label]["strength"], preds[label]["toughness"]])
        out["correct_within_tolerance"][label] = float(np.mean(np.abs(tn_both - guess).mean(1) <= tol))
    print(f"\nFraction 'correct' (mean abs err <= {tol}, per-sequence averaged predictions): "
          + "  ".join(f"{k}: {v:.0%}" for k, v in out["correct_within_tolerance"].items() if k != "tolerance"))

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
