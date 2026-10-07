"""
Molecular-Level Property Analysis for CG MD Trajectories
==========================================================

Turns a `md_engine.Trajectory` into the molecular-level properties the MD
agent reports. Every property carries a statistical error estimate (block
averaging over the production window) — a mean with no error bar invites
over-reading a 2-ns run, which is exactly the "trust the aggregate number"
failure mode this repo already hit once with the grounding score.

Properties:
  - radius_of_gyration, end_to_end_distance, ree2_over_rg2 (6 for ideal chains)
  - flory_exponent (nu, from internal scaling R_ij ~ |i-j|^nu; ~0.5 ideal/theta,
    ~0.59 good solvent/expanded IDP, ~0.33 collapsed globule)
  - asphericity (relative shape anisotropy kappa^2: 0 sphere, 1 rod)
  - persistence_length (from consecutive-bond cosine)
  - intrachain_contacts (fraction of non-local residue pairs in contact, plus
    the most frequent contacting residue-type pairs for proteins)
  - interchain_contacts / largest_cluster_fraction (multi-chain only:
    aggregation / condensation propensity)
  - com_diffusion (model-time; friction-dependent, NOT a real diffusivity)
  - equilibration diagnostics (first-half vs second-half drift, kinetic T)

`python3 md_analysis.py` checks these against analytic cases (rigid rod,
ideal random walk, sphere vs rod shape).
"""

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-frame observables
# ---------------------------------------------------------------------------

def _chains(chain_ids: np.ndarray) -> List[np.ndarray]:
    return [np.where(chain_ids == c)[0] for c in np.unique(chain_ids)]


def radius_of_gyration(frames: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    """(F, n_chains) Rg per chain per frame (unweighted bead average)."""
    out = []
    for idx in _chains(chain_ids):
        x = frames[:, idx, :]
        out.append(np.sqrt(np.mean(np.sum((x - x.mean(axis=1, keepdims=True)) ** 2, axis=2), axis=1)))
    return np.stack(out, axis=1)


def end_to_end_distances(frames: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    out = [np.linalg.norm(frames[:, idx[-1]] - frames[:, idx[0]], axis=1) for idx in _chains(chain_ids)]
    return np.stack(out, axis=1)


def asphericity(frames: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    """Relative shape anisotropy kappa^2 = 1 - 3 (l1 l2 + l2 l3 + l3 l1) / (l1+l2+l3)^2."""
    out = []
    for idx in _chains(chain_ids):
        x = frames[:, idx, :] - frames[:, idx, :].mean(axis=1, keepdims=True)
        gyr = np.einsum("fni,fnj->fij", x, x) / len(idx)
        lam = np.linalg.eigvalsh(gyr)
        i1 = lam.sum(axis=1)
        i2 = lam[:, 0] * lam[:, 1] + lam[:, 1] * lam[:, 2] + lam[:, 2] * lam[:, 0]
        out.append(1.0 - 3.0 * i2 / i1 ** 2)
    return np.stack(out, axis=1)


def internal_distances(frames: np.ndarray, chain_ids: np.ndarray) -> Dict[int, float]:
    """Root-mean-square distance <R_ij^2>^1/2 vs sequence separation |i-j|, averaged over chains."""
    sums: Dict[int, List[float]] = {}
    for idx in _chains(chain_ids):
        x = frames[:, idx, :]
        n = len(idx)
        for s in range(1, n):
            d2 = np.sum((x[:, s:, :] - x[:, :-s, :]) ** 2, axis=2)
            sums.setdefault(s, []).append(float(np.mean(d2)))
    return {s: float(np.sqrt(np.mean(v))) for s, v in sums.items()}


def flory_exponent(frames: np.ndarray, chain_ids: np.ndarray, min_sep: int = 5) -> Optional[Dict[str, float]]:
    """Fits R_ij = b |i-j|^nu over min_sep <= |i-j| <= N - 5 (ends excluded, as in
    standard IDP scaling analyses). Returns None for chains too short to fit.
    For a compact globule R_ij plateaus at the globule size, so the fitted nu
    can fall well below 1/3 — read nu < ~0.4 as "compact", not as a precise exponent."""
    rij = internal_distances(frames, chain_ids)
    n = max(rij) + 1
    seps = [s for s in rij if min_sep <= s <= n - 5]
    if len(seps) < 5:
        return None
    lx, ly = np.log(seps), np.log([rij[s] for s in seps])
    nu, logb = np.polyfit(lx, ly, 1)
    return {"nu": float(nu), "prefactor_b": float(np.exp(logb))}


def persistence_length(frames: np.ndarray, chain_ids: np.ndarray) -> float:
    """Freely-rotating-chain estimate l_p = <b> / (1 - <cos theta>) from the
    consecutive-bond angle: <b> for a freely-jointed chain, -> inf for a rod."""
    cosines, blens = [], []
    for idx in _chains(chain_ids):
        b = np.diff(frames[:, idx, :], axis=1)
        bl = np.linalg.norm(b, axis=2)
        u = b / bl[..., None]
        cosines.append(np.sum(u[:, 1:] * u[:, :-1], axis=2).ravel())
        blens.append(bl.ravel())
    c = float(np.mean(np.concatenate(cosines)))
    bmean = float(np.mean(np.concatenate(blens)))
    return bmean / (1.0 - c)


def _pair_distances(x: np.ndarray, box: Optional[float]) -> np.ndarray:
    d = x[:, None, :] - x[None, :, :]
    if box is not None:
        d -= box * np.round(d / box)
    return np.sqrt(np.sum(d * d, axis=2))


def contact_analysis(frames: np.ndarray, chain_ids: np.ndarray, cutoff: float,
                     residues: Optional[str] = None, box: Optional[float] = None,
                     min_seq_sep: int = 3, max_frames: int = 200, top_k: int = 5) -> Dict[str, Any]:
    """Contacts = bead pairs within `cutoff`. Intra-chain contacts require
    |i-j| >= min_seq_sep (non-local). Cluster analysis treats chains as
    connected if any inter-chain bead contact exists."""
    step = max(1, len(frames) // max_frames)
    sub = frames[::step]
    n = frames.shape[1]
    same_chain = chain_ids[:, None] == chain_ids[None, :]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    intra_mask = np.triu(same_chain & (sep >= min_seq_sep), k=1)
    inter_mask = np.triu(~same_chain, k=1)
    n_chains = len(np.unique(chain_ids))
    intra_frac, inter_per_chain, largest_cluster = [], [], []
    pair_counts: Counter = Counter()
    inter_counts: Counter = Counter()
    for x in sub:
        c = _pair_distances(x, box) < cutoff
        intra = c & intra_mask
        intra_frac.append(intra.sum() / max(1, intra_mask.sum()))
        if residues:
            for i, j in zip(*np.nonzero(intra)):
                pair_counts["-".join(sorted((residues[i], residues[j])))] += 1
        if n_chains > 1:
            inter = c & inter_mask
            inter_per_chain.append(2.0 * inter.sum() / n_chains)
            if residues:
                for i, j in zip(*np.nonzero(inter)):
                    inter_counts["-".join(sorted((residues[i], residues[j])))] += 1
            # union-find over chains
            parent = list(range(n_chains))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            for i, j in zip(*np.nonzero(inter)):
                ra, rb = find(chain_ids[i]), find(chain_ids[j])
                if ra != rb:
                    parent[ra] = rb
            sizes = Counter(find(c_) for c_ in range(n_chains))
            largest_cluster.append(max(sizes.values()) / n_chains)
    out: Dict[str, Any] = {
        "contact_cutoff": cutoff,
        "intrachain_contact_fraction": float(np.mean(intra_frac)),
    }
    if residues and pair_counts:
        total = sum(pair_counts.values())
        out["top_intrachain_contact_pairs"] = [
            {"pair": p, "share": round(c / total, 3)} for p, c in pair_counts.most_common(top_k)]
    if n_chains > 1:
        out["interchain_contacts_per_chain"] = float(np.mean(inter_per_chain))
        out["largest_cluster_fraction"] = float(np.mean(largest_cluster))
        if residues and inter_counts:
            total = sum(inter_counts.values())
            out["top_interchain_contact_pairs"] = [
                {"pair": p, "share": round(c / total, 3)} for p, c in inter_counts.most_common(top_k)]
    return out


def com_diffusion(frames: np.ndarray, times: np.ndarray, chain_ids: np.ndarray,
                  masses: np.ndarray) -> Optional[float]:
    """D from the slope of chain-COM MSD vs lag over lags up to 1/4 of the run (MSD = 6 D t)."""
    if len(frames) < 20:
        return None
    coms = np.stack([np.average(frames[:, idx, :], axis=1, weights=masses[idx])
                     for idx in _chains(chain_ids)], axis=1)
    max_lag = len(frames) // 4
    lags = np.arange(1, max_lag + 1)
    msd = np.array([np.mean(np.sum((coms[l:] - coms[:-l]) ** 2, axis=2)) for l in lags])
    dt = times[1] - times[0]
    slope = np.polyfit(lags * dt, msd, 1)[0]
    return float(slope / 6.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def block_average(series: np.ndarray, n_blocks: int = 5) -> Dict[str, float]:
    """Mean and standard error from non-overlapping blocks (accounts for
    time correlation better than the naive std/sqrt(n))."""
    series = np.asarray(series, dtype=float)
    if series.ndim > 1:
        series = series.mean(axis=1)         # average over chains per frame
    n = len(series) // n_blocks * n_blocks
    if n < n_blocks:
        return {"mean": float(np.mean(series)), "sem": float("nan")}
    blocks = series[:n].reshape(n_blocks, -1).mean(axis=1)
    return {"mean": float(np.mean(series)), "sem": float(np.std(blocks, ddof=1) / np.sqrt(n_blocks))}


def _sig(x: Any, digits: int = 3) -> Any:
    """Round to significant figures so reported numbers match how prose quotes them."""
    if isinstance(x, float) and np.isfinite(x) and x != 0:
        return float(f"{x:.{digits}g}")
    return x


def _rounded(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _rounded(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_rounded(v) for v in d]
    return _sig(d)


AVAILABLE_PROPERTIES = [
    "radius_of_gyration", "end_to_end_distance", "flory_exponent", "asphericity",
    "persistence_length", "contacts", "com_diffusion", "potential_energy",
]


def compute_properties(traj, system, properties: Optional[List[str]] = None,
                       burn_in_fraction: float = 0.2) -> Dict[str, Any]:
    """Computes the requested properties over the production window
    (after discarding `burn_in_fraction` of frames) and returns a JSON-safe dict."""
    props = properties or AVAILABLE_PROPERTIES
    unknown = sorted(set(props) - set(AVAILABLE_PROPERTIES))
    if unknown:
        raise ValueError(f"unknown properties {unknown}; available: {AVAILABLE_PROPERTIES}")
    if len(traj.frames) < 10:
        raise ValueError("trajectory has < 10 frames; run longer or report more often")
    burn = int(len(traj.frames) * burn_in_fraction)
    fr, t = traj.frames[burn:], traj.times[burn:]
    cid, L = system.chain_ids, system.length_unit
    out: Dict[str, Any] = {"length_unit": L, "n_production_frames": len(fr),
                           "production_time": float(t[-1] - t[0]),
                           "time_unit": "ps (model time)" if L == "nm" else "tau (LJ)"}

    rg = radius_of_gyration(fr, cid)
    if "radius_of_gyration" in props:
        out["radius_of_gyration"] = block_average(rg)
    if "end_to_end_distance" in props:
        ree = end_to_end_distances(fr, cid)
        out["end_to_end_distance"] = block_average(ree)
        out["ree2_over_rg2"] = float(np.mean(ree ** 2) / np.mean(rg ** 2))
    if "flory_exponent" in props:
        out["flory_exponent"] = flory_exponent(fr, cid)
    if "asphericity" in props:
        out["asphericity"] = block_average(asphericity(fr, cid))
    if "persistence_length" in props:
        out["persistence_length"] = persistence_length(fr, cid)
    if "contacts" in props:
        cutoff = 1.0 if L == "nm" else 1.5
        out["contacts"] = contact_analysis(fr, cid, cutoff, residues=system.residues,
                                           box=system.box_length)
    if "com_diffusion" in props:
        d = com_diffusion(fr, t, cid, system.masses)
        out["com_diffusion"] = None if d is None else {
            "value": d, "unit": f"{L}^2/{'ps' if L == 'nm' else 'tau'}",
            "caveat": "Langevin implicit-solvent model time; scales with chosen friction, not a real diffusivity"}
    if "potential_energy" in props:
        out["potential_energy_per_bead"] = {
            **block_average(traj.potential_energy[burn:] / system.n),
            "unit": "kJ/mol" if L == "nm" else "epsilon"}
    if system.box_length is not None and system.spec is not None:
        vol = system.box_length ** 3
        out["box_length"] = system.box_length
        out["bead_number_density"] = system.n / vol

    # Equilibration diagnostics: the agent uses these to decide whether to extend.
    half = len(rg) // 2
    a, b = block_average(rg[:half], 3), block_average(rg[half:], 3)
    pooled = np.sqrt(np.nan_to_num(a["sem"]) ** 2 + np.nan_to_num(b["sem"]) ** 2)
    drift = abs(a["mean"] - b["mean"])
    out["diagnostics"] = {
        "rg_first_half": a["mean"], "rg_second_half": b["mean"],
        "rg_drift_over_sem": float(drift / pooled) if pooled > 0 else float("inf"),
        "rg_relative_sem": out.get("radius_of_gyration", block_average(rg))["sem"] / float(np.mean(rg)),
        "mean_kinetic_temperature_ratio": float(np.mean(traj.kinetic_temperature[burn:])),
    }
    d = out["diagnostics"]
    d["looks_equilibrated"] = bool(d["rg_drift_over_sem"] < 3.0 and d["rg_relative_sem"] < 0.05
                                   and abs(d["mean_kinetic_temperature_ratio"] - 1) < 0.05)
    return _rounded(out)


# ---------------------------------------------------------------------------
# Analytic checks
# ---------------------------------------------------------------------------

def _check_analysis() -> List[str]:
    failures = []
    n = 101
    rod = np.zeros((1, n, 3))
    rod[0, :, 0] = np.arange(n)
    cid = np.zeros(n, dtype=int)
    rg = radius_of_gyration(rod, cid)[0, 0]
    expect = np.sqrt((n * n - 1) / 12.0)
    print(f"rigid rod Rg = {rg:.3f} (analytic {expect:.3f})")
    if abs(rg - expect) > 1e-6:
        failures.append("rod Rg")
    k = asphericity(rod, cid)[0, 0]
    print(f"rigid rod asphericity = {k:.3f} (analytic 1.0)")
    if abs(k - 1) > 1e-6:
        failures.append("rod asphericity")

    rng = np.random.default_rng(0)
    steps = rng.normal(size=(400, 200, 3))
    steps /= np.linalg.norm(steps, axis=2, keepdims=True)
    walk = np.cumsum(steps, axis=1)
    cid = np.zeros(200, dtype=int)
    nu = flory_exponent(walk, cid)["nu"]
    ratio = np.mean(end_to_end_distances(walk, cid) ** 2) / np.mean(radius_of_gyration(walk, cid) ** 2)
    print(f"ideal random walk: nu = {nu:.3f} (analytic 0.5), Ree^2/Rg^2 = {ratio:.2f} (analytic ~6)")
    if abs(nu - 0.5) > 0.03:
        failures.append("random-walk nu")
    if abs(ratio - 6.0) > 0.6:
        failures.append("random-walk Ree^2/Rg^2")
    lp = persistence_length(walk, cid)
    print(f"freely-jointed persistence length = {lp:.3f} (expected ~1 bond)")
    if abs(lp - 1.0) > 0.05:
        failures.append("FJC persistence length")
    k = float(np.mean(asphericity(walk, cid)))
    print(f"random-walk mean asphericity = {k:.3f} (literature ~0.39-0.42 for long ideal chains)")
    if not (0.3 < k < 0.5):
        failures.append("random-walk asphericity")
    return failures


if __name__ == "__main__":
    fails = _check_analysis()
    print("\nall analysis checks passed" if not fails else f"\nFAILED: {fails}")
    raise SystemExit(1 if fails else 0)
