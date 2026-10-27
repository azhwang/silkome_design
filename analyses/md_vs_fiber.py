"""Does single-chain MD unfolding mechanics track measured fiber mechanics across species?
Fragments are mapped to species by exact substring match against Silkome genes."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import json, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import silkome_fiber_model as m

D = REPO + "/"
md = pd.read_csv(D + "d5ma00154d1_suppl.csv")
md["seq"] = md.seq.str.upper()
md["L"] = md.seq.str.len()
# length-adjusted molecular properties: residual of log value on log length (fit on all fragments)
for col in ["strength", "toughness"]:
    y, x = np.log(md[col]), np.log(md.L)
    b = np.polyfit(x, y, 2)
    md[col + "_resid"] = y - np.polyval(b, x)
frag = md.groupby("seq")[["strength", "toughness", "strength_resid", "toughness_resid", "L"]].mean()

genes = []
for line in open(D + "silkome_data/genes.jsonl"):
    r = json.loads(line)
    for g in r["genes"]:
        s = (g.get("amino_acid_sequence") or "").upper().rstrip("*")
        if s:
            genes.append((s, r["genus"] + " " + r["species"], g["protein_name"]))

per_species = defaultdict(list)
n_unique = 0
mapping = []
for f in frag.index:
    hits = [g for g in genes if f in g[0]]
    sp = {g[1] for g in hits}
    mapping.append({"fragment": f, "n_species": len(sp), "species": ";".join(sorted(sp)),
                    "gene_types": ";".join(sorted({g[2] for g in hits}))})
    if len(sp) == 1:
        n_unique += 1
        per_species[sp.pop()].append(f)
print(f"{n_unique} fragments map to exactly one species; {len(per_species)} species")
map_path = os.path.join(REPO, "silkome_data", "md_fragment_species_map.csv")
pd.DataFrame(mapping).to_csv(map_path, index=False)
print(f"wrote {map_path} ({sum(r['n_species'] > 0 for r in mapping)} of {len(mapping)} fragments matched)")

S = m.load_species_table().set_index("taxon")
rows = []
for sp, fs in per_species.items():
    if sp not in S.index:
        continue
    sub = frag.loc[fs]
    rows.append({"taxon": sp, "family": S.loc[sp, "family"], "n_frag": len(fs),
                 **{f"md_{c}": sub[c].mean() for c in ["strength", "toughness", "strength_resid", "toughness_resid", "L"]},
                 **{p: S.loc[sp, t] for p, t in [("fiber_strength", "log_strength"), ("fiber_toughness", "log_toughness"),
                                                 ("fiber_modulus", "log_modulus"), ("fiber_strain", "log_strain")]}})
T = pd.DataFrame(rows)
print(f"{len(T)} species with both MD fragments and measured fibers; median {int(T.n_frag.median())} fragments/species; "
      f"families {T.family.value_counts().head(4).to_dict()}\n")

def within(x, y, fam):
    d = pd.DataFrame({"x": x, "y": y, "f": fam}).dropna()
    d = d[d.f.map(d.f.value_counts()) >= 3]
    dm = d.groupby("f")[["x", "y"]].transform(lambda c: c - c.mean())
    return spearmanr(dm.x, dm.y).correlation, len(dm)

md_cols = ["md_strength", "md_toughness", "md_strength_resid", "md_toughness_resid", "md_L"]
fib_cols = ["fiber_strength", "fiber_toughness", "fiber_modulus", "fiber_strain"]
print("Spearman ρ across species (within-family ρ in brackets)")
print(f"{'MD (species mean)':<22}" + "".join(f"{c:>22}" for c in fib_cols))
for mc in md_cols:
    cells = []
    for fc in fib_cols:
        ok = T[[mc, fc]].dropna().index
        r = spearmanr(T.loc[ok, mc], T.loc[ok, fc]).correlation
        rw, n = within(T[mc], T[fc], T.family)
        cells.append(f"{r:+.2f} [{rw:+.2f}] n={len(ok)}")
    print(f"{mc:<22}" + "".join(f"{c:>22}" for c in cells))

# null scale: |rho| expected by chance at this n
n = len(T)
print(f"\nchance level: |ρ| > {1.96 / np.sqrt(n - 1):.2f} is nominally significant at n = {n} (20 tests, uncorrected)")
