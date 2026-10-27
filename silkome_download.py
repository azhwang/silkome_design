"""
Download the measured Spider Silkome dataset (Arakawa et al. 2022, Sci Adv
8:eabo6043) from the public spider-silkome.org API.

Writes into silkome_data/:
  mechanical_properties.json : one record per measured silk sample (446),
                               with toughness / Young's modulus / tensile
                               strength / strain at break (+SD), crystallinity,
                               birefringence, diameter, ... and individual_id
  individuals.json           : all 1,774 sequenced individuals (id -> species)
  genes.jsonl                : spidroin gene records (amino-acid sequence,
                               protein_name e.g. MaSp1/MaSp2/MiSp, side =
                               nterm/cterm/...) for every individual of every
                               species that has measured properties

These are *real* fiber-level lab measurements, unlike the `pre_*` columns in
d5ma00154d2_suppl.csv (which are SilkomeGPT model outputs).

Requests are rate-limited and results cached, so re-running only fetches
what's missing.  Usage:  python3 silkome_download.py
"""

import json
import os
import time
import urllib.request

BASE = "https://spider-silkome.org"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silkome_data")
DELAY_S = 0.25  # be polite to a small academic server


def get_json(path: str):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "Mozilla/5.0 (silkome research)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # transient network errors: back off and retry
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry {path}: {e}")


def cached(name: str, fetch):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = fetch()
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    props = cached("mechanical_properties.json",
                   lambda: get_json("/api/mechanical_properties.json?per_page=5000"))
    individuals = cached("individuals.json", lambda: get_json("/api/individuals.json?per_page=5000"))
    print(f"{len(props)} measured silk samples, {len(individuals)} sequenced individuals")

    measured_species = {(p["genus"], p["species"]) for p in props}
    measured_individuals = {p["individual_id"] for p in props}
    targets = [i for i in individuals
               if (i["genus"], i["species"]) in measured_species or i["id"] in measured_individuals]
    print(f"{len(targets)} individuals belong to measured species -> fetching their genes")

    genes_path = os.path.join(OUT, "genes.jsonl")
    done = set()
    if os.path.exists(genes_path):
        with open(genes_path) as f:
            done = {json.loads(line)["individual_id"] for line in f}
    with open(genes_path, "a") as f:
        for k, ind in enumerate(targets):
            if ind["id"] in done:
                continue
            genes = get_json(f"/api/individuals/{ind['id']}/genes.json")
            f.write(json.dumps({"individual_id": ind["id"], "idv_id": ind["idv_id"],
                                "genus": ind["genus"], "species": ind["species"],
                                "genes": genes}) + "\n")
            f.flush()
            time.sleep(DELAY_S)
            if k % 50 == 0:
                print(f"  {k}/{len(targets)}")
    print("done")


if __name__ == "__main__":
    main()
