"""Fold-proxy features from ESM-2 650M predicted contact maps (cached HF weights).
Writes embeddings/esm650M_contact_features.npz, used by eval_both.py."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import silk_regressor as sr

D = REPO + "/"
NAMES = ["lr_contacts_per_res",      # sum P over |i-j|>=24, per residue  (tertiary fold)
         "mr_contacts_per_res",      # 12<=|i-j|<24
         "sr_contacts_per_res",      # 6<=|i-j|<12
         "frac_res_with_lr_contact", # residues with any P>0.3 at |i-j|>=12
         "helix_i4",                 # mean P(i,i+4)  (helical H-bond geometry)
         "hairpin_antidiag",         # mean of max anti-diagonal P (beta-hairpin pairing)
         "max_lr_prob"]

def features_from_contacts(P):
    L = P.shape[0]
    i, j = np.triu_indices(L, 1)
    sep, p = j - i, P[i, j]
    lr = p[sep >= 24].sum() / L
    mr = p[(sep >= 12) & (sep < 24)].sum() / L
    sr_ = p[(sep >= 6) & (sep < 12)].sum() / L
    Q = P.copy(); np.fill_diagonal(Q, 0)
    band = np.abs(np.subtract.outer(np.arange(L), np.arange(L))) >= 12
    frac_lr = float(((Q * band) > 0.3).any(1).mean())
    h4 = float(np.mean(np.diag(P, 4))) if L > 4 else 0.0
    flipped = np.fliplr(Q)
    anti = [np.diag(flipped, k).max() for k in range(-L + 8, L - 8) if len(np.diag(flipped, k))]
    hp = float(np.mean(anti)) if anti else 0.0
    mx = float(p[sep >= 12].max()) if (sep >= 12).any() else 0.0
    return [lr, mr, sr_, frac_lr, h4, hp, mx]

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, EsmModel
    df, _ = sr.load_data(D + sr.DEFAULT_DATA_PATH)
    seqs = df.seq.tolist()
    name = "facebook/esm2_t33_650M_UR50D"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name).to(dev).eval()

    def contacts(seq):
        enc = tok([seq], return_tensors="pt").to(dev)
        with torch.no_grad():
            return model.predict_contacts(enc["input_ids"], enc["attention_mask"])[0].float().cpu().numpy()

    # sanity check: a C-terminal-domain fragment (folded helical bundle) should show far
    # more long-range contacts than a Gly-rich repeat (disordered)
    ctd = "SRLSSPQASSRVSSAVSNLVASGPTNSAALSSTISNVVSQIGASNPGLSGCDVLIQALLEVVSALIQILGSSSIGQVNYGSAGQATQIVGQSVYQALG"
    ggx = "GPGGYGPGQQ" * 10
    for label, s in [("CTD-like", ctd), ("GGX repeat", ggx)]:
        print(label, dict(zip(NAMES, np.round(features_from_contacts(contacts(s)), 3))))
    F = []
    for k, s in enumerate(seqs):
        F.append(features_from_contacts(contacts(s)))
        if k % 200 == 0:
            print(f"  {k}/{len(seqs)}", flush=True)
    np.savez(D + "embeddings/esm650M_contact_features.npz", F=np.array(F), seqs=np.array(seqs, dtype=object), names=NAMES)
    print("done")
