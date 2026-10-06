"""An invented screening library: about 10,000 small molecules built from cores, linkers and substituents, each a real
molecule (RDKit canonical SMILES, InChIKey, descriptors), a hidden binding rule for one target that turns on a few
pharmacophores and punishes bulk, and an assay history of a few hundred measured compounds. Everything is a pure function of a seed."""
import functools
import itertools

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
CORES = ["c1ccccc1", "c1ccncc1", "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1", "C1CCNCC1", "c1cnc2ccccc2n1", "c1ccsc1", "c1ccoc1", "c1cc2ccccc2o1", "C1CCOCC1"]
LINKERS = ["", "C", "CC", "C(=O)", "C(=O)N", "NC(=O)", "S(=O)(=O)N", "O", "N", "OC"]
SUBS = ["Cl", "F", "Br", "C", "CC", "OC", "N", "C(F)(F)F", "C(=O)O", "S(=O)(=O)N", "C#N", "O", "N(C)C", "C(=O)N", "c1ccccc1", "c1ccc(F)cc1", "C1CC1", "OC(F)(F)F"]
ACTIVE_LIMIT = 7.0                 # pIC50 at or above this counts as active
MEASURED_SEED_COMPOUNDS = 300
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def smiles_for(core, linker, sub1, sub2):
    """Core with up to two substituents: the first through the linker on the core's first atom, the second bare."""
    s = core
    if sub1:
        s = f"{sub1}{linker}" + core if linker or sub1 else core
    # attach by SMILES concatenation at the ring's first atom; RDKit tells us if it is not a molecule
    if sub2:
        s = s + f"({sub2})" if s.endswith("1") is False else s[:-1] + f"({sub2})1"
    return s


@functools.lru_cache(maxsize=4)
def build(seed=17, size=10000):
    r = np.random.default_rng(seed)
    combos = list(itertools.product(CORES, LINKERS, SUBS + [""], SUBS + [""]))
    r.shuffle(combos)
    seen, mols = set(), []
    for core, linker, s1, s2 in combos:
        m = Chem.MolFromSmiles(smiles_for(core, linker, s1, s2))
        if m is None:
            continue
        smi = Chem.MolToSmiles(m)
        if smi in seen:
            continue
        seen.add(smi)
        mols.append({"index": len(mols), "smiles": smi, "mol": m, "parts": (core, linker, s1, s2)})
        if len(mols) >= size:
            break
    fps = np.array([list(MORGAN.GetFingerprintAsNumPy(x["mol"])) for x in mols], dtype=np.uint8)
    desc = np.array([[Descriptors.MolWt(x["mol"]), Crippen.MolLogP(x["mol"]), rdMolDescriptors.CalcTPSA(x["mol"]), Lipinski.NumHDonors(x["mol"]), Lipinski.NumHAcceptors(x["mol"]), rdMolDescriptors.CalcNumRotatableBonds(x["mol"]), rdMolDescriptors.CalcNumAromaticRings(x["mol"])] for x in mols])
    # the hidden binding rule: a sulfonamide and a basic amine on an aromatic core bind; trifluoromethyl helps a little; bulk and grease hurt
    pharm = {"sulfonamide": Chem.MolFromSmarts("S(=O)(=O)N"), "aryl_amine": Chem.MolFromSmarts("c[NX3;H2,H1]"), "cf3": Chem.MolFromSmarts("C(F)(F)F"), "acid": Chem.MolFromSmarts("C(=O)[OH]"), "indole": Chem.MolFromSmarts("c1ccc2[nH]ccc2c1"), "piperidine": Chem.MolFromSmarts("C1CCNCC1")}
    has = {k: np.array([x["mol"].HasSubstructMatch(p) for x in mols]) for k, p in pharm.items()}
    truth = (5.0 + 1.4 * has["sulfonamide"] + 0.9 * has["aryl_amine"] + 0.8 * (has["sulfonamide"] & has["aryl_amine"]) + 0.4 * has["cf3"] - 0.9 * has["acid"] + 0.7 * has["indole"] - 0.5 * has["piperidine"]
             - 1.6 * (has["sulfonamide"] & has["piperidine"])                                      # the cliff: the basic ring kills the sulfonamide series
             - 0.004 * np.maximum(0, desc[:, 0] - 350) - 0.25 * np.maximum(0, desc[:, 1] - 3.5) ** 2 - 0.3 * (desc[:, 6] >= 3) + r.normal(0, 0.45, len(mols)))
    return {"seed": seed, "mols": mols, "fps": fps, "desc": desc, "truth": truth.round(3), "has": has, "rule": "pIC50 = 5 + 1.4 sulfonamide + 0.9 aryl amine + 0.8 both + 0.4 CF3 - 0.9 carboxylic acid + 0.7 indole - 0.5 piperidine - 1.6 sulfonamide with piperidine - 0.004 (MW - 350)+ - 0.25 (logP - 3.5)+^2 - 0.3 (3+ aromatic rings) + noise 0.45"}


def descriptors(mol):
    return {"mol_weight": round(Descriptors.MolWt(mol), 2), "logp": round(Crippen.MolLogP(mol), 2), "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1), "h_donors": Lipinski.NumHDonors(mol), "h_acceptors": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol), "heavy_atoms": mol.GetNumHeavyAtoms()}


def lipinski(d):
    fails = [k for k, bad in (("mol_weight", d["mol_weight"] > 500), ("logp", d["logp"] > 5), ("h_donors", d["h_donors"] > 5), ("h_acceptors", d["h_acceptors"] > 10)) if bad]
    return {"passes": len(fails) <= 1, "fails": fails}


def assay(w, idx, seed, replicates=2):
    """Measured pIC50 for compounds `idx`: the hidden value plus assay noise, replicated; values under 4.5 come back as '<4.5'."""
    r = np.random.default_rng([w["seed"], seed])
    out = []
    for i in idx:
        for rep in range(1, replicates + 1):
            v = float(w["truth"][i] + r.normal(0, 0.3))
            out.append({"index": int(i), "replicate": rep, "value": round(max(v, 4.5), 2), "qualifier": "<" if v < 4.5 else "="})
    return out


def history_indices(w, n=MEASURED_SEED_COMPOUNDS):
    """What a medicinal chemistry team would have measured first: a spread of the library with a bias toward what looked drug-like."""
    r = np.random.default_rng([w["seed"], 11])
    weight = np.exp(-0.002 * np.abs(w["desc"][:, 0] - 350))
    return np.sort(r.choice(len(w["mols"]), n, replace=False, p=weight / weight.sum()))
