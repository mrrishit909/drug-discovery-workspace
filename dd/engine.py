"""Affinity model with uncertainty, its calibration, the nearest-neighbour baseline, active-learning plate selection and the
trace from a recommendation back to measured neighbours and model features."""
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor

from . import world

VERSION_PREFIX = "affinity-ensemble"
MEMBERS = 5
Z90 = 1.645
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def features(fps, desc):
    return np.hstack([fps.astype(np.float32), desc.astype(np.float32)])


def aggregate(measurements):
    """One value per compound from its replicates: the mean; a compound measured only as '<4.5' is 4.5 (censored, kept)."""
    by = {}
    for m in measurements:
        by.setdefault(m["index"], []).append(float(m["value"]))
    return {k: float(np.mean(v)) for k, v in by.items()}


def train(x, y, seed=0):
    """Five gradient-boosted regressors on bootstrap resamples; their spread, rescaled on a held-out fifth so that the 90% band
    holds 90% of the time there, is the uncertainty. -> {"members", "scale", "floor"}"""
    r = np.random.default_rng(seed)
    hold = r.random(len(y)) < 0.2
    fit = lambda xs, ys: [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=5, min_samples_leaf=5, random_state=k).fit(xs[pick], ys[pick]) for k in range(MEMBERS) for pick in [r.integers(0, len(ys), len(ys))]]      # noqa: E731
    members = fit(x[~hold], y[~hold])
    p = np.array([m.predict(x[hold]) for m in members])
    raw = np.maximum(p.std(axis=0), 0.05)
    scale = float(np.quantile(np.abs(p.mean(axis=0) - y[hold]) / raw, 0.9) / Z90)       # one number makes the band honest on held-out data
    return {"members": fit(x, y), "scale": scale, "floor": 0.05, "held_out": int(hold.sum())}


def predict(model, x):
    p = np.array([m.predict(x) for m in model["members"]])
    return p.mean(axis=0), np.maximum(p.std(axis=0), model["floor"]) * model["scale"]


def bitvectors(fps):
    return [DataStructs.CreateFromBitString("".join(map(str, row))) for row in fps]


def knn_predict(fps_train, y_train, fps_query, k=3, bv_train=None):
    """The baseline: Tanimoto nearest neighbours' mean. Pass `bv_train` to reuse the training bit vectors."""
    bv_train = bv_train or bitvectors(fps_train)
    out, nn = np.empty(len(fps_query)), []
    for i, row in enumerate(fps_query):
        q = DataStructs.CreateFromBitString("".join(map(str, row)))
        sims = np.array(DataStructs.BulkTanimotoSimilarity(q, bv_train))
        top = np.argsort(-sims)[:k]
        out[i] = y_train[top].mean()
        nn.append([(int(j), round(float(sims[j]), 3)) for j in top])
    return out, nn


def calibration(mu, sigma, y):
    """How often the truth fell inside the 90% band, and the error by uncertainty tercile."""
    err = np.abs(mu - y)
    order = np.argsort(sigma)
    thirds = np.array_split(order, 3)
    return {"rmse": round(float(np.sqrt(np.mean((mu - y) ** 2))), 3), "mae": round(float(err.mean()), 3), "inside_90_band": round(float((err <= Z90 * sigma).mean()), 3),
            "by_uncertainty": [{"tercile": k + 1, "mean_sigma": round(float(sigma[t].mean()), 3), "mae": round(float(err[t].mean()), 3)} for k, t in enumerate(thirds)],
            "spearman_sigma_vs_error": round(float(np.corrcoef(np.argsort(np.argsort(sigma)), np.argsort(np.argsort(err)))[0, 1]), 3)}


def select_plate(mu, sigma, fps, candidates, cost, budget, size=96, kappa=1.0, max_sim=0.7):
    """Upper-confidence selection with diversity and a cost cap: take the highest mean + kappa * sigma that is not within
    Tanimoto `max_sim` of anything already on the plate and still fits the budget. -> list of (index, rationale)"""
    score = mu + kappa * sigma
    order = [i for i in np.argsort(-score) if i in candidates]
    chosen, spent, bvs = [], 0.0, []
    for i in order:
        if len(chosen) >= size:
            break
        if spent + cost[i] > budget:
            continue
        bv = DataStructs.CreateFromBitString("".join(map(str, fps[i])))
        if bvs and max(DataStructs.BulkTanimotoSimilarity(bv, bvs)) > max_sim:
            continue
        chosen.append((int(i), {"predicted": round(float(mu[i]), 2), "uncertainty": round(float(sigma[i]), 2), "score": round(float(score[i]), 2), "cost": float(cost[i]), "why": "high predicted affinity" if mu[i] > np.percentile(mu, 95) else "uncertain and unexplored" if sigma[i] > np.percentile(sigma, 80) else "good score, adds diversity"}))
        spent += cost[i]
        bvs.append(bv)
    return chosen, spent


def top_by_mean(mu, candidates, cost, budget, size=96):
    chosen, spent = [], 0.0
    for i in np.argsort(-mu):
        if len(chosen) >= size:
            break
        if i in candidates and spent + cost[i] <= budget:
            chosen.append(int(i))
            spent += cost[i]
    return chosen


def bit_contributions(model, x_row, fps_row, top=5):
    """Which fingerprint bits move this compound's prediction most: switch each on-bit off and see the change (one batch)."""
    on = np.flatnonzero(fps_row)
    alts = np.repeat(x_row[None], len(on) + 1, axis=0)
    alts[np.arange(1, len(on) + 1), on] = 0
    p = np.mean([m.predict(alts) for m in model["members"]], axis=0)
    out = sorted(((int(b), round(float(p[0] - p[k + 1]), 3)) for k, b in enumerate(on)), key=lambda t: -abs(t[1]))
    return out[:top]


def bit_fragments(mol, bits):
    """The substructure each bit stands for in this molecule, as SMILES."""
    from rdkit import Chem
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    MORGAN.GetFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()
    out = {}
    for b in bits:
        if b in info:
            atom, radius = info[b][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom) if radius else []
            atoms = {atom} | {a for bond in env for a in (mol.GetBondWithIdx(bond).GetBeginAtomIdx(), mol.GetBondWithIdx(bond).GetEndAtomIdx())}
            out[b] = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=list(env) or None)
    return out
