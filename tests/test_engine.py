"""Unit tests for the library generator, the affinity model, calibration and plate selection: no database."""
import numpy as np
from rdkit import Chem

from dd import engine, world

W = world.build()
X = engine.features(W["fps"], W["desc"])


def test_library_is_real_molecules_and_deterministic():
    assert len(W["mols"]) == 10000 and len({m["smiles"] for m in W["mols"]}) == 10000
    assert all(Chem.MolFromSmiles(m["smiles"]) is not None for m in W["mols"][:500]) and np.array_equal(world.build.__wrapped__(17)["truth"], W["truth"])
    assert W["fps"].shape == (10000, 1024) and W["desc"].shape == (10000, 7) and 500 < (W["truth"] >= world.ACTIVE_LIMIT).sum() < 2000
    d = world.descriptors(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert abs(d["mol_weight"] - 180.16) < 0.1 and world.lipinski(d)["passes"]


def test_hidden_rule_rewards_what_it_says():
    has = W["has"]
    both = has["sulfonamide"] & has["aryl_amine"] & ~has["piperidine"]
    assert W["truth"][both].mean() > W["truth"][~has["sulfonamide"] & ~has["aryl_amine"]].mean() + 2
    cliff = has["sulfonamide"] & has["piperidine"]
    assert W["truth"][cliff].mean() < W["truth"][has["sulfonamide"] & ~has["piperidine"]].mean() - 1


def test_assay_noise_and_censoring():
    m = world.assay(W, [0, 1, 2], 1)
    assert len(m) == 6 and all(v["value"] >= 4.5 for v in m) and all((v["qualifier"] == "<") == (v["value"] == 4.5) for v in m)
    agg = engine.aggregate(m)
    assert set(agg) == {0, 1, 2} and all(abs(agg[i] - max(4.5, W["truth"][i])) < 1.5 for i in agg)


def test_model_beats_knn_and_band_is_roughly_honest():
    hist = world.history_indices(W)
    agg = engine.aggregate(world.assay(W, hist, 1))
    idx = np.array(sorted(agg))
    y = np.array([agg[i] for i in idx])
    model = engine.train(X[idx], y)
    rest = np.setdiff1d(np.arange(10000), idx)[:3000]
    mu, sigma = engine.predict(model, X[rest])
    cal = engine.calibration(mu, sigma, W["truth"][rest])
    knn, nn = engine.knn_predict(W["fps"][idx], y, W["fps"][rest[:500]])
    assert cal["rmse"] < np.sqrt(np.mean((knn - W["truth"][rest[:500]]) ** 2)) and 0.65 < cal["inside_90_band"] < 0.97 and (sigma > 0).all()
    assert all(0 <= s <= 1 for row in nn for _, s in row) and model["scale"] > 1          # the raw ensemble spread is too narrow and is rescaled


def test_plate_selection_respects_budget_and_diversity():
    hist = world.history_indices(W)
    agg = engine.aggregate(world.assay(W, hist, 1))
    idx = np.array(sorted(agg))
    model = engine.train(X[idx], np.array([agg[i] for i in idx]))
    mu, sigma = engine.predict(model, X)
    cost = np.where(np.arange(10000) % 3 == 0, 5.0, 1.0)
    cands = set(range(10000)) - set(agg)
    chosen, spent = engine.select_plate(mu, sigma, W["fps"], cands, cost, 120, 96, 1.0, 0.7)
    ids = [i for i, _ in chosen]
    assert spent <= 120 and len(ids) <= 96 and not set(ids) & set(agg) and len(set(ids)) == len(ids)
    from rdkit import DataStructs
    bvs = [DataStructs.CreateFromBitString("".join(map(str, W["fps"][i]))) for i in ids]
    assert all(DataStructs.TanimotoSimilarity(bvs[a], bvs[b]) <= 0.7 for a in range(len(bvs)) for b in range(a))
    exploit = engine.top_by_mean(mu, cands, cost, 120, 96)
    assert cost[exploit].sum() <= 120 and mu[exploit].mean() >= mu[ids].mean()
    bits = engine.bit_contributions(model, X[ids[0]], W["fps"][ids[0]])
    frags = engine.bit_fragments(W["mols"][ids[0]]["mol"], [b for b, _ in bits])
    assert len(bits) <= 5 and all(isinstance(v, str) for v in frags.values())
