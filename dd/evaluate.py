"""Measure the affinity model, its calibration and the plate-selection strategies on libraries the demo never uses. Writes docs/evaluation.md.

    python -m dd.evaluate
"""
import pathlib

import numpy as np

from . import engine, world

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
SEEDS = (101, 102, 103)


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    model_rows, plate_rows = [], []
    for seed in SEEDS:
        w = world.build(seed)
        X = engine.features(w["fps"], w["desc"])
        N = len(w["mols"])
        hist = world.history_indices(w)
        meas = world.assay(w, hist, 1)
        agg = engine.aggregate(meas)
        idx = np.array(sorted(agg))
        y = np.array([agg[i] for i in idx])
        rest = np.setdiff1d(np.arange(N), idx)
        model = engine.train(X[idx], y)
        mu, sigma = engine.predict(model, X)
        cal = engine.calibration(mu[rest], sigma[rest], w["truth"][rest])
        knn, _ = engine.knn_predict(w["fps"][idx], y, w["fps"][rest[:3000]])
        model_rows.append([seed, len(idx), cal["rmse"], round(float(np.sqrt(np.mean((knn - w["truth"][rest[:3000]]) ** 2))), 3), pct(cal["inside_90_band"]), cal["spearman_sigma_vs_error"], round(model["scale"], 2)])
        r = np.random.default_rng(seed)
        cost = np.where(r.random(N) < 0.6, 1.0, 5.0)
        cands = set(rest.tolist())
        picks = {"upper confidence + diversity": [i for i, _ in engine.select_plate(mu, sigma, w["fps"], cands, cost, 200)[0]], "exploit (top predicted)": engine.top_by_mean(mu, cands, cost, 200), "random": r.choice(rest, 96, replace=False).tolist()}
        for name, pick in picks.items():
            pick = np.array(pick)
            agg2 = engine.aggregate(meas + world.assay(w, pick, 2))
            idx2 = np.array(sorted(agg2))
            rest2 = np.setdiff1d(np.arange(N), idx2)
            mu2, _ = engine.predict(engine.train(X[idx2], np.array([agg2[i] for i in idx2])), X)
            plate_rows.append([seed, name, len(pick), int((w["truth"][pick] >= world.ACTIVE_LIMIT).sum()), len({w["mols"][i]["parts"][0] for i in pick}), round(float(np.sqrt(np.mean((mu[rest2] - w["truth"][rest2]) ** 2))), 3), round(float(np.sqrt(np.mean((mu2[rest2] - w["truth"][rest2]) ** 2))), 3)])
    by = lambda name, col: np.mean([r[col] for r in plate_rows if r[1] == name])      # noqa: E731
    names = ["upper confidence + diversity", "exploit (top predicted)", "random"]
    doc = f"""# Evaluation

Everything here is measured on synthetic libraries generated with seeds the demo does not use (`{SEEDS}`): the same cores, linkers and
substituents, a different hidden binding rule's noise and a different measured history. Reproduce with `python -m dd.evaluate`
(about three minutes). The numbers say how well the methods recover what the generator hid. They are not a claim about a real
target: the hidden rule is a sum of substructure effects with one interaction, which fingerprints were made to find, and a real
structure-activity landscape is far rougher.

## 1. The affinity model

Five gradient-boosted regressors on fingerprints and descriptors, trained on the 300 measured compounds, scored on every unmeasured
compound against the hidden truth (assay noise 0.3; the hidden rule's own noise 0.45, so about 0.45 is the floor).

{table(["library (seed)", "measured", "RMSE", "nearest-neighbour RMSE", "inside the 90% band", "uncertainty tracks error (rank corr.)", "uncertainty scale applied"], model_rows)}

The ensemble's raw spread was two to three times too narrow; one scale factor fitted on a held-out fifth brings the band to about
{pct(np.mean([float(r[4].rstrip('%')) / 100 for r in model_rows]))}, still short of 90%. The uncertainty says little about which compounds will be wrong (rank correlation near
zero): it is wide where the training data is thin, not where the rule is hard.

## 2. Which 96 compounds for the next plate?

Three ways to fill a plate under a budget of 200 cost units (1 in stock, 5 to make), each scored on the hidden truth: actives found
(pIC50 at or above 7), distinct scaffolds, and the model's error on the rest of the library after the plate is folded in.

{table(["library", "selection", "compounds", "actives found", "scaffolds", "RMSE before", "RMSE after"], plate_rows)}

Means: actives found, {" / ".join(f"{n}: {by(n, 3):.0f}" for n in names)}; RMSE after the plate, {" / ".join(f"{n}: {by(n, 6):.3f}" for n in names)}.

Exploitation finds the most actives and the model learns nothing from them: it fills the plate from series it already knows.
Random finds almost no actives and is the only selection that improves the model. Upper confidence with a diversity rule found
fewer actives than exploitation and taught the model no more, which is a worse result than the method's reputation: with a
rule this additive, the uncertain compounds it reaches for are expensive and uninformative. The honest conclusion is that on
this generator the choice is between actives now (exploit) and a better model (random), and the API exposes the knob (kappa)
rather than pretending to have resolved it.
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
