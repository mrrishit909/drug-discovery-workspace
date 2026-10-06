"""Public API (modular monolith). Blueprint services map to: compound-registry, assay-ingestion, molecular-featurizer and
property-prediction-service (RDKit descriptors, an affinity ensemble with calibrated uncertainty), active-learning-service and
experiment-planner, model-registry (core's table), and the evidence/trace views. Not built: a docking adapter, literature RAG,
a target knowledge graph beyond one target.

    uvicorn dd.api:app          python -m core.jobs dd.api      # the worker
"""
import datetime
import functools
import pickle
import uuid

import numpy as np
from fastapi import Depends, Header, Query
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field
from rdkit import Chem

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, world

READ = {"library:read", "jobs:read"}
WORK = READ | {"compounds:write", "measurements:write", "predict:run", "experiments:propose"}
PERMISSIONS = {"viewer": READ, "scientist": WORK, "lead": WORK | {"library:load", "models:train", "experiments:approve", "audit:read"}}
app = create_app("drug-discovery-workspace", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
TARGET = {"gene_symbol": "RBK1", "uniprot_id": "P0RBK1", "name": "Riverbend kinase 1 (invented)"}
HISTORY_SEED, PLATE_SEED = 1, 2
MODEL_NAME = "affinity"


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def cid(t, k):
    return uuid.uuid5(t, f"compound-{k}")


def library(c):
    lib = c.execute("SELECT * FROM libraries").fetchone()
    if not lib:
        raise Problem(409, "no_library", "load a library first: POST /v1/library:load")
    return lib, world.build(lib["model_seed"])


def target(c):
    return c.execute("SELECT * FROM targets LIMIT 1").fetchone()


def measured(c, assay_id):
    """Aggregated measured values by compound ordinal, from storage."""
    rows = c.execute("SELECT cp.ordinal AS index, m.value FROM measurements m JOIN compounds cp ON cp.id = m.compound_id WHERE m.assay_id = %s AND cp.ordinal IS NOT NULL", [assay_id]).fetchall()
    return engine.aggregate(rows)


def current_model(c, tenant_id):
    row = c.execute("SELECT version, metrics FROM model_registry WHERE name = %s AND state = 'production' ORDER BY created_at DESC LIMIT 1", [f"{MODEL_NAME}-{tenant_id}"]).fetchone()
    if not row:
        raise Problem(409, "no_model", "train the affinity model first: POST /v1/models/train")
    return row["version"], _artifact(f"{MODEL_NAME}-{tenant_id}", row["version"]), row["metrics"]


@functools.lru_cache(maxsize=8)
def _artifact(name, version):
    """A model version never changes once written, so it is unpickled once per process."""
    with db.tx() as c:
        return pickle.loads(c.execute("SELECT artifact FROM model_registry WHERE name = %s AND version = %s", [name, version]).fetchone()["artifact"])


@functools.lru_cache(maxsize=8)
def measured_bitvectors(seed, idx):
    return engine.bitvectors(world.build(seed)["fps"][list(idx)])


@functools.lru_cache(maxsize=8)
def cliffs_for(seed, idx, vals):
    """Activity cliffs among the measured compounds; the measured set changes only when a plate lands."""
    from rdkit import DataStructs
    w, out = world.build(seed), []
    bvs = measured_bitvectors(seed, idx)
    for a in range(len(idx)):
        sims = DataStructs.BulkTanimotoSimilarity(bvs[a], bvs[a + 1:])
        for off, s_ in enumerate(sims):
            b = a + 1 + off
            if s_ >= 0.7 and abs(vals[a] - vals[b]) >= 1.5:
                out.append((idx[a], vals[a], idx[b], vals[b], round(float(s_), 2)))
    out.sort(key=lambda x: -abs(x[1] - x[3]))
    return out[:8]


def canonical(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return m, Chem.MolToSmiles(m), Chem.MolToInchiKey(m)


# ---------------------------------------------------------------- compound-registry

class LoadIn(BaseModel):
    seed: int = Field(17, ge=0, le=10 ** 6)


@app.post("/v1/library:load", status_code=202, tags=["library"], summary="(+) Register a synthetic library of about 10,000 real molecules, one target, one assay and 300 measured compounds")
def load_library(body: LoadIn, ctx: Ctx = Depends(auth("library:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM libraries LIMIT 1").fetchone():
            raise Problem(409, "library_exists")
        return 202, jobs.enqueue(c, ctx, "library.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("library.load")
def load_job(c, job):
    t, w = job["tenant_id"], world.build(job["payload"]["seed"])
    lid = uuid.uuid5(t, "library")
    c.execute("INSERT INTO libraries (id, tenant_id, name, model_seed) VALUES (%s,%s,'Riverbend screening library',%s)", [lid, t, w["seed"]])
    tid = uuid.uuid5(t, "target")
    c.execute("INSERT INTO targets (id, tenant_id, gene_symbol, uniprot_id, name, metadata) VALUES (%s,%s,%s,%s,%s,%s)", [tid, t, TARGET["gene_symbol"], TARGET["uniprot_id"], TARGET["name"], Jsonb({"family": "kinase", "note": "invented target with a hidden binding rule"})])
    aid = uuid.uuid5(t, "assay")
    c.execute("INSERT INTO assays (id, tenant_id, target_id, assay_type, protocol, unit) VALUES (%s,%s,%s,'biochemical IC50',%s,'pIC50')", [aid, t, tid, Jsonb({"format": "384-well", "readout": "fluorescence", "replicates": 2, "lower_limit_pIC50": 4.5})])
    r = np.random.default_rng([w["seed"], 3])
    cost = np.where(r.random(len(w["mols"])) < 0.6, 1.0, 5.0)                             # in stock, or to be made
    rows = [(cid(t, m["index"]), t, m["smiles"], Chem.MolToInchiKey(m["mol"]), round(float(w["desc"][m["index"], 0]), 2), "library", m["index"], Jsonb(world.descriptors(m["mol"])), float(cost[m["index"]])) for m in w["mols"]]
    db.load(c, "compounds", ["id", "tenant_id", "canonical_smiles", "inchikey", "mol_weight", "source", "ordinal", "descriptors", "cost"], rows)
    hist = world.history_indices(w)
    meas = world.assay(w, hist, HISTORY_SEED)
    db.load(c, "measurements", ["id", "tenant_id", "assay_id", "compound_id", "value", "qualifier", "replicate", "measured_at", "plate"],
            [(uuid.uuid7(), t, aid, cid(t, m["index"]), m["value"], m["qualifier"], m["replicate"], datetime.datetime(2026, 9, 1, tzinfo=datetime.UTC) + datetime.timedelta(days=int(k // 96)), f"HIST-{k // 96 + 1:02d}") for k, m in enumerate(meas)])
    out = {"library_id": lid, "target_id": tid, "assay_id": aid, "compounds": len(rows), "in_stock": int((cost == 1).sum()), "to_synthesise": int((cost == 5).sum()), "measured_compounds": len(hist), "measurements": len(meas), "censored_below_4_5": sum(m["qualifier"] == "<" for m in meas),
           "actives_among_measured": int((np.array([engine.aggregate(meas)[i] for i in hist]) >= world.ACTIVE_LIMIT).sum()), "descriptor_summary": {"mol_weight_mean": round(float(w["desc"][:, 0].mean()), 1), "logp_mean": round(float(w["desc"][:, 1].mean()), 2)},
           "simulation_truth": {"what": "the hidden binding rule the generator used; the models never see it", "rule": w["rule"], "actives_in_library": int((w["truth"] >= world.ACTIVE_LIMIT).sum())}}
    audit.record(c, worker_ctx(job), "library.loaded", "library", lid, {"compounds": len(rows), "measurements": len(meas)})
    return out


class CompoundsIn(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=1000)
    source: str = Field("upload", max_length=40)


@app.post("/v1/compounds", status_code=201, tags=["compounds"], summary="Register compounds by SMILES: canonicalised, deduplicated by InChIKey, descriptors computed")
def compounds(body: CompoundsIn, ctx: Ctx = Depends(auth("compounds:write")), idem: str | None = IdemKey):
    def work(c):
        library(c)
        added, dupes, invalid = [], [], []
        for s in body.smiles:
            parsed = canonical(s)
            if not parsed:
                invalid.append(s)
                continue
            mol, smi, key = parsed
            if c.execute("SELECT 1 FROM compounds WHERE inchikey = %s", [key]).fetchone():
                dupes.append(smi)
                continue
            d = world.descriptors(mol)
            k = uuid.uuid7()
            c.execute("INSERT INTO compounds (id, tenant_id, canonical_smiles, inchikey, mol_weight, source, descriptors, cost, registered_by) VALUES (%s,%s,%s,%s,%s,%s,%s,5,%s)", [k, ctx.tenant_id, smi, key, d["mol_weight"], body.source, Jsonb(d), ctx.actor_id])
            added.append({"compound_id": k, "canonical_smiles": smi, "inchikey": key, **d, "lipinski": world.lipinski(d)})
        if invalid and not added and not dupes:
            raise Problem(422, "invalid_smiles", invalid[0])
        if added:
            audit.record(c, ctx, "compounds.registered", "library", ctx.tenant_id, {"added": len(added), "source": body.source})
        return 201, {"added": added, "duplicates_ignored": dupes, "invalid": invalid}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- assay-ingestion

class MeasurementIn(BaseModel):
    compound_id: uuid.UUID
    value: float = Field(ge=3, le=12)
    qualifier: str = Field("=", pattern="^[=<>]$")
    replicate: int = Field(ge=1, le=10)
    measured_at: datetime.datetime


class MeasurementsIn(BaseModel):
    plate: str = Field(max_length=30)
    measurements: list[MeasurementIn] = Field(min_length=1, max_length=2000)


@app.post("/v1/assays/{assay_id}/measurements", status_code=201, tags=["assays"], summary="Upload assay results for a plate; a result already stored for that compound, replicate and plate is ignored. Marks the proposed experiment measured")
def measurements(assay_id: uuid.UUID, body: MeasurementsIn, ctx: Ctx = Depends(auth("measurements:write")), idem: str | None = IdemKey):
    def work(c):
        library(c)
        if not c.execute("SELECT 1 FROM assays WHERE id = %s", [assay_id]).fetchone():
            raise Problem(404, "assay_not_found")
        if any(m.measured_at.tzinfo is None for m in body.measurements):
            raise Problem(422, "timezone_required")
        ids = {m.compound_id for m in body.measurements}
        known = {r["id"] for r in c.execute("SELECT id FROM compounds WHERE id = ANY(%s)", [list(ids)])}
        if ids - known:
            raise Problem(404, "compound_not_found", str(sorted(ids - known)[0]))
        stored = sum(bool(c.execute("INSERT INTO measurements (id, tenant_id, assay_id, compound_id, value, qualifier, replicate, measured_at, plate, recorded_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id",
                                    [uuid.uuid7(), ctx.tenant_id, assay_id, m.compound_id, m.value, m.qualifier, m.replicate, m.measured_at, body.plate, ctx.actor_id]).fetchone()) for m in body.measurements)
        done = c.execute("UPDATE experiments e SET status = 'measured' WHERE status = 'approved' AND NOT EXISTS (SELECT 1 FROM experiment_candidates ec WHERE ec.experiment_id = e.id AND ec.compound_id NOT IN (SELECT compound_id FROM measurements WHERE assay_id = %s)) RETURNING id", [assay_id]).fetchall()
        agg = engine.aggregate([{"index": str(m.compound_id), "value": m.value} for m in body.measurements])
        audit.record(c, ctx, "measurements.uploaded", "assay", assay_id, {"plate": body.plate, "stored": stored})
        return 201, {"assay_id": assay_id, "plate": body.plate, "received": len(body.measurements), "stored": stored, "duplicates_ignored": len(body.measurements) - stored, "compounds": len(ids), "actives_on_plate": int(sum(v >= world.ACTIVE_LIMIT for v in agg.values())),
                     "experiments_completed": [d["id"] for d in done], "next": "POST /v1/models/train to fold the plate into the model"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- property-prediction-service and model-registry

class SmilesIn(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=200)


@app.post("/v1/predict/properties", tags=["predict"], summary="Physicochemical properties by RDKit: computed from the structure, not predicted")
def properties(body: SmilesIn, ctx: Ctx = Depends(auth("predict:run"))):
    out = []
    for s in body.smiles:
        parsed = canonical(s)
        if not parsed:
            out.append({"smiles": s, "error": "not a parseable SMILES"})
            continue
        mol, smi, key = parsed
        d = world.descriptors(mol)
        out.append({"smiles": s, "canonical_smiles": smi, "inchikey": key, **d, "lipinski": world.lipinski(d)})
    return {"items": out, "how": "RDKit descriptors (Wildman-Crippen logP, TPSA, Lipinski counts); nothing here is a model"}


@app.post("/v1/models/train", status_code=202, tags=["models"], summary="(+) Train a new affinity model version on every measurement so far, report calibration on a held-out split and against the k-nearest-neighbour baseline, and write predictions for the whole library")
def train(ctx: Ctx = Depends(auth("models:train")), idem: str | None = IdemKey):
    def work(c):
        library(c)
        return 202, jobs.enqueue(c, ctx, "model.train", {})
    return run(ctx, idem, {}, work)


@jobs.handler("model.train")
def train_job(c, job):
    t, (lib, w) = job["tenant_id"], library(c)
    tg, aid = target(c), c.execute("SELECT id FROM assays LIMIT 1").fetchone()["id"]
    agg = measured(c, aid)
    idx = np.array(sorted(agg))
    y = np.array([agg[i] for i in idx])
    X = engine.features(w["fps"], w["desc"])
    # honest test: the last 20% of measured compounds (by ordinal order of acquisition is unknown, so a fixed random fifth) held out for the report, then refit on all
    r = np.random.default_rng(len(idx))
    test = r.random(len(idx)) < 0.2
    probe = engine.train(X[idx[~test]], y[~test])
    mu_t, sg_t = engine.predict(probe, X[idx[test]])
    knn, _ = engine.knn_predict(w["fps"][idx[~test]], y[~test], w["fps"][idx[test]])
    cal = engine.calibration(mu_t, sg_t, y[test])
    cal["knn_rmse"] = round(float(np.sqrt(np.mean((knn - y[test]) ** 2))), 3)
    cal["trained_on"], cal["tested_on"] = int((~test).sum()), int(test.sum())
    model = engine.train(X[idx], y)
    n = c.execute("SELECT count(*) AS n FROM model_registry WHERE name = %s", [f"{MODEL_NAME}-{t}"]).fetchone()["n"]
    version = f"{engine.VERSION_PREFIX}-v{n + 1}"
    prev = c.execute("SELECT version, metrics FROM model_registry WHERE name = %s AND state = 'production' ORDER BY created_at DESC LIMIT 1", [f"{MODEL_NAME}-{t}"]).fetchone()
    c.execute("UPDATE model_registry SET state = 'retired' WHERE name = %s", [f"{MODEL_NAME}-{t}"])
    c.execute("INSERT INTO model_registry (name, version, state, data_snapshot, metrics, baseline, artifact) VALUES (%s,%s,'production',%s,%s,%s,%s)",
              [f"{MODEL_NAME}-{t}", version, f"{len(idx)} compounds, {len(y)} aggregated values, assay {aid}", Jsonb(cal), Jsonb({"knn_rmse": cal["knn_rmse"], "k": 3}), pickle.dumps(model)])
    mu, sigma = engine.predict(model, X)
    c.execute("DELETE FROM predictions WHERE model_version = %s", [version])
    db.load(c, "predictions", ["id", "tenant_id", "compound_id", "target_id", "endpoint", "value", "uncertainty", "model_version"], [(uuid.uuid7(), t, cid(t, i), tg["id"], "pIC50", round(float(mu[i]), 3), round(float(sigma[i]), 3), version) for i in range(len(w["mols"]))])
    rest = np.setdiff1d(np.arange(len(w["mols"])), idx)
    truth_check = {"what": "the model's error on every unmeasured compound against the hidden truth", "rmse": round(float(np.sqrt(np.mean((mu[rest] - w["truth"][rest]) ** 2))), 3), "inside_90_band": round(float((np.abs(mu[rest] - w["truth"][rest]) <= engine.Z90 * sigma[rest]).mean()), 3),
                   "actives_in_top_96_by_prediction": int((w["truth"][rest[np.argsort(-mu[rest])[:96]]] >= world.ACTIVE_LIMIT).sum())}
    out = {"version": version, "measured_compounds": len(idx), "held_out_report": cal, "uncertainty_scale": round(model["scale"], 2), "previous_version": prev and {"version": prev["version"], "rmse": prev["metrics"]["rmse"], "inside_90_band": prev["metrics"]["inside_90_band"]},
           "predictions_written": len(w["mols"]), "predicted_actives": int((mu >= world.ACTIVE_LIMIT).sum()), "simulation_truth": truth_check,
           "method": f"{engine.MEMBERS} gradient-boosted regressors on bootstrap resamples of Morgan fingerprints (1024 bits) and seven descriptors; the spread between them, rescaled on a held-out fifth so the 90% band is honest there, is the uncertainty"}
    audit.record(c, worker_ctx(job), "model.trained", "model", t, {"version": version, "rmse": cal["rmse"], "knn_rmse": cal["knn_rmse"], "measured": len(idx)})
    return out


class AffinityIn(BaseModel):
    compound_ids: list[uuid.UUID] = Field([], max_length=200)
    smiles: list[str] = Field([], max_length=50)


@app.post("/v1/predict/target-affinity", tags=["predict"], summary="Predicted pIC50 with a 90% band for registered compounds or new SMILES, with the model version and the nearest measured neighbours")
def affinity(body: AffinityIn, ctx: Ctx = Depends(auth("predict:run"))):
    with db.tx(ctx.tenant_id) as c:
        lib, w = library(c)
        version, model, _ = current_model(c, ctx.tenant_id)
        aid = c.execute("SELECT id FROM assays LIMIT 1").fetchone()["id"]
        agg = measured(c, aid)
        rows = c.execute("SELECT id, ordinal, canonical_smiles FROM compounds WHERE id = ANY(%s)", [body.compound_ids]).fetchall()
    if len(rows) < len(set(body.compound_ids)):
        raise Problem(404, "compound_not_found")
    mols = [(str(r["id"]), r["canonical_smiles"], r["ordinal"], Chem.MolFromSmiles(r["canonical_smiles"])) for r in rows]
    for s in body.smiles:
        parsed = canonical(s)
        if not parsed:
            raise Problem(422, "invalid_smiles", s)
        mols.append((None, parsed[1], None, parsed[0]))
    if not mols:
        raise Problem(422, "nothing_to_predict")
    fps = np.array([list(world.MORGAN.GetFingerprintAsNumPy(m)) for *_, m in mols], dtype=np.uint8)
    desc = np.array([[world.descriptors(m)[k] for k in ("mol_weight", "logp", "tpsa", "h_donors", "h_acceptors", "rotatable_bonds", "aromatic_rings")] for *_, m in mols])
    mu, sigma = engine.predict(model, engine.features(fps, desc))
    midx = np.array(sorted(agg))
    _, nn = engine.knn_predict(w["fps"][midx], np.array([agg[i] for i in midx]), fps, bv_train=measured_bitvectors(w["seed"], tuple(int(i) for i in midx)))
    items = [{"compound_id": k, "smiles": smi, "predicted_pIC50": round(float(mu[j]), 2), "band_90": [round(float(mu[j] - engine.Z90 * sigma[j]), 2), round(float(mu[j] + engine.Z90 * sigma[j]), 2)], "uncertainty": round(float(sigma[j]), 2), "predicted_active": bool(mu[j] >= world.ACTIVE_LIMIT),
              "measured": round(agg[o], 2) if o in agg else None, "nearest_measured": [{"compound_id": cid(ctx.tenant_id, int(midx[a])), "smiles": w["mols"][int(midx[a])]["smiles"], "tanimoto": s_, "measured_pIC50": round(agg[int(midx[a])], 2)} for a, s_ in nn[j]]} for j, (k, smi, o, _) in enumerate(mols)]
    return jsonable_encoder({"model_version": version, "endpoint": "pIC50", "items": items})


# ---------------------------------------------------------------- active-learning-service and experiment-planner

class BatchIn(BaseModel):
    plate_size: int = Field(96, ge=8, le=384)
    budget: float = Field(200, ge=8, le=5000, description="Cost units: 1 for a compound in stock, 5 for one to be made")
    kappa: float = Field(1.0, ge=0, le=3, description="Weight on uncertainty: 0 exploits, 2 explores")
    max_similarity: float = Field(0.7, ge=0.3, le=1.0)


def plan_batch(c, ctx, body):
    lib, w = library(c)
    version, model, _ = current_model(c, ctx.tenant_id)
    aid = c.execute("SELECT id FROM assays LIMIT 1").fetchone()["id"]
    agg = measured(c, aid)
    cost = np.array([r["cost"] for r in c.execute("SELECT cost FROM compounds WHERE ordinal IS NOT NULL ORDER BY ordinal")])
    X = engine.features(w["fps"], w["desc"])
    mu, sigma = engine.predict(model, X)
    cands = set(range(len(w["mols"]))) - set(agg)
    chosen, spent = engine.select_plate(mu, sigma, w["fps"], cands, cost, body.budget, body.plate_size, body.kappa, body.max_similarity)
    exploit = engine.top_by_mean(mu, cands, cost, body.budget, body.plate_size)
    r = np.random.default_rng(len(agg))
    rnd = r.choice(sorted(cands), body.plate_size, replace=False)
    truth = {"what": "actives (pIC50 >= 7) each selection would really find, from the hidden truth", "proposed": int((w["truth"][[i for i, _ in chosen]] >= world.ACTIVE_LIMIT).sum()), "top_by_prediction": int((w["truth"][exploit] >= world.ACTIVE_LIMIT).sum()), "random": int((w["truth"][rnd] >= world.ACTIVE_LIMIT).sum())}
    cores = lambda ids: len({w["mols"][i]["parts"][0] for i in ids})      # noqa: E731
    return {"model_version": version, "plate_size": body.plate_size, "budget": body.budget, "spent": spent, "selected": len(chosen), "candidates": len(cands), "kappa": body.kappa, "max_similarity": body.max_similarity,
            "items": [{"rank": k + 1, "compound_id": cid(ctx.tenant_id, i), "smiles": w["mols"][i]["smiles"], **why} for k, (i, why) in enumerate(chosen)],
            "comparison": {"proposed": {"n": len(chosen), "scaffolds": cores([i for i, _ in chosen]), "mean_predicted": round(float(mu[[i for i, _ in chosen]].mean()), 2), "cost": spent}, "top_by_prediction": {"n": len(exploit), "scaffolds": cores(exploit), "mean_predicted": round(float(mu[exploit].mean()), 2), "cost": float(cost[exploit].sum())}, "random": {"n": len(rnd), "scaffolds": cores(rnd.tolist()), "mean_predicted": round(float(mu[rnd].mean()), 2), "cost": float(cost[rnd].sum())}},
            "simulation_truth": truth, "how": "mean + kappa x uncertainty, highest first; skip anything within Tanimoto max_similarity of a compound already on the plate; stop at the plate size or the budget"}


@app.post("/v1/active-learning/batch", tags=["active learning"], summary="Select the next plate: upper-confidence scores with diversity and a cost cap; compared with pure exploitation and random")
def batch(body: BatchIn, ctx: Ctx = Depends(auth("experiments:propose"))):
    with db.tx(ctx.tenant_id) as c:
        return jsonable_encoder(plan_batch(c, ctx, body))


class ProposeIn(BatchIn):
    hypothesis: str = Field(min_length=5, max_length=300)


@app.post("/v1/experiments/propose", status_code=201, tags=["experiments"], summary="Propose the next assay plate as an experiment: the selection, its rationale per compound, the budget; a lead approves before plates are made")
def propose(body: ProposeIn, ctx: Ctx = Depends(auth("experiments:propose")), idem: str | None = IdemKey):
    def work(c):
        plan = plan_batch(c, ctx, body)
        eid = uuid.uuid7()
        design = {k: plan[k] for k in ("plate_size", "budget", "spent", "selected", "kappa", "max_similarity", "comparison", "how")}
        c.execute("INSERT INTO experiments (id, tenant_id, hypothesis, budget, design, owner_id, model_version) VALUES (%s,%s,%s,%s,%s,%s,%s)", [eid, ctx.tenant_id, body.hypothesis, body.budget, Jsonb(design), ctx.actor_id, plan["model_version"]])
        db.load(c, "experiment_candidates", ["experiment_id", "compound_id", "rank", "rationale", "tenant_id"], [(eid, it["compound_id"], it["rank"], Jsonb({k: it[k] for k in ("predicted", "uncertainty", "score", "cost", "why")}), ctx.tenant_id) for it in plan["items"]])
        audit.record(c, ctx, "experiment.proposed", "experiment", eid, {"compounds": plan["selected"], "spent": plan["spent"], "model_version": plan["model_version"]})
        return 201, {"experiment_id": eid, "status": "proposed", "hypothesis": body.hypothesis, **plan}
    return run(ctx, idem, body, work)


@app.post("/v1/experiments/{experiment_id}/approve", tags=["experiments"], summary="(+) A lead approves a proposed experiment")
def approve(experiment_id: uuid.UUID, ctx: Ctx = Depends(auth("experiments:approve")), idem: str | None = IdemKey):
    def work(c):
        e = c.execute("SELECT * FROM experiments WHERE id = %s FOR UPDATE", [experiment_id]).fetchone()
        if not e:
            raise Problem(404, "experiment_not_found")
        if e["status"] != "proposed":
            raise Problem(409, "not_proposed", f"already {e['status']}")
        if e["owner_id"] == ctx.actor_id:
            raise Problem(409, "same_person", "the person who proposed an experiment cannot approve it")
        c.execute("UPDATE experiments SET status = 'approved', approved_by = %s WHERE id = %s", [ctx.actor_id, experiment_id])
        audit.record(c, ctx, "experiment.approved", "experiment", experiment_id, {"owner": str(e["owner_id"])})
        return 200, {"experiment_id": experiment_id, "status": "approved", "compounds": c.execute("SELECT count(*) AS n FROM experiment_candidates WHERE experiment_id = %s", [experiment_id]).fetchone()["n"]}
    return run(ctx, idem, {}, work)


@app.post("/v1/experiments/{experiment_id}/results:simulate", status_code=201, tags=["experiments"], summary="(+ demo only) The lab returns the plate: the generator's hidden values plus assay noise, stored through the same path as an upload")
def simulate_results(experiment_id: uuid.UUID, ctx: Ctx = Depends(auth("measurements:write")), idem: str | None = IdemKey):
    def work(c):
        lib, w = library(c)
        e = c.execute("SELECT * FROM experiments WHERE id = %s", [experiment_id]).fetchone()
        if not e:
            raise Problem(404, "experiment_not_found")
        if e["status"] != "approved":
            raise Problem(409, "not_approved", f"the experiment is {e['status']}; a lead must approve it before plates are made")
        aid = c.execute("SELECT id FROM assays LIMIT 1").fetchone()["id"]
        ords = [r["ordinal"] for r in c.execute("SELECT cp.ordinal FROM experiment_candidates ec JOIN compounds cp ON cp.id = ec.compound_id WHERE ec.experiment_id = %s ORDER BY ec.rank", [experiment_id])]
        results = world.assay(w, ords, PLATE_SEED)
        plate = f"AL-{str(experiment_id)[-4:]}"
        when = datetime.datetime.now(datetime.UTC)
        db.load(c, "measurements", ["id", "tenant_id", "assay_id", "compound_id", "value", "qualifier", "replicate", "measured_at", "plate", "recorded_by"], [(uuid.uuid7(), ctx.tenant_id, aid, cid(ctx.tenant_id, m["index"]), m["value"], m["qualifier"], m["replicate"], when, plate, ctx.actor_id) for m in results])
        c.execute("UPDATE experiments SET status = 'measured' WHERE id = %s", [experiment_id])
        agg = engine.aggregate(results)
        pred = {r["ordinal"]: float(r["value"]) for r in c.execute("SELECT cp.ordinal, p.value FROM predictions p JOIN compounds cp ON cp.id = p.compound_id WHERE p.model_version = %s AND cp.ordinal = ANY(%s)", [e["model_version"], ords])}
        audit.record(c, ctx, "measurements.uploaded", "assay", aid, {"plate": plate, "stored": len(results), "simulated": True})
        return 201, {"experiment_id": experiment_id, "plate": plate, "compounds": len(ords), "measurements": len(results), "actives_found": int(sum(v >= world.ACTIVE_LIMIT for v in agg.values())), "mean_measured": round(float(np.mean(list(agg.values()))), 2), "mean_predicted": round(float(np.mean([pred[o] for o in ords])), 2),
                     "rmse_prediction_vs_plate": round(float(np.sqrt(np.mean([(pred[o] - agg[o]) ** 2 for o in ords]))), 3), "status": "measured", "note": "values are the generator's hidden truth plus assay noise; a real lab would upload them through POST /v1/assays/{id}/measurements"}
    return run(ctx, idem, {}, work)


# ---------------------------------------------------------------- evidence and trace

@app.get("/v1/targets/{target_id}/evidence", tags=["targets"], summary="Everything known about the target: measurements, actives, model versions with calibration, the top predicted candidates, and activity cliffs in the data")
def evidence(target_id: uuid.UUID, ctx: Ctx = Depends(auth("library:read"))):
    with db.tx(ctx.tenant_id) as c:
        lib, w = library(c)
        tg = target(c)
        if not tg or tg["id"] != target_id:
            raise Problem(404, "target_not_found")
        aid = c.execute("SELECT id FROM assays LIMIT 1").fetchone()["id"]
        agg = measured(c, aid)
        versions = c.execute("SELECT version, state, data_snapshot, metrics, created_at FROM model_registry WHERE name = %s ORDER BY created_at", [f"{MODEL_NAME}-{ctx.tenant_id}"]).fetchall()
        n_meas = c.execute("SELECT count(*) AS n, count(DISTINCT plate) AS plates FROM measurements WHERE assay_id = %s", [aid]).fetchone()
        top = c.execute("SELECT p.compound_id, cp.canonical_smiles, p.value, p.uncertainty FROM predictions p JOIN compounds cp ON cp.id = p.compound_id WHERE p.model_version = %s AND cp.id NOT IN (SELECT compound_id FROM measurements) ORDER BY p.value DESC LIMIT 10", [versions[-1]["version"]]).fetchall() if versions else []
        experiments = c.execute("SELECT id, status, hypothesis, model_version, (design->>'selected')::int AS compounds, created_at FROM experiments ORDER BY created_at").fetchall()
    midx = np.array(sorted(agg))
    vals = np.array([agg[i] for i in midx])
    cliffs = [{"a": {"compound_id": cid(ctx.tenant_id, ia), "smiles": w["mols"][ia]["smiles"], "pIC50": round(va, 2)}, "b": {"compound_id": cid(ctx.tenant_id, ib), "smiles": w["mols"][ib]["smiles"], "pIC50": round(vb, 2)}, "tanimoto": s_, "gap": round(abs(va - vb), 2)}
              for ia, va, ib, vb, s_ in cliffs_for(w["seed"], tuple(int(i) for i in midx), tuple(round(float(v), 4) for v in vals))]
    return jsonable_encoder({"target": {k: tg[k] for k in ("id", "gene_symbol", "uniprot_id", "name")}, "assay_id": aid, "measurements": n_meas["n"], "plates": n_meas["plates"], "measured_compounds": len(midx), "actives_measured": int((vals >= world.ACTIVE_LIMIT).sum()), "active_limit_pIC50": world.ACTIVE_LIMIT,
                             "model_versions": versions, "top_predicted_unmeasured": top, "experiments": experiments, "activity_cliffs": cliffs[:8], "cliff_rule": "measured pairs with Tanimoto >= 0.7 and a potency gap >= 1.5 log units"})


@app.get("/v1/compounds/{compound_id}/trace", tags=["compounds"], summary="(+) Why a compound is recommended: prediction and band, the model version and its calibration, the measured neighbours, and the substructures that move the prediction")
def trace(compound_id: uuid.UUID, ctx: Ctx = Depends(auth("library:read"))):
    with db.tx(ctx.tenant_id) as c:
        lib, w = library(c)
        cp = c.execute("SELECT * FROM compounds WHERE id = %s", [compound_id]).fetchone()
        if not cp:
            raise Problem(404, "compound_not_found")
        version, model, metrics = current_model(c, ctx.tenant_id)
        aid = c.execute("SELECT id FROM assays LIMIT 1").fetchone()["id"]
        agg = measured(c, aid)
        pred = c.execute("SELECT value, uncertainty FROM predictions WHERE compound_id = %s AND model_version = %s", [compound_id, version]).fetchone()
        in_experiments = c.execute("SELECT e.id, e.status, ec.rank, ec.rationale FROM experiment_candidates ec JOIN experiments e ON e.id = ec.experiment_id WHERE ec.compound_id = %s", [compound_id]).fetchall()
        own = [dict(m) for m in c.execute("SELECT value, qualifier, replicate, plate, measured_at FROM measurements WHERE compound_id = %s ORDER BY measured_at", [compound_id])]
    mol = Chem.MolFromSmiles(cp["canonical_smiles"])
    fp = np.array(list(world.MORGAN.GetFingerprintAsNumPy(mol)), dtype=np.uint8)
    desc = np.array([[world.descriptors(mol)[k] for k in ("mol_weight", "logp", "tpsa", "h_donors", "h_acceptors", "rotatable_bonds", "aromatic_rings")]])
    x = engine.features(fp[None], desc)[0]
    mu, sigma = engine.predict(model, x[None])
    midx = np.array(sorted(agg))
    _, nn = engine.knn_predict(w["fps"][midx], np.array([agg[i] for i in midx]), fp[None], k=5, bv_train=measured_bitvectors(w["seed"], tuple(int(i) for i in midx)))
    bits = engine.bit_contributions(model, x, fp)
    frags = engine.bit_fragments(mol, [b for b, _ in bits])
    truth = {"what": "the hidden value for this compound", "pIC50": round(float(w["truth"][cp["ordinal"]]), 2)} if cp["ordinal"] is not None else None
    return jsonable_encoder({"compound_id": compound_id, "smiles": cp["canonical_smiles"], "inchikey": cp["inchikey"], "source": cp["source"], "descriptors": cp["descriptors"], "lipinski": world.lipinski(cp["descriptors"]),
                             "prediction": {"model_version": version, "pIC50": round(float(mu[0]), 2), "band_90": [round(float(mu[0] - engine.Z90 * sigma[0]), 2), round(float(mu[0] + engine.Z90 * sigma[0]), 2)], "uncertainty": round(float(sigma[0]), 2), "stored": pred and {"value": float(pred["value"]), "uncertainty": float(pred["uncertainty"])}},
                             "model_calibration": {k: metrics[k] for k in ("rmse", "inside_90_band", "knn_rmse", "trained_on", "tested_on")}, "own_measurements": own,
                             "nearest_measured": [{"compound_id": cid(ctx.tenant_id, int(midx[a])), "smiles": w["mols"][int(midx[a])]["smiles"], "tanimoto": s_, "measured_pIC50": round(agg[int(midx[a])], 2)} for a, s_ in nn[0]],
                             "substructures_driving_the_prediction": [{"bit": b, "effect_on_pIC50": e, "fragment": frags.get(b)} for b, e in bits], "in_experiments": in_experiments, "simulation_truth": truth,
                             "how": "effects are what the prediction loses when that fingerprint bit is switched off; fragments are the atom environment the bit stands for in this molecule"})
