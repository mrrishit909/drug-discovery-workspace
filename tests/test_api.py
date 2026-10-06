"""Integration, end-to-end and security tests against a real Postgres."""
import uuid

import pytest

from core import db, jobs, scenario
from dd import api

from .conftest import bearer

LEAD, SCI, VIEWER, OTHER = bearer("lab", "lead"), bearer("lab", "scientist"), bearer("lab", "viewer"), bearer("other", "lead")
ZERO = "00000000-0000-0000-0000-000000000000"


def test_nothing_works_before_a_library_is_loaded(client):
    assert client.post("/v1/compounds", headers=SCI, json={"smiles": ["CCO"]}).json()["code"] == "no_library" and client.post("/v1/models/train", headers=LEAD).json()["code"] == "no_library"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    load = demo["load"]["result"]
    assert load["compounds"] == 10000 == one("SELECT count(*) AS n FROM compounds WHERE source = 'library'")["n"] and load["measurements"] == 600
    t1 = demo["train1"]["result"]
    assert t1["version"].endswith("v1") and t1["held_out_report"]["rmse"] < t1["held_out_report"]["knn_rmse"] and one("SELECT count(*) AS n FROM predictions WHERE model_version = %s", [t1["version"]])["n"] == 10000
    r = demo["reg"]
    assert len(r["added"]) == 1 and len(r["duplicates_ignored"]) == 1 and r["invalid"] == ["C1CC(("] and r["added"][0]["lipinski"]["passes"]
    assert len(demo["props"]["items"]) == 2 and demo["props"]["items"][0]["inchikey"] == r["added"][0]["inchikey"]
    a = demo["aff"]
    assert a["model_version"] == t1["version"] and len(a["items"]) == 3 and all(i["band_90"][0] <= i["predicted_pIC50"] <= i["band_90"][1] and len(i["nearest_measured"]) == 3 for i in a["items"])
    p = demo["prop"]
    assert p["status"] == "proposed" and p["spent"] <= p["budget"] and p["selected"] == len(p["items"]) == one("SELECT count(*) AS n FROM experiment_candidates")["n"] and p["simulation_truth"]["proposed"] > p["simulation_truth"]["random"]
    assert demo["deny"]["code"] == "forbidden" and demo["ok"]["status"] == "approved"
    pl = demo["plate"]
    assert pl["compounds"] == p["selected"] and pl["measurements"] == 2 * p["selected"] and one("SELECT status FROM experiments WHERE id = %s", [p["experiment_id"]])["status"] == "measured"
    assert demo["manual"] == {**demo["manual"], "received": 3, "stored": 2, "duplicates_ignored": 1}
    t2 = demo["train2"]["result"]
    assert t2["version"].endswith("v2") and t2["measured_compounds"] > t1["measured_compounds"] and t2["previous_version"]["version"] == t1["version"] and one("SELECT state FROM model_registry WHERE version = %s", [t1["version"]])["state"] == "retired"
    tr = demo["trace"]
    assert tr["prediction"]["model_version"] == t2["version"] and len(tr["nearest_measured"]) == 5 and len(tr["substructures_driving_the_prediction"]) <= 5 and tr["own_measurements"] and tr["in_experiments"][0]["rank"] == 1
    ev = demo["evid"]
    assert ev["measured_compounds"] == t2["measured_compounds"] and len(ev["model_versions"]) == 2 and len(ev["experiments"]) == 1 and all(c["tanimoto"] >= 0.7 and c["gap"] >= 1.5 for c in ev["activity_cliffs"])
    assert demo["audit"]["chain_valid"] and {"library.loaded", "model.trained", "compounds.registered", "experiment.proposed", "experiment.approved", "measurements.uploaded"} <= {e["action"] for e in demo["audit"]["events"]}


def test_registration_and_prediction_validation(client, demo):
    assert client.post("/v1/compounds", headers=SCI, json={"smiles": ["not a molecule"]}).json()["code"] == "invalid_smiles"
    assert client.post("/v1/predict/target-affinity", headers=SCI, json={"smiles": ["C1CC(("]}).json()["code"] == "invalid_smiles" and client.post("/v1/predict/target-affinity", headers=SCI, json={}).json()["code"] == "nothing_to_predict"
    assert client.post("/v1/predict/target-affinity", headers=SCI, json={"compound_ids": [ZERO]}).json()["code"] == "compound_not_found"
    assert client.post("/v1/predict/properties", headers=SCI, json={"smiles": ["xyz"]}).json()["items"][0]["error"]


def test_measurement_validation(client, demo, seeded):
    aid = str(uuid.uuid5(seeded["lab"], "assay"))
    good = {"compound_id": str(api.cid(seeded["lab"], 9)), "value": 6.0, "qualifier": "=", "replicate": 1, "measured_at": "2026-10-06T10:00:00Z"}
    post = lambda **kw: client.post(f"/v1/assays/{aid}/measurements", headers=SCI, json={"plate": "T-1", "measurements": [{**good, **kw}]})     # noqa: E731
    assert post().json()["stored"] == 1 and post().json()["duplicates_ignored"] == 1 and post(compound_id=ZERO).json()["code"] == "compound_not_found"
    assert post(measured_at="2026-10-06T10:00:00").json()["code"] == "timezone_required" and post(value=20).status_code == 422 and post(qualifier="~").status_code == 422
    assert client.post(f"/v1/assays/{ZERO}/measurements", headers=SCI, json={"plate": "T-1", "measurements": [good]}).status_code == 404


def test_experiment_rules(client, demo):
    eid = demo["prop"]["experiment_id"]
    assert client.post(f"/v1/experiments/{eid}/approve", headers=LEAD).json()["code"] == "not_proposed" and client.post(f"/v1/experiments/{ZERO}/approve", headers=LEAD).status_code == 404
    assert client.post(f"/v1/experiments/{eid}/results:simulate", headers=SCI).json()["code"] == "not_approved"
    own = client.post("/v1/experiments/propose", headers=LEAD, json={"hypothesis": "a lead proposing", "plate_size": 16, "budget": 20}).json()
    assert client.post(f"/v1/experiments/{own['experiment_id']}/approve", headers=LEAD).json()["code"] == "same_person"
    b = client.post("/v1/active-learning/batch", headers=SCI, json={"plate_size": 24, "budget": 30, "kappa": 0}).json()
    assert len(b["items"]) <= 24 and b["spent"] <= 30 and b["comparison"]["proposed"]["n"] == len(b["items"])


def test_roles(client, demo):
    assert client.get(f"/v1/compounds/{ZERO}/trace").status_code == 401 and client.get(f"/v1/compounds/{ZERO}/trace", headers={"Authorization": "Bearer nope"}).status_code == 401
    for path, payload in (("/v1/compounds", {"smiles": ["CCO"]}), ("/v1/predict/properties", {"smiles": ["CCO"]}), ("/v1/experiments/propose", {"hypothesis": "viewer tries"}), ("/v1/active-learning/batch", {})):
        assert client.post(path, headers=VIEWER, json=payload).status_code == 403, path
    for path in ("/v1/library:load", "/v1/models/train", f"/v1/experiments/{ZERO}/approve"):
        assert client.post(path, headers=SCI, json={}).status_code == 403, path
    assert client.get("/v1/audit", headers=SCI).status_code == 403 and client.get("/v1/audit", headers=LEAD).status_code == 200


def test_another_lab_sees_nothing(client, demo, seeded):
    cpd = str(api.cid(seeded["lab"], 2))
    assert client.get(f"/v1/compounds/{cpd}/trace", headers=OTHER).json()["code"] == "no_library" and client.post("/v1/predict/target-affinity", headers=OTHER, json={"smiles": ["CCO"]}).json()["code"] == "no_library"
    for table in ("libraries", "targets", "compounds", "assays", "measurements", "predictions", "experiments", "experiment_candidates"):
        assert one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["other"])["n"] == 0 < one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["lab"])["n"], table
    with pytest.raises(Exception, match="row-level security"), db.tx(seeded["other"]) as c:
        c.execute("INSERT INTO libraries (id, tenant_id, name, model_seed) VALUES (%s,%s,'x',1)", [uuid.uuid4(), seeded["lab"]])


def test_idempotent_writes(client, demo):
    body = {"hypothesis": "Aryl sulfonamides with a free aniline bind RBK1; test the series breadth and the trifluoromethyl question", "plate_size": 96, "budget": 200, "kappa": 1.0}
    again = client.post("/v1/experiments/propose", headers={**SCI, "Idempotency-Key": "demo-1:prop"}, json=body)
    assert again.status_code == 201 and again.headers["Idempotent-Replay"] == "true" and again.json()["experiment_id"] == demo["prop"]["experiment_id"]
    assert client.post("/v1/experiments/propose", headers={**SCI, "Idempotency-Key": "demo-1:prop"}, json={**body, "kappa": 2.0}).json()["code"] == "idempotency_key_reused"
    assert client.post("/v1/library:load", headers=LEAD, json={"seed": 17}).json()["code"] == "library_exists"
