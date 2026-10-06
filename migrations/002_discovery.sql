-- Domain model. Names follow the blueprint's schema; additions are marked (+). The model artefact lives in core's model_registry.

CREATE TABLE libraries (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants UNIQUE, name text NOT NULL, model_seed int NOT NULL);   -- (+)
SELECT enable_tenant_rls('libraries');

CREATE TABLE targets (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, gene_symbol text NOT NULL, uniprot_id text NOT NULL, name text NOT NULL, metadata jsonb NOT NULL);
SELECT enable_tenant_rls('targets');

CREATE TABLE compounds (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, canonical_smiles text NOT NULL, inchikey text NOT NULL, mol_weight numeric NOT NULL, source text NOT NULL, ordinal int,
  descriptors jsonb NOT NULL, cost real NOT NULL DEFAULT 1, registered_by uuid, registered_at timestamptz NOT NULL DEFAULT now(), UNIQUE (tenant_id, canonical_smiles), UNIQUE (tenant_id, inchikey));   -- (+) library ordinal, descriptors, cost, provenance
SELECT enable_tenant_rls('compounds');

CREATE TABLE assays (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, target_id uuid NOT NULL REFERENCES targets, assay_type text NOT NULL, protocol jsonb NOT NULL, unit text NOT NULL);
SELECT enable_tenant_rls('assays');

CREATE TABLE measurements (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, assay_id uuid NOT NULL REFERENCES assays, compound_id uuid NOT NULL REFERENCES compounds, value numeric NOT NULL, qualifier text NOT NULL DEFAULT '=', replicate int NOT NULL,
  measured_at timestamptz NOT NULL, plate text, recorded_by uuid, UNIQUE (assay_id, compound_id, replicate, plate));   -- (+) plate, provenance
CREATE INDEX measurements_compound ON measurements (compound_id);
SELECT enable_tenant_rls('measurements');

CREATE TABLE predictions (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, compound_id uuid NOT NULL REFERENCES compounds, target_id uuid NOT NULL REFERENCES targets, endpoint text NOT NULL, value numeric NOT NULL, uncertainty numeric NOT NULL,
  model_version text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (compound_id, target_id, endpoint, model_version));
SELECT enable_tenant_rls('predictions');

CREATE TABLE experiments (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, hypothesis text NOT NULL, budget numeric NOT NULL, status text NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed','approved','measured')), design jsonb NOT NULL, owner_id uuid NOT NULL,
  model_version text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), approved_by uuid);   -- (+) which model proposed it, who approved
SELECT enable_tenant_rls('experiments');

CREATE TABLE experiment_candidates (experiment_id uuid NOT NULL REFERENCES experiments ON DELETE CASCADE, compound_id uuid NOT NULL REFERENCES compounds, rank int NOT NULL, rationale jsonb NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants,
  PRIMARY KEY (experiment_id, compound_id));
SELECT enable_tenant_rls('experiment_candidates');
