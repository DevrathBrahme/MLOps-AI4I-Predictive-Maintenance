# AI4I Predictive Maintenance: recommendations for a human operator

[![CI](https://github.com/DevrathBrahme/MLOps-AI4I-Predictive-Maintenance/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DevrathBrahme/MLOps-AI4I-Predictive-Maintenance/actions/workflows/ci.yml)

A multi-class failure-mode classifier on the AI4I 2020 predictive-maintenance
dataset, wrapped in a production-style MLOps stack: PostgreSQL queried with SQL,
MLflow experiment tracking and model registry, a FastAPI service, Docker Compose,
a pytest suite and GitHub Actions.

**The model informs; a person decides.** The API never triggers an action. For
each sensor reading it returns a flagged recommendation (risk tier, most likely
failure mode, runner-up mode and an uncalibrated confidence score) for a human
operator, and it logs every recommendation to Postgres before returning it.

> **Status:** work in progress. Drift monitoring (Evidently) and the full
> write-up (physics-informed features, evaluation, the human-in-the-loop design
> and its limitations) come next. This page currently covers what CI checks and
> how to run the stack.

## What CI checks on every push

| Job | What it proves |
|---|---|
| Tests (no services) | Unit, contract, API and model-registry tests on a clean machine, with no database or MLflow server |
| Integration tests (Postgres) | Postgres built from `db/init` and loaded from the CSV; Python features bit-identical to the SQL view; every `predictions_log` constraint |
| Bootstrap smoke (full stack) | The whole system rebuilt from the repo on an empty machine: the new model must reproduce the registered champion, and the API must log each recommendation before returning it |

## Run the stack

Requires Docker Engine with Compose v2. These commands rebuild everything from
an empty database, exactly as the CI smoke job does.

```bash
cp .env.example .env              # then set POSTGRES_PASSWORD
docker compose build
docker compose up -d --wait postgres mlflow
docker compose run --rm loader    # CSV -> Postgres (the only step that reads the CSV)
docker compose run --rm trainer python -m ai4i.train --model xgb
RUN_ID=$(docker compose run --rm -T trainer python -c "import mlflow; print(mlflow.search_runs(experiment_names=['ai4i-failure-classifier'])['run_id'].iloc[0])" | tail -n1)
docker compose run --rm trainer python -m ai4i.evaluate --run-id "$RUN_ID"   # sealed test set, used once
docker compose run --rm trainer python -m ai4i.registry --run-id "$RUN_ID"
docker compose run --rm trainer python -m ai4i.thresholds --model xgb
docker compose up -d --wait api
```

API docs: http://localhost:8000/docs · MLflow UI: http://localhost:5000

## Run the tests

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt && python -m pip install -e .
python -m pytest -m "not integration"   # no services needed
set -a; source .env; set +a
python -m pytest                        # everything; needs the stack above running
```