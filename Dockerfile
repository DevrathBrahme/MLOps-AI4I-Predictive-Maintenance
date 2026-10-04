# One Dockerfile, two images (D26, extended by D54):
#   runtime  default target: api, mlflow, loader, trainer
#   monitor  the runtime environment plus Evidently, for drift reports
# Dependencies, the project wheel and the OS layer are built once and shared.

# ---- Python dependencies: rebuilt only when requirements.txt changes ----
FROM python:3.12-slim AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Monitoring dependencies: the runtime set plus Evidently ----
FROM deps AS monitor-deps
COPY requirements-monitor.txt .
RUN pip install -r requirements-monitor.txt

# ---- The project as a wheel: the only stage that changes with src/ ----
FROM deps AS wheel
COPY pyproject.toml .
COPY src/ src/
RUN pip wheel --no-deps --wheel-dir /wheels .

# ---- Shared OS layer: OpenMP for XGBoost, non-root user, artifact dir ----
FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# /mlartifacts is owned by app so a named volume mounted there inherits it.
RUN useradd --create-home --uid 10001 app \
    && mkdir /mlartifacts \
    && chown app:app /mlartifacts
# The images contain neither git nor a .git directory, so MLflow's automatic git
# detection always fails; quiet stops GitPython warning about it on every run.
# Commit provenance comes from GIT_COMMIT instead (ai4i.train.provenance_tags).
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet
# Relative paths such as ingest's DEFAULT_CSV resolve from here.
WORKDIR /app

# ---- monitor image ----
FROM base AS monitor
COPY --from=monitor-deps /opt/venv /opt/venv
RUN --mount=type=bind,from=wheel,source=/wheels,target=/tmp/wheels \
    pip install --no-deps /tmp/wheels/*.whl
# Only Evidently's UI server reports usage, and this image never runs it;
# opt out anyway so no monitoring container can phone home.
ENV DO_NOT_TRACK=1
USER app
CMD ["python", "-c", "import evidently; print('evidently', evidently.__version__)"]

# ---- runtime image (the default target: keep it last) ----
FROM base AS runtime
COPY --from=deps /opt/venv /opt/venv
RUN --mount=type=bind,from=wheel,source=/wheels,target=/tmp/wheels \
    pip install --no-deps /tmp/wheels/*.whl
USER app
EXPOSE 8000
CMD ["uvicorn", "ai4i.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]