# ---- builder: install pinned deps and the package into a venv ----
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# Dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then the package itself (non-editable; deps already pinned above).
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .

# ---- runtime: only the venv, a non-root user, and libgomp for XGBoost ----
FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /mlartifacts is owned by app so a named volume mounted there inherits it.
RUN useradd --create-home --uid 10001 app \
    && mkdir /mlartifacts \
    && chown app:app /mlartifacts

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Relative paths such as ingest's DEFAULT_CSV resolve from here.
WORKDIR /app
USER app

EXPOSE 8000
CMD ["uvicorn", "ai4i.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]