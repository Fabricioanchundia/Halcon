# Imagen del scorer de Halcón. Se construye en dos pasos para aprovechar la caché:
# primero las dependencias (cambian poco), después el código (cambia seguido).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

# Sin privilegios de root dentro del contenedor.
RUN useradd --create-home --uid 10001 halcon
USER halcon

EXPOSE 8000
ENTRYPOINT ["halcon"]
CMD ["--help"]
