# AI DTP QA Agent -- production container image.
#
# Build:  docker build -t dtp-qa-agent .
# Run:    docker run -p 8501:8501 --env-file .env dtp-qa-agent
#
# GEMINI_API_KEY (optional) and any other variables documented in
# .env.example are read from the container environment -- pass them with
# --env-file .env or -e GEMINI_API_KEY=... at `docker run` time.

FROM python:3.11-slim

# PyMuPDF and Pillow need these system libraries to build/run correctly on
# a slim base image; keep the layer thin and clean up apt lists afterward.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg62-turbo-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user -- standard container hardening.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
