FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/cache DESYNTH_MODEL_DIR=/models DESYNTH_OUTPUT_DIR=/data/output
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN pip install -r requirements.txt -r requirements-web.txt
COPY desynth.py webapp.py embeds_cache.pt ./
COPY templates/ templates/
COPY static/ static/
RUN mkdir -p /models /data/cache /data/output
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)"
# One process owns the GPU lock; threads keep health and previews responsive.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "0", "webapp:create_app()"]
