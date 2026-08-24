FROM python:3.13-slim

# The dashboard uses the Kubernetes Python client for its read-only TKE
# inspection and SwanLab only for asynchronous scalar-metric publishing. Both
# dependencies stay out of the Flower training image.
RUN pip install --no-cache-dir --disable-pip-version-check \
    swanlab==0.9.4 \
    'kubernetes>=28.1.0,<33'

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESULTS_ROOT=/app/results \
    PORT=8080

WORKDIR /app/dashboard
COPY experiment-dashboard/server.py ./server.py
COPY experiment-dashboard/swanlab_sync.py ./swanlab_sync.py
COPY experiment-dashboard/static ./static
RUN useradd --create-home --uid 10001 dashboard
USER 10001

EXPOSE 8080
CMD ["python", "server.py"]
