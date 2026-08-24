FROM flwr/superexec:1.28.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    meson \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]
