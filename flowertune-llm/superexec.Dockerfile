FROM flwr/superexec:1.28.0

USER root

# Use Aliyun mirror for faster package installation in China
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    meson \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
    
# Install Kubernetes Python client
RUN pip install --no-cache-dir kubernetes>=28.1.0

WORKDIR /app

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir . -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
ENTRYPOINT ["flower-superexec"]
