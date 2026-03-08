FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    kmod \
    libelf-dev \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

COPY pyproject.toml uv.lock* ./
RUN uv sync

WORKDIR /workspace

CMD ["/bin/bash"]
