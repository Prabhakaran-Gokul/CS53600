FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Create a venv to avoid Ubuntu's externally-managed-environment restriction
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps (expects requirements.txt under /app/src/cs536/assignment_2)
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/src/cs536/assignment_2/requirements.txt

ENV PYTHONPATH=/app/src

CMD ["python3", "-m", "cs536.assignment_2.run", "--file", "src/cs536/assignment_2/results/ip_addresses.txt", "--n", "5", "--duration", "60", "--interval", "1", "--verbose"]
