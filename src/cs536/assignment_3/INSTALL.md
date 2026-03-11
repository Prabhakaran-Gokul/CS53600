# Installation Guide

## Docker (Privileged/Host-Networked)

This Docker workflow uses privileged and host-networked settings. It can modify
host kernel/network state (module load/unload and `sysctl` changes), so it is
not a no-system-change path.

```bash
# From assignment_3 folder
cd src/cs536/assignment_3

# Build and run
docker-compose up -d
docker-compose exec dev bash

# Inside container
cd src/cs536/assignment_3
make && insmod tcp_custom.ko
sysctl -w net.ipv4.tcp_allowed_congestion_control="reno cubic custom"

# Run tests
PYTHONPATH=/workspace/src python3 -m cs536.assignment_3.run_tests \
    --server YOUR_SERVER --algorithms cubic reno custom --runs 5
```

## Native Installation

```bash
# Install dependencies
sudo apt-get install build-essential linux-headers-$(uname -r)

# Build and load
cd src/cs536/assignment_3
make
sudo insmod tcp_custom.ko
sudo sysctl -w net.ipv4.tcp_allowed_congestion_control="reno cubic custom"

# Test
PYTHONPATH=src python3 -m cs536.assignment_3.run_tests \
    --server YOUR_SERVER --algorithms cubic reno custom --runs 5

# Unload
sudo rmmod tcp_custom
```
