# Assignment 3: Congestion Control Algorithm Comparison

This assignment tests and compares the performance of different TCP congestion control algorithms using the socket program from Assignment 2.

## Overview

The code in this directory:
1. Extends the Assignment 2 TCP client to support congestion control algorithm selection
2. Runs automated tests with CUBIC, RENO, and (later) custom congestion control algorithms
3. Collects throughput, RTT, and packet loss metrics
4. Generates comparative visualizations and statistical analysis
5. Uses **loguru** for professional logging and output formatting

## Files

- `tcp_client_cc.py` - TCP client with congestion control algorithm selection
- `run_tests.py` - Script to run multiple test iterations with different algorithms
- `analyze_results.py` - Analysis and visualization of test results
- `check_cc_algorithms.py` - Utility to check available congestion control algorithms
- `logger_config.py` - Loguru logging configuration
- `requirements.txt` - Python dependencies including loguru
- `results/` - Directory for storing test results and plots

## Installation

Install the required dependencies:

```bash
# Install from requirements file
pip install -r src/cs536/assignment_3/requirements.txt

# Or install individually
pip install loguru pandas numpy matplotlib tyro
```

## Custom Congestion Control Algorithm

### Implementation

The custom algorithm is implemented as a **Linux kernel module** in [tcp_custom.c](tcp_custom.c).

**Build and install:**
```bash
cd src/cs536/assignment_3
make
make install
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Usage

### Step 1: Run Tests

Run tests with CUBIC and RENO algorithms:

```bash
PYTHONPATH=src python -m cs536.assignment_3.run_tests --server SERVER_IP --runs 5
```

**Parameters:**
- `--server`: Target iperf3 server IP or hostname (required)
- `--port`: Server port (default: 5201)
- `--duration`: Test duration in seconds (default: 10.0)
- `--interval`: Sampling interval in seconds (default: 0.5)
- `--runs`: Number of test runs per algorithm (default: 3)
- `--algorithms`: List of algorithms to test (default: cubic reno)
- `--wait-between-runs`: Wait time between runs in seconds (default: 2.0)
- `--verbose`: Enable verbose output

**Example with custom algorithm:**
```bash
PYTHONPATH=src python -m cs536.assignment_3.run_tests \
    --server 185.93.1.65 \
    --algorithms cubic reno custom \
    --runs 5
```

**Example with verbose output:**
```bash
PYTHONPATH=src python -m cs536.assignment_3.run_tests \
    --server 185.93.1.65 \
    --duration 20 \
    --runs 5 \
    --verbose
```

### Step 2: Analyze Results

After running tests, analyze and visualize the results:

```bash
PYTHONPATH=src python -m cs536.assignment_3.analyze_results
```

This will generate:
- Throughput comparison plots (time series and box plots)
- RTT comparison plots
- Packet loss comparison
- Congestion window evolution
- Summary statistics table

All outputs are saved in the `results/` directory.

## Collected Metrics

For each congestion control algorithm, the tests collect:

1. **Throughput**: Application-layer goodput in bits per second
   - Measured using `TCP_INFO` bytes_acked
   - Sampled at regular intervals (default: 0.5s)

2. **Round-Trip Time (RTT)**: Network latency in microseconds/milliseconds
   - Obtained from `TCP_INFO` statistics
   - Includes RTT variance (rttvar)

3. **Packet Loss**: Retransmissions and lost packets
   - Total retransmissions
   - Lost packets count
   - Retransmitted packets count

4. **Congestion Window (cwnd)**: TCP congestion window size
   - Measured in packets
   - Shows algorithm's congestion response

## Output Files

After running tests and analysis, the `results/` directory contains:

- `goodput_all_runs.csv` - Raw goodput data for all runs
- `tcp_stats_all_runs.csv` - Raw TCP statistics for all runs
- `summary_statistics.csv` - Summary statistics per algorithm and run
- `analysis_summary_table.csv` - Aggregated comparison table
- `throughput_comparison.png` - Throughput visualization
- `rtt_comparison.png` - RTT visualization
- `loss_comparison.png` - Packet loss visualization
- `cwnd_comparison.png` - Congestion window visualization

## System Requirements

### Linux Kernel Modules

To test different congestion control algorithms, they must be available in the kernel:

```bash
# List available congestion control algorithms
sysctl net.ipv4.tcp_available_congestion_control

# List currently loaded modules
sysctl net.ipv4.tcp_allowed_congestion_control

# Load specific modules (if needed)
sudo modprobe tcp_cubic
sudo modprobe tcp_reno
sudo modprobe tcp_bbr  # for BBR testing
```

### Python Dependencies

The code uses dependencies from Assignment 2 plus loguru for logging:
- pandas
- numpy
- matplotlib
- tyro
- **loguru** - Beautiful logging for clear, colorized output

Install all dependencies:
```bash
pip install -r src/cs536/assignment_3/requirements.txt
```

## Logging with Loguru

Assignment 3 uses [loguru](https://github.com/Delphire/loguru) for professional, colorized logging output. All test progress, results, and errors are logged with appropriate severity levels:

- `logger.info()` - General information (green)
- `logger.success()` - Successful operations (green, bold)
- `logger.warning()` - Warnings (yellow)
- `logger.error()` - Errors (red)
- `logger.debug()` - Detailed debug info (when verbose mode is enabled)

The logs are automatically timestamped and color-coded for easy reading in the terminal.

## Adding Custom Congestion Control Algorithm

To test your custom congestion control algorithm:

1. Load your kernel module or eBPF program
2. Verify it's available: `sysctl net.ipv4.tcp_available_congestion_control`
3. Run tests including your algorithm:
   ```bash
   PYTHONPATH=src python -m cs536.assignment_3.run_tests \
       --server SERVER_IP \
       --algorithms cubic reno YOUR_ALGORITHM \
       --runs 5
   ```

## Notes

- Tests require a running iperf3 server on the target host
- Ensure sufficient permissions to set TCP socket options
- Network conditions should be stable for meaningful comparisons
- Multiple runs help account for network variability
- 10-second duration per test is default (configurable)

## Troubleshooting

**"Failed to set congestion control"**: The algorithm may not be available in your kernel. Check available algorithms with `sysctl`.

**Connection errors**: Verify the iperf3 server is running and reachable.

**Permission denied**: Some systems require elevated privileges to set TCP options. Try running with sudo if necessary.

**Inconsistent results**: Increase the number of runs (`--runs`) and ensure stable network conditions.
