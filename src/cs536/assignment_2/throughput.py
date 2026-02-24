# goodput_tcpinfo_monitor.py
# -*- coding: utf-8 -*-
"""
Measure application goodput using bytes_acked from TCP_INFO at fixed intervals (e.g., 0.2s or 1s)
for one or more iperf3 servers, plot a combined time series, and print/save summary stats.

Usage examples:
  python goodput_tcpinfo_monitor.py --dest 192.0.2.10,203.0.113.5:5202 --duration 20 --interval 1
  python goodput_tcpinfo_monitor.py --dest 10.0.0.5 --duration 10 --interval 0.2 --outfile plot.png

Place this script in the same folder as your `iperf3_tcp_client.py`.
"""


from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import tyro 
import random
import requests
from pathlib import Path
from cs536.assignment_2 import ASSIGNMENT_2_PATH
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from cs536.assignment_2.iperf3_tcp_client import run_one_destination_with_sampling


def fetch_ip_list(
    url: str = "https://iperf3serverlist.net/api/servers",
    timeout: float = 10.0,
    file: Path = ASSIGNMENT_2_PATH / "results" / "ip_addresses.txt",
) -> list[str]:
    """Fetch and extract IP addresses from the iperf3 server API and saves it to a file.

    Retrieves server data from the API and extracts unique IPv4 addresses.

    Args:
        url: The API URL to fetch server data from. Defaults to iperf3serverlist.net API.
        timeout: Timeout in seconds for the HTTP request. Defaults to 10.0.
        file: Path to save the IP addresses.

    Returns:
        A sorted list of unique IPv4 addresses.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    servers = resp.json()

    ips: set[str] = set()
    for server in servers:
        ip = server.get("ip", "")
        # Add both IPv4 addresses and hostnames
        if ip and ip.strip():
            ips.add(ip)

    # Save to file
    with open(file, "w", encoding="utf-8") as f:
        for ip in sorted(ips):
            f.write(f"{ip}\n")

    return list(sorted(ips))




def format_bps(bps: float) -> str:
    #bps /= 1000
    units = ["b/s", "Kb/s", "Mb/s", "Gb/s", "Tb/s"]
    val = float(bps)
    i = 0
    '''
    while val >= 1000 and i < len(units) - 1:
        val /= 1000.0
        i += 1
    return f"{val:.2f} {units[i]}"
    '''

    return f"{val} bits / s"

def run_on_host(ip : str, duration : float, interval : float, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:

    df, stats = run_one_destination_with_sampling(host= ip, port = 5201, interval=interval, duration=duration, verbose=verbose)

    return df, stats

def run(n : int = 2, duration: int = 10, interval: float = 1.0, verbose: bool = False):
    ''''
    runs on n hosts and retry on failure
    '''
    frames = []
    summary_rows = []
    success_counter = 0
    used_ips = []
    ip_list: List[str] = fetch_ip_list()
    ip_list = ["185.93.1.65", "109.61.86.65", "185.152.67.2", "195.181.162.195", "185.59.223.8", "66.35.22.79", "209.40.123.215",
               "109.61.86.65"] # for testing

    while success_counter < n:

        target_ip = random.choice(ip_list)
        try:
            ip_list.remove(target_ip)
        except ValueError:
            continue

        try:
            print(f"\n=== {target_ip} | duration={duration}s | interval={interval}s ===")
            df, stats = run_on_host(ip=target_ip, duration=duration,interval=interval,verbose=verbose)
            print(f"  min={format_bps(stats['min'])}, median={format_bps(stats['median'])}, "
              f"avg={format_bps(stats['avg'])}, p95={format_bps(stats['p95'])}")
            frames.append(df)
            summary_rows.append(stats)
            used_ips.append(target_ip)
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            print(f"[error] error on ip {target_ip}: {e}")
            continue
        success_counter += 1

    ## plotting time series, currently the y axis is in scientific notation because the value is large
    ## TODO: sanity check of goodut value
    fig, ax = plt.subplots(figsize=(12, 6))

    for df, target_ip in zip(frames, used_ips):
        ax.scatter(df['t_mid'], df['goodput_bps'])
        ax.plot(df['t_mid'], df['goodput_bps'], label=target_ip)
    ax.set_xlabel("Interval time mid point")
    ax.set_ylabel(f"bit / s")
    ax.set_title("Goodput")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Destination", loc="best")
    plt.tight_layout()

    plot_stored_path = ASSIGNMENT_2_PATH / "results" / "1c_plot.png"
    plt.savefig(plot_stored_path)
    #plt.show()

    ## plotting table
    # a summary table (min/median/avg/p95 throughput) for each destination.
    fig, ax = plt.subplots(figsize=(12,6))
    data = []
    col_labels = ["IP", "min", "median", "avg", "p95"]

    for summary, target_ip in zip(summary_rows, used_ips):
        data.append([target_ip, summary["min"], summary["median"], summary["avg"], summary["p95"]])

    ax.axis("off")  # hide axes

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # (x, y) scaling of cell padding

    plt.tight_layout()
    table_stored_path = ASSIGNMENT_2_PATH / "results" / "1c_table.png"
    plt.savefig(table_stored_path)


'''
    Example: PYTHONPATH=src python3 -m cs536.assignment_2.throughput --n 4 --duration 10 --interval 1 --verbose
'''

if __name__ == "__main__":
    tyro.cli(run)

