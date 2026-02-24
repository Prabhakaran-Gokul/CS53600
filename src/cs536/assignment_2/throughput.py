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

def store_q2_result(ip_used : List[str], tcp_stats : List[pd.DataFrame], frames: List[pd.DataFrame]):
    """
        Stores the required part of q2 in a csv 
    """
    

    for ip, tcp_stat, frame in zip(ip_used, tcp_stats, frames):
        tcp_stat["ip"] = ip
        frame["ip"] = ip

    tcp_all = pd.concat(tcp_stats, ignore_index=True) if tcp_stats else pd.DataFrame()
    frames_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    combined = pd.concat([tcp_all, frames_all], axis = 1, sort=False)
    out_csv = ASSIGNMENT_2_PATH / "results" / "q2_combined.csv"
    combined.to_csv(out_csv, index=False)


def plot_1c(frames : List[pd.DataFrame], used_ips : List[str], summary_rows : List[Dict[str, float]]):
    """
    Creates the plot and table for 1 c
    """
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


def plot_2b(ip_used : List[str], tcp_stats : List[pd.DataFrame], frames: List[pd.DataFrame]):

    

    ## pick first ip
    ip: str = ip_used[0]
    tcp_stat: pd.DataFrame = tcp_stats[0]
    throughput_df: pd.DataFrame = frames[0]

    ## i)
    nrows, ncols = 4, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 6 * nrows))

    axes[0].scatter(tcp_stat['ts'], tcp_stat['snd_cwnd'])
    axes[0].plot(tcp_stat['ts'], tcp_stat['snd_cwnd'], label = ip)
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel(f"snd_cwnd")
    axes[0].set_title("snd_cwnd")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Destination", loc="best")

    axes[1].scatter(tcp_stat['ts'], tcp_stat['rtt_us'])
    axes[1].plot(tcp_stat['ts'], tcp_stat['rtt_us'], label = ip)
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel(f"rtt_us")
    axes[1].set_title("rtt_us")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Destination", loc="best")

    #if not tcp_stat["total_retrans"] == -1:
    axes[2].scatter(tcp_stat['ts'], tcp_stat['total_retrans'])
    axes[2].plot(tcp_stat['ts'], tcp_stat['total_retrans'], label = ip)
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel(f"total_retrans (loss signal)")
    axes[2].set_title("total_retrans (loss signal)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(title="Destination", loc="best")

    axes[3].scatter(throughput_df['t_mid'], throughput_df['goodput_bps'])
    axes[3].plot(throughput_df['t_mid'], throughput_df['goodput_bps'], label = ip)
    axes[3].set_xlabel("Timestamp")
    axes[3].set_ylabel(f"goodput_bps")
    axes[3].set_title("goodput_bps")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(title="Destination", loc="best")

    plt.tight_layout()
    time_series_stored_path = ASSIGNMENT_2_PATH / "results" / "2b_i.png"
    plt.savefig(time_series_stored_path)

    
    '''
    ii) Scatter plots showing relationships:
        • snd cwnd vs goodput,
        • RTT vs goodput,
        • loss signal (e.g., # retransmissions or # timeouts) vs goodput.
    '''

    nrows, ncols = 3, 1
    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 12, nrows * 6))

    axes[0].scatter(throughput_df['goodput_bps'], tcp_stat['snd_cwnd'])
    axes[0].set_xlabel("goodput_bps")
    axes[0].set_ylabel(f"snd_cwnd")
    axes[0].set_title(f"goodput_bps vs snd_cwnd {ip}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(throughput_df['goodput_bps'], tcp_stat['rtt_us'])
    axes[1].set_xlabel("goodput_bps")
    axes[1].set_ylabel(f"rtt_us")
    axes[1].set_title(f"goodput_bps vs rtt_us {ip}")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(throughput_df['goodput_bps'], tcp_stat['total_retrans'])
    axes[2].set_xlabel("goodput_bps")
    axes[2].set_ylabel(f"total_retrans (loss_signal)")
    axes[2].set_title(f"goodput_bps vs loss_signal {ip}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_stored_path = ASSIGNMENT_2_PATH / "results" / "2b_ii.png"
    plt.savefig(scatter_stored_path)





def run_on_host(ip : str, duration : float, interval : float,
                 verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:

    df, stats, tcp_stats = run_one_destination_with_sampling(host= ip, port = 5201, interval=interval, duration=duration, verbose=verbose)

    return df, stats, tcp_stats

def run(n : int = 2, duration: int = 10, interval: float = 1.0, 
        verbose: bool = False, q1: bool = True, q2:bool = True):
    ''''
    runs on n hosts and retry on failure
    '''
    frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float]] = []
    tcp_summary: List[pd.DataFrame] = []
    success_counter: int = 0
    used_ips: List[str] = []
    ip_list: List[str] = fetch_ip_list()
    ip_list = ["160.242.19.254",
        "185.93.1.65", "109.61.86.65", "185.152.67.2", "195.181.162.195", "185.59.223.8", "66.35.22.79", "209.40.123.215",
               "109.61.86.65"] # for testing

    while success_counter < n:

        target_ip = random.choice(ip_list)
        try:
            ip_list.remove(target_ip)
        except ValueError:
            continue

        try:
            print(f"\n=== {target_ip} | duration={duration}s | interval={interval}s ===")
            df, stats, tcp_stats = run_on_host(ip=target_ip, duration=duration,interval=interval,verbose=verbose)
            print(f"  min={format_bps(stats['min'])}, median={format_bps(stats['median'])}, "
              f"avg={format_bps(stats['avg'])}, p95={format_bps(stats['p95'])}")
            frames.append(df)
            summary_rows.append(stats)
            used_ips.append(target_ip)
            tcp_summary.append(tcp_stats)
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            print(f"[error] error on ip {target_ip}: {e}")
            continue
        success_counter += 1

    store_q2_result(ip_used= used_ips, tcp_stats= tcp_summary, frames = frames)
    if q1:
        plot_1c(frames=frames, used_ips=used_ips, summary_rows=summary_rows)

    if q2:
        plot_2b(ip_used=used_ips, tcp_stats=tcp_summary, frames=frames)



'''
    Example: PYTHONPATH=src python3 -m cs536.assignment_2.throughput --n 4 --duration 10 --interval 1 --verbose --q1 --q2
'''

if __name__ == "__main__":
    tyro.cli(run)

