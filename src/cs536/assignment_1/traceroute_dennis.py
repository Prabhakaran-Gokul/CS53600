from pathlib import Path
from subprocess import PIPE, Popen

import pandas as pd
import tyro
from matplotlib import pyplot as plt

from cs536.assignment_1 import ASSIGNMENT_1_PATH


def sort_trace(lines):
    cleaned_lines = []
    rtt = 0

    ## extract only information we want
    for line in lines:
        cleaned_line = []
        line = line.strip()
        line = line.split(" ")
        for item in line:
            if len(item) > 0 and not item == "ms":
                if item == "*":
                    item = "0"
                elif not item.find("<") == -1:
                    item = item[(item.find("<") + 1) :]
                cleaned_line.append(item)

        cleaned_lines.append(cleaned_line)

    hop_to_rtt = {}
    for line in cleaned_lines:
        if len(line) > 0:
            hop = int(line[0])
            trip_one = int(line[1])
            trip_two = int(line[2])
            trip_three = int(line[3])
            hop_to_rtt[hop] = (trip_one + trip_two + trip_three) / 3

    rtt = (int(cleaned_lines[-2][1]) + int(cleaned_lines[-2][2]) + int(cleaned_lines[-2][3])) / 3
    return hop_to_rtt, rtt


def rtt_by_hop(ip_to_hop_to_rtt, max_hops):
    hop_to_rtt_all = {}

    for ip, hop_rtt_map in ip_to_hop_to_rtt.items():
        for i in range(1, max_hops + 1):
            if i in hop_rtt_map:
                if i in hop_to_rtt_all:
                    hop_to_rtt_all[i].append(hop_rtt_map[i])
                else:
                    hop_to_rtt_all[i] = [hop_rtt_map[i]]

            else:
                if i in hop_to_rtt_all:
                    hop_to_rtt_all[i].append(0)
                else:
                    hop_to_rtt_all[i] = [0]
    return hop_to_rtt_all


"""
run tracer rouse on the give IP
"""


def tracrt(target_host):
    p = Popen(["tracert", target_host], stdout=PIPE)
    lines = []
    while True:
        line = p.stdout.readline()
        if not line:
            break
        else:
            lines.append(line.decode())

    ## when traceroute returns error message
    if len(lines) <= 2:
        return None

    lines = lines[:-1]
    lines = lines[4:]

    return lines


"""
Finds the IP addresses listed in input_file
Plots rounds trip time to each intermediate hop and plots them
Stores them in latency.pdf
"""


def run(input_file: Path = ASSIGNMENT_1_PATH / "traceroute_input.txt"):
    input_file = open(input_file, "r")
    target_hosts = []
    for line in input_file.readlines():
        target_hosts.append(line.strip())

    rtts = []
    hops = []
    ip_to_hop_to_rtt = {}

    for host in target_hosts:
        tracrt_res = tracrt(host)
        # tracrt_res = lines_one
        if tracrt_res is None:
            print("Unable to work with host " + str(host))
            continue
        hop_to_rtt, rtt = sort_trace(tracrt_res)
        ip_to_hop_to_rtt[host] = hop_to_rtt
        hops.append(len(hop_to_rtt.keys()))
        rtts.append(rtt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hop_to_rtt_all = rtt_by_hop(ip_to_hop_to_rtt, max(hops))

    df = pd.DataFrame.from_dict(hop_to_rtt_all, orient="index", columns=target_hosts)

    print(df)
    df.T.plot(kind="bar", stacked=True, ax=axes[0])

    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, ha="right")
    axes[0].set_title("RTT by Hop")
    axes[0].set_ylabel("RTT (MS)")
    axes[0].set_xlabel("IP Address")
    axes[0].legend(loc="upper left", title="hop number", bbox_to_anchor=(1.05, 1))

    axes[1].scatter(hops, rtts)
    axes[1].set_title("RTT VS HOPS")
    axes[1].set_ylabel("RTT (MS)")
    axes[1].set_xlabel("Hops")

    plt.savefig(ASSIGNMENT_1_PATH / "results" / "latency.pdf")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tyro.cli(run)
