import platform
import re
import shutil
import subprocess

import matplotlib.pyplot as plt


def traceroute(ip):
    # Use appropriate command based on platform
    plat = platform.system().lower()
    if "windows" in plat:
        cmd = ["tracert", ip]
    else:
        # Check which traceroute command is available
        if shutil.which("traceroute"):
            cmd = ["traceroute", "-m", "30", ip]
        elif shutil.which("tracepath"):
            cmd = ["tracepath", "-m", "30", ip]
        else:
            raise RuntimeError(
                "Neither 'traceroute' nor 'tracepath' found. "
                "Please install traceroute: sudo apt-get install traceroute"
            )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"Traceroute to {ip} timed out")
        return []

    hops = []

    for line in result.stdout.splitlines():
        # get average RTT（ms）for every router
        # filter: just get lines have value, throw lines like (* * * time out)
        times = re.findall(r"(\d+)\s*ms", line)
        if times:
            avg_rtt = round(sum(map(float, times)) / len(times), 2)
            hops.append(avg_rtt)

    return hops


################################
# should be do more work for give a .csv file (contain ips) and return a list contain random 5 ip addresses
################################
ips = ["160.242.19.254", "spd-uswb.hostkey.com"]

hop_rtts_list = []
hop_rtt_increase = []

# get hop list for every ip address.
for ip in ips:
    hops = traceroute(ip)
    hop_rtts_list.append(hops)
    rtt_increase = [hops[0]]
    for i in range(len(hops)):
        if i > 0:
            rtt_increase.append(hops[i] - hops[i - 1])
    hop_rtt_increase.append(rtt_increase)

print(hop_rtts_list)
print(hop_rtt_increase)


#############################
#   GET Stacked bar chart
#############################
plt.figure()
bottom = [0] * len(ips)
for hop_idx in range(max(len(h) for h in hop_rtt_increase)):
    hop_values = []
    for h in hop_rtt_increase:
        hop_values.append(h[hop_idx] if hop_idx < len(h) else 0)

    plt.bar(ips, hop_values, bottom=bottom)
    bottom = [bottom[i] + hop_values[i] for i in range(len(ips))]

plt.ylabel("RTT (ms)")
plt.title("Latency Breakdown per Hop")
plt.savefig("latency_breakdown.pdf")

#####################################
#   GET hop count vs RTT graph
#####################################
plt.figure()
hop_counts = []
total_rtts = []

for hop_list in hop_rtts_list:
    hop_counts.append(len(hop_list))
    total_rtts.append(hop_list[-1])

plt.scatter(hop_counts, total_rtts)
plt.xlabel("Hop Count")
plt.ylabel("Total RTT (ms)")
plt.title("Hop Count vs RTT")
plt.savefig("hop_vs_rtt.pdf")
