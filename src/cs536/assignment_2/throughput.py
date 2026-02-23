import tyro 
import requests
from pathlib import Path
from cs536.assignment_2 import ASSIGNMENT_2_PATH

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

