"""Network latency measurement tool.

This module fetches IP addresses from iperf3serverlist.net, pings each host,
and geolocates them using the ip-api.com service.
"""

import csv
import math
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import tyro
from loguru import logger

from cs536.assignment_1 import ASSIGNMENT_1_PATH


def fetch_ip_list(
    url: str = "https://iperf3serverlist.net/api/servers",
    timeout: float = 10.0,
    file: Path = ASSIGNMENT_1_PATH / "results" / "ip_addresses.txt",
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


def ping_host(ip: str, count: int = 4, timeout: int = 3) -> dict[str, float] | None:
    """Ping a host and return latency statistics.

    Executes a ping command appropriate for the current platform (Windows/Linux/Mac)
    and parses the output to extract latency statistics.

    Args:
        ip: The IP address to ping.
        count: Number of ping packets to send. Defaults to 4.
        timeout: Timeout in seconds for each ping. Defaults to 3.

    Returns:
        A dictionary with keys 'min', 'avg', 'max' containing latency values in ms,
        or None if the ping fails or statistics cannot be parsed.
    """
    plat = platform.system().lower()
    if "windows" in plat:
        cmd = ["ping", ip, "-n", str(count), "-w", str(timeout * 1000)]
    else:
        cmd = ["ping", ip, "-c", str(count), "-W", str(timeout)]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        return None

    # Parse based on platform
    if "windows" in plat:
        # Windows format: Minimum = 10ms, Maximum = 20ms, Average = 15ms
        min_match = re.search(r"Minimum = ([0-9]+)ms", output)
        max_match = re.search(r"Maximum = ([0-9]+)ms", output)
        avg_match = re.search(r"Average = ([0-9]+)ms", output)
        
        if not (min_match and max_match and avg_match):
            return None
        
        min_rtt = float(min_match.group(1))
        max_rtt = float(max_match.group(1))
        avg_rtt = float(avg_match.group(1))
    else:
        # Linux/Mac format: rtt min/avg/max/mdev = 10.123/15.432/20.345/2.123 ms
        match = re.search(r"= ([0-9\.]+)/([0-9\.]+)/([0-9\.]+)", output)
        if not match:
            return None
        
        min_rtt, avg_rtt, max_rtt = map(float, match.groups())
    
    return {"min": min_rtt, "avg": avg_rtt, "max": max_rtt}


def geolocate_ip(ip: str) -> dict[str, str | float] | None:
    """Geolocate an IP address using ip-api.com.

    Queries the ip-api.com service to obtain geographic information for the given IP.

    Args:
        ip: The IP address to geolocate.

    Returns:
        A dictionary containing 'lat', 'lon', 'city', and 'country' keys,
        or None if the geolocation fails or the API returns an error.
    """
    try:
        r = requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,message,lat,lon,city,country",
            timeout=5,
        )
        data = r.json()
        if data.get("status") == "success":
            return {
                "lat": float(data.get("lat")),
                "lon": float(data.get("lon")),
                "city": data.get("city"),
                "country": data.get("country"),
            }
    except Exception:
        pass

    return None


def get_my_location() -> dict[str, str | float] | None:
    """Get the geographic location of the current machine.

    Uses ip-api.com to determine location based on public IP.

    Returns:
        A dictionary containing 'lat', 'lon', 'city', and 'country' keys,
        or None if the location cannot be determined.
    """
    try:
        r = requests.get(
            "http://ip-api.com/json/?fields=status,lat,lon,city,country,query",
            timeout=5,
        )
        data = r.json()
        if data.get("status") == "success":
            return {
                "lat": float(data.get("lat")),
                "lon": float(data.get("lon")),
                "city": data.get("city"),
                "country": data.get("country"),
                "ip": data.get("query"),
            }
    except Exception as e:
        logger.error(f"Failed to get current location: {e}")

    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth.

    Uses the Haversine formula to calculate distance in kilometers.

    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.

    Returns:
        Distance in kilometers.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Earth's radius in kilometers
    r = 6371

    return c * r


def read_ips_from_file(filepath: Path) -> list[str]:
    """Read IP addresses or hostnames from a file.

    Args:
        filepath: Path to file containing IP addresses or hostnames (one per line).

    Returns:
        List of IP addresses and hostnames.
    """
    ips: list[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Accept both IP addresses and hostnames
                ips.append(line)
    return ips


def plot_distance_vs_rtt(
    results: list[dict[str, str | dict[str, float] | dict[str, str | float] | None]],
    my_location: dict[str, str | float],
    output_file: Path = ASSIGNMENT_1_PATH / "results" / "distance_vs_rtt.pdf",
):
    """Create a scatter plot of distance vs RTT.

    Args:
        results: List of result dictionaries containing ping and geo data.
        my_location: Dictionary with 'lat' and 'lon' keys for current location.
        output_file: Path to save the PDF plot.
    """
    distances: list[float] = []
    rtts: list[float] = []
    labels: list[str] = []

    for result in results:
        ping: dict[str, float] | None = result.get("ping")
        geo: dict[str, str | float] | None = result.get("geo")
        ip: str | None = result.get("ip")

        # Skip if ping failed or geolocation failed
        if not ping or not geo:
            continue

        # Calculate distance
        assert isinstance(my_location["lat"], float)
        assert isinstance(my_location["lon"], float)
        assert isinstance(geo["lat"], float)
        assert isinstance(geo["lon"], float)
        distance = haversine_distance(
            my_location["lat"], my_location["lon"], geo["lat"], geo["lon"]
        )

        distances.append(distance)
        rtts.append(ping["avg"])
        labels.append(f"{ip} ({geo.get('city', 'Unknown')}, {geo.get('country', 'Unknown')})")

    if not distances:
        logger.warning("No valid data points to plot")
        return

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(distances, rtts, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)

    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("Average RTT (ms)", fontsize=12)
    plt.title(
        f"Network Latency vs Geographic Distance\nFrom: {my_location.get('city', 'Unknown')}, {my_location.get('country', 'Unknown')}",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(distances) > 1:
        z = np.polyfit(distances, rtts, 1)
        p = np.poly1d(z)
        plt.plot(
            sorted(distances),
            p(sorted(distances)),
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}",
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")
    plt.close()


def run(
    file: Optional[Path] = ASSIGNMENT_1_PATH / "results" / "ip_addresses.txt",
    output: Path = ASSIGNMENT_1_PATH / "results" / "distance_vs_rtt.pdf",
    csv_output: Path = ASSIGNMENT_1_PATH / "results" / "ping_results.csv",
    count: int = 4,
    timeout: int = 3,
    limit: Optional[int] = None,
    delay: float = 1.0,
):
    """Main execution function.

    Fetches IP addresses from iperf3serverlist.net API or file, pings each host,
    geolocates them, calculates distances, and generates plots.

    Args:
        file: Optional path to a file containing IP addresses (one per line).
        output: Path to save the output PDF plot. Defaults to 'distance_vs_rtt.pdf'.
        csv_output: Path to save the CSV file with ping results. Defaults to 'ping_results.csv'.
        count: Number of ping packets to send to each host. Defaults to 4.
        timeout: Timeout in seconds for each ping. Defaults to 3.
        limit: Optional limit on the number of IPs to process. Defaults to None (no limit).
        delay: Delay in seconds between processing each IP to avoid rate limiting. Defaults to 1.0.
    """

    # Get current location
    logger.info("Getting your current location...")
    my_location = get_my_location()
    if not my_location:
        logger.error("Failed to determine your location. Cannot proceed.")
        return

    logger.info(
        f"Your location: {my_location['city']}, {my_location['country']} ({my_location['lat']}, {my_location['lon']})"
    )

    # Get IP list
    if file:
        logger.info(f"Reading IP addresses from {file}...")
        ip_list = read_ips_from_file(file)
    else:
        url = "https://iperf3serverlist.net/api/servers"
        logger.info(f"Fetching IP list from {url}...")
        ip_list = fetch_ip_list(url)

    if limit:
        ip_list = ip_list[:limit]

    logger.info(f"Found {len(ip_list)} unique IPs to process")

    results: list[dict[str, str | dict[str, float] | dict[str, str | float] | None]] = []

    for idx, ip in enumerate(ip_list, 1):
        logger.info(f"[{idx}/{len(ip_list)}] Processing {ip}...")
        ping_stats = ping_host(ip, count=count, timeout=timeout)

        if not ping_stats:
            logger.warning(f"Unable to ping {ip}, skipping geolocation.")
            results.append({"ip": ip, "ping": None, "geo": None})
            continue

        geo = geolocate_ip(ip)
        if geo:
            assert isinstance(my_location["lat"], float)
            assert isinstance(my_location["lon"], float)
            assert isinstance(geo["lat"], float)
            assert isinstance(geo["lon"], float)
            distance = haversine_distance(
                my_location["lat"], my_location["lon"], geo["lat"], geo["lon"]
            )
            logger.info(
                f"   → Ping: {ping_stats['avg']:.2f} ms (min: {ping_stats['min']:.2f}, max: {ping_stats['max']:.2f})"
            )
            logger.info(
                f"   → Location: {geo['city']}, {geo['country']} - Distance: {distance:.2f} km"
            )
        else:
            logger.warning(f"   → Unable to geolocate {ip}")

        results.append({"ip": ip, "ping": ping_stats, "geo": geo})

        # Rate limiting
        if idx < len(ip_list):
            time.sleep(delay)

    # Save results to CSV
    logger.info("\n=== SAVING RESULTS TO CSV ===")
    with open(csv_output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["IP Address", "Min RTT (ms)", "Avg RTT (ms)", "Max RTT (ms)", "Status"])

        for result in results:
            ip = result["ip"]
            ping_stats = result.get("ping")

            if ping_stats:
                writer.writerow(
                    [
                        ip,
                        f"{ping_stats['min']:.2f}",
                        f"{ping_stats['avg']:.2f}",
                        f"{ping_stats['max']:.2f}",
                        "Success",
                    ]
                )
            else:
                writer.writerow([ip, "N/A", "N/A", "N/A", "Failed/Skipped"])

    logger.info(f"Results saved to {csv_output}")

    # Generate plot
    logger.info("\n=== GENERATING PLOT ===")
    valid_results = [r for r in results if r.get("ping") and r.get("geo")]
    logger.info(f"Valid data points: {len(valid_results)}/{len(results)}")

    if valid_results:
        plot_distance_vs_rtt(results, my_location, output)
    else:
        logger.warning("No valid data points to plot")

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total IPs processed: {len(results)}")
    logger.info(f"Successful pings: {len([r for r in results if r.get('ping')])}")
    logger.info(f"Failed pings: {len([r for r in results if not r.get('ping')])}")
    logger.info(f"Successful geolocations: {len([r for r in results if r.get('geo')])}")


if __name__ == "__main__":
    tyro.cli(run)
