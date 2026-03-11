"""
Quick Start Example for Assignment 3.

Supported invocations:
  python src/cs536/assignment_3/quickstart.py --server <IP_OR_HOST>
  python -m cs536.assignment_3.quickstart --server <IP_OR_HOST>
"""

# ==================== CONFIGURATION ====================
# Defaults can be overridden with CLI flags.
SERVER_IP = "185.93.1.65"
SERVER_PORT = 5201

TEST_DURATION = 10.0
SAMPLING_INTERVAL = 0.5
NUM_RUNS = 3

ALGORITHMS = ["cubic", "reno", "custom"]
# ========================================================

import argparse
import sys
from pathlib import Path
from typing import List


def _bootstrap_imports() -> None:
    """Ensure `cs536` imports work when run as a direct script."""
    try:
        import cs536  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    src_dir = Path(__file__).resolve().parents[2]
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_bootstrap_imports()


def quick_start(
    server: str,
    port: int,
    duration: float,
    interval: float,
    runs: int,
    algorithms: List[str],
    verbose: bool = False,
) -> bool:
    """Run a quick test and analysis."""
    try:
        from cs536.assignment_3.analyze_results import main as analyze_results
        from cs536.assignment_3.run_tests import main as run_tests
    except ModuleNotFoundError as e:
        print(f"[error] Missing dependency: {e}")
        print("Install requirements with: pip install -r src/cs536/assignment_3/requirements.txt")
        return False

    print("=" * 80)
    print(" Congestion Control Algorithm Testing - Quick Start")
    print("=" * 80)
    print(f"\nServer: {server}:{port}")
    print(f"Algorithms: {', '.join([a.upper() for a in algorithms])}")
    print(f"Test duration: {duration}s per run")
    print(f"Runs per algorithm: {runs}")
    print("\n" + "=" * 80 + "\n")

    print("Step 1: Running tests...")
    print("-" * 80)

    try:
        run_tests(
            server=server,
            port=port,
            duration=duration,
            interval=interval,
            runs=runs,
            algorithms=algorithms,
            verbose=verbose,
        )
        print("\n[ok] Tests completed successfully!\n")
    except Exception as e:
        print(f"\n[error] Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify the iperf3 server is running and accessible")
        print("2. Check your network connectivity")
        print("3. Ensure the congestion control algorithms are available:")
        print("   Run: sysctl net.ipv4.tcp_available_congestion_control")
        return False

    print("Step 2: Analyzing results...")
    print("-" * 80)

    try:
        analyze_results()
        print("\n[ok] Analysis completed!\n")
    except Exception as e:
        print(f"\n[error] Analysis failed: {e}")
        return False

    print("=" * 80)
    print(" Quick Start Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check the results/ directory for CSV files and plots")
    print("2. Review the summary statistics table")
    print("3. Add your custom congestion control algorithm and rerun")
    print("\n" + "=" * 80)

    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Assignment 3 tests and analysis in one quick command.",
    )
    parser.add_argument("--server", default=SERVER_IP, help="iperf3 server hostname/IP")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="iperf3 server port")
    parser.add_argument("--duration", type=float, default=TEST_DURATION, help="test duration in seconds")
    parser.add_argument("--interval", type=float, default=SAMPLING_INTERVAL, help="sampling interval in seconds")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="runs per algorithm")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHMS,
        help="space-separated congestion control algorithms",
    )
    parser.add_argument("--yes", action="store_true", help="skip interactive confirmation prompt")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.yes:
        response = input(f"Run tests against {args.server}:{args.port}? (y/n): ")
        if response.lower().strip() != "y":
            print("\nCancelled.")
            sys.exit(0)

    success = quick_start(
        server=args.server,
        port=args.port,
        duration=args.duration,
        interval=args.interval,
        runs=args.runs,
        algorithms=args.algorithms,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)
