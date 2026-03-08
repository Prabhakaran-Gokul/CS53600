"""
Quick Start Example for Assignment 3

This script demonstrates basic usage of the congestion control testing framework.
Edit the SERVER_IP variable and run this script to get started.
"""

# ==================== CONFIGURATION ====================
# Edit this to point to your iperf3 server
SERVER_IP = "185.93.1.65"  # Example public iperf3 server
SERVER_PORT = 5201

# Test parameters
TEST_DURATION = 10  # seconds
SAMPLING_INTERVAL = 0.5  # seconds
NUM_RUNS = 3  # runs per algorithm

# Algorithms to test
ALGORITHMS = ['cubic', 'reno', 'custom']  # Add 'custom' if available
# ========================================================

import sys
from pathlib import Path

# Add src to path
# workspace_root = Path(__file__).parent.parent.parent.parent
# sys.path.insert(0, str(workspace_root / "src"))

from cs536.assignment_3.run_tests import main as run_tests
from cs536.assignment_3.analyze_results import main as analyze_results


def quick_start():
    """Run a quick test and analysis."""
    
    print("="*80)
    print(" Congestion Control Algorithm Testing - Quick Start")
    print("="*80)
    print(f"\nServer: {SERVER_IP}:{SERVER_PORT}")
    print(f"Algorithms: {', '.join([a.upper() for a in ALGORITHMS])}")
    print(f"Test duration: {TEST_DURATION}s per run")
    print(f"Runs per algorithm: {NUM_RUNS}")
    print("\n" + "="*80 + "\n")
    
    # Step 1: Run tests
    print("Step 1: Running tests...")
    print("-" * 80)
    
    try:
        run_tests(
            server=SERVER_IP,
            port=SERVER_PORT,
            duration=TEST_DURATION,
            interval=SAMPLING_INTERVAL,
            runs=NUM_RUNS,
            algorithms=ALGORITHMS,
            verbose=True
        )
        print("\n✓ Tests completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify the iperf3 server is running and accessible")
        print("2. Check your network connectivity")
        print("3. Ensure the congestion control algorithms are available:")
        print("   Run: sysctl net.ipv4.tcp_available_congestion_control")
        return False
    
    # Step 2: Analyze results
    print("Step 2: Analyzing results...")
    print("-" * 80)
    
    try:
        analyze_results()
        print("\n✓ Analysis completed!\n")
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        return False
    
    # Summary
    print("="*80)
    print(" Quick Start Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check the results/ directory for CSV files and plots")
    print("2. Review the summary statistics table")
    print("3. Add your custom congestion control algorithm and rerun")
    print("\n" + "="*80)
    
    return True


if __name__ == "__main__":
    print("\nIMPORTANT: Edit SERVER_IP in this script before running!\n")
    
    # Ask user to confirm
    response = input(f"Run tests against {SERVER_IP}:{SERVER_PORT}? (y/n): ")
    if response.lower().strip() == 'y':
        success = quick_start()
        sys.exit(0 if success else 1)
    else:
        print("\nCancelled. Update SERVER_IP and run again.")
        sys.exit(0)
