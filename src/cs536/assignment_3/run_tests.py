"""
Test Runner for Congestion Control Algorithm Comparison

This script runs multiple tests with different congestion control algorithms
(CUBIC, RENO, and later custom) and collects performance metrics.

Usage:
    PYTHONPATH=src python -m cs536.assignment_3.run_tests --server SERVER_IP --runs 5
"""

import tyro
import time
import pandas as pd
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger
from cs536.assignment_3 import ASSIGNMENT_3_PATH
from cs536.assignment_3.tcp_client_cc import run_with_congestion_control


@dataclass
class TestConfig:
    """Configuration for congestion control tests."""
    server: str
    """Target iperf3 server IP or hostname"""
    port: int = 5201
    """Target server port"""
    duration: float = 10.0
    """Test duration in seconds"""
    interval: float = 0.5
    """Sampling interval in seconds"""
    runs: int = 3
    """Number of test runs per algorithm"""
    algorithms: Optional[List[str]] = None
    """List of congestion control algorithms to test"""
    wait_between_runs: float = 2.0
    """Wait time between consecutive runs (seconds)"""
    verbose: bool = False
    """Enable verbose output"""

    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ['cubic', 'reno']


def run_single_test(
    server: str,
    port: int,
    duration: float,
    interval: float,
    cc_algorithm: str,
    run_id: int,
    verbose: bool = False
) -> tuple:
    """
    Run a single test with specified parameters.
    
    Returns:
        Tuple of (goodput_df, goodput_stats, tcp_stats_df, success)
    """
    try:
        logger.info(f"  Run {run_id + 1}: Testing {cc_algorithm.upper()}...")
        
        goodput_df, goodput_stats, tcp_stats = run_with_congestion_control(
            host=server,
            port=port,
            duration=duration,
            interval=interval,
            cc_algorithm=cc_algorithm,
            verbose=verbose
        )
        
        # Add run ID to dataframes
        goodput_df['run_id'] = run_id
        if not tcp_stats.empty:
            tcp_stats['run_id'] = run_id
        
        logger.success(f"    ✓ Completed - Avg throughput: {goodput_stats['avg'] / 1e6:.2f} Mbps")
        return goodput_df, goodput_stats, tcp_stats, True
        
    except Exception as e:
        logger.error(f"    ✗ Failed: {e}")
        return pd.DataFrame(), {}, pd.DataFrame(), False


def run_all_tests(config: TestConfig) -> dict:
    """
    Run all tests according to configuration.
    
    Returns:
        Dictionary containing all results
    """
    results = {
        'goodput_dataframes': [],
        'tcp_stats_dataframes': [],
        'summary_stats': []
    }
    
    results_dir = ASSIGNMENT_3_PATH / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Ensure algorithms is not None (set in __post_init__)
    algorithms = config.algorithms or ['cubic', 'reno']
    
    logger.info(f"\n{'='*60}")
    logger.info("Starting Congestion Control Algorithm Tests")
    logger.info(f"{'='*60}")
    logger.info(f"Server: {config.server}:{config.port}")
    logger.info(f"Duration: {config.duration}s per test")
    logger.info(f"Algorithms: {', '.join([a.upper() for a in algorithms])}")
    logger.info(f"Runs per algorithm: {config.runs}")
    logger.info(f"{'='*60}\n")
    
    total_tests = len(algorithms) * config.runs
    completed_tests = 0
    failed_tests = 0
    
    for cc_algo in algorithms:
        logger.info(f"\nTesting {cc_algo.upper()} ({config.runs} runs)")
        logger.info("-" * 40)
        
        for run_id in range(config.runs):
            goodput_df, stats, tcp_stats, success = run_single_test(
                server=config.server,
                port=config.port,
                duration=config.duration,
                interval=config.interval,
                cc_algorithm=cc_algo,
                run_id=run_id,
                verbose=config.verbose
            )
            
            if success:
                results['goodput_dataframes'].append(goodput_df)
                results['tcp_stats_dataframes'].append(tcp_stats)
                
                # Store summary statistics
                summary = {
                    'algorithm': cc_algo,
                    'run_id': run_id,
                    'server': f"{config.server}:{config.port}",
                    **stats
                }
                results['summary_stats'].append(summary)
                completed_tests += 1
            else:
                failed_tests += 1
            
            # Wait between runs
            if run_id < config.runs - 1:
                time.sleep(config.wait_between_runs)
        
        # Longer wait between different algorithms
        if cc_algo != algorithms[-1]:
            logger.info("\nWaiting before next algorithm...")
            time.sleep(config.wait_between_runs * 2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Completed: {completed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"{'='*60}\n")
    
    return results


def save_results(results: dict, output_dir: Path):
    """Save all results to CSV files."""
    
    logger.info("Saving results...")
    
    # Combine all goodput dataframes
    if results['goodput_dataframes']:
        goodput_all = pd.concat(results['goodput_dataframes'], ignore_index=True)
        goodput_file = output_dir / "goodput_all_runs.csv"
        goodput_all.to_csv(goodput_file, index=False)
        logger.success(f"  ✓ Saved goodput data: {goodput_file}")
    
    # Combine all TCP stats dataframes
    if results['tcp_stats_dataframes']:
        tcp_stats_all = pd.concat(results['tcp_stats_dataframes'], ignore_index=True)
        tcp_stats_file = output_dir / "tcp_stats_all_runs.csv"
        tcp_stats_all.to_csv(tcp_stats_file, index=False)
        logger.success(f"  ✓ Saved TCP stats: {tcp_stats_file}")
    
    # Save summary statistics
    if results['summary_stats']:
        summary_df = pd.DataFrame(results['summary_stats'])
        summary_file = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.success(f"  ✓ Saved summary stats: {summary_file}")
        
        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("Summary Statistics (Throughput in Mbps)")
        logger.info("="*80)
        
        for algo in summary_df['algorithm'].unique():
            algo_data = summary_df[summary_df['algorithm'] == algo]
            logger.info(f"\n{algo.upper()}:")
            logger.info(f"  Average Throughput: {algo_data['avg'].mean() / 1e6:.2f} Mbps (±{algo_data['avg'].std() / 1e6:.2f})")
            logger.info(f"  Median Throughput:  {algo_data['median'].mean() / 1e6:.2f} Mbps")
            logger.info(f"  Min Throughput:     {algo_data['min'].mean() / 1e6:.2f} Mbps")
            logger.info(f"  P95 Throughput:     {algo_data['p95'].mean() / 1e6:.2f} Mbps")
        
        logger.info("\n" + "="*80)


def main(
    server: str,
    port: int = 5201,
    duration: float = 10.0,
    interval: float = 0.5,
    runs: int = 3,
    algorithms: Optional[List[str]] = None,
    wait_between_runs: float = 2.0,
    verbose: bool = False
):
    """
    Run congestion control algorithm comparison tests.
    
    Args:
        server: Target iperf3 server IP or hostname
        port: Target server port (default: 5201)
        duration: Test duration in seconds (default: 10.0)
        interval: Sampling interval in seconds (default: 0.5)
        runs: Number of test runs per algorithm (default: 3)
        algorithms: List of CC algorithms to test (default: ['cubic', 'reno'])
        wait_between_runs: Wait time between runs in seconds (default: 2.0)
        verbose: Enable verbose output (default: False)
    """
    
    config = TestConfig(
        server=server,
        port=port,
        duration=duration,
        interval=interval,
        runs=runs,
        algorithms=algorithms,
        wait_between_runs=wait_between_runs,
        verbose=verbose
    )
    
    # Run all tests
    results = run_all_tests(config)
    
    # Save results
    output_dir = ASSIGNMENT_3_PATH / "results"
    output_dir.mkdir(exist_ok=True)
    save_results(results, output_dir)
    
    logger.success(f"\n✓ All results saved to: {output_dir}")
    logger.info("\nTo analyze results, run:")
    logger.info("  PYTHONPATH=src python -m cs536.assignment_3.analyze_results")


if __name__ == "__main__":
    tyro.cli(main)
