"""
Analysis and Visualization for Congestion Control Algorithm Comparison

This script analyzes the results from run_tests.py and generates comparison plots.

Usage:
    PYTHONPATH=src python -m cs536.assignment_3.analyze_results
"""

import tyro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger
from cs536.assignment_3 import ASSIGNMENT_3_PATH


def _time_bin(series: pd.Series, decimals: int = 2) -> pd.Series:
    """Bin floating timestamps to reduce cross-run alignment noise."""
    return series.astype(float).round(decimals)


def load_results(results_dir: Path) -> dict:
    """Load all result files from the results directory."""
    
    results = {}
    
    # Load goodput data
    goodput_file = results_dir / "goodput_all_runs.csv"
    if goodput_file.exists():
        results['goodput'] = pd.read_csv(goodput_file)
        logger.success(f"✓ Loaded goodput data: {len(results['goodput'])} samples")
    else:
        logger.warning(f"⚠ Goodput file not found: {goodput_file}")
        results['goodput'] = pd.DataFrame()
    
    # Load TCP stats
    tcp_stats_file = results_dir / "tcp_stats_all_runs.csv"
    if tcp_stats_file.exists():
        results['tcp_stats'] = pd.read_csv(tcp_stats_file)
        logger.success(f"✓ Loaded TCP stats: {len(results['tcp_stats'])} samples")
    else:
        logger.warning(f"⚠ TCP stats file not found: {tcp_stats_file}")
        results['tcp_stats'] = pd.DataFrame()
    
    # Load summary statistics
    summary_file = results_dir / "summary_statistics.csv"
    if summary_file.exists():
        results['summary'] = pd.read_csv(summary_file)
        logger.success(f"✓ Loaded summary statistics: {len(results['summary'])} entries")
    else:
        logger.warning(f"⚠ Summary file not found: {summary_file}")
        results['summary'] = pd.DataFrame()
    
    return results


def plot_throughput_comparison(goodput_df: pd.DataFrame, output_dir: Path):
    """Plot throughput comparison across algorithms."""
    
    if goodput_df.empty:
        logger.warning("⚠ No goodput data to plot")
        return
    
    # Convert to Mbps
    goodput_df['throughput_mbps'] = goodput_df['goodput_bps'] / 1e6
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series for each algorithm
    ax = axes[0]
    algorithms = goodput_df['cc_algorithm'].unique()
    
    for algo in algorithms:
        algo_data = goodput_df[goodput_df['cc_algorithm'] == algo]
        
        # Plot each run with some transparency
        for run_id in algo_data['run_id'].unique():
            run_data = algo_data[algo_data['run_id'] == run_id]
            ax.plot(run_data['t_mid'], run_data['throughput_mbps'], 
                   alpha=0.3, linewidth=1)
        
        # Plot average across runs
        avg_by_time = (
            algo_data.assign(t_bin=_time_bin(algo_data['t_mid']))
            .groupby('t_bin')['throughput_mbps']
            .mean()
        )
        ax.plot(avg_by_time.index, avg_by_time.values, 
               label=algo.upper(), linewidth=2)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Throughput (Mbps)', fontsize=12)
    ax.set_title('Throughput Over Time by Congestion Control Algorithm', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax = axes[1]
    
    # Prepare data for box plot
    box_data = []
    labels = []
    for algo in algorithms:
        algo_data = goodput_df[goodput_df['cc_algorithm'] == algo]['throughput_mbps']
        box_data.append(algo_data)
        labels.append(algo.upper())
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Throughput (Mbps)', fontsize=12)
    ax.set_title('Throughput Distribution by Congestion Control Algorithm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "throughput_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"✓ Saved throughput comparison: {output_file}")
    plt.close()


def plot_rtt_comparison(tcp_stats_df: pd.DataFrame, output_dir: Path):
    """Plot RTT comparison across algorithms."""
    
    if tcp_stats_df.empty or 'rtt_us' not in tcp_stats_df.columns:
        logger.warning("⚠ No RTT data to plot")
        return
    
    # Convert to milliseconds
    tcp_stats_df['rtt_ms'] = tcp_stats_df['rtt_us'] / 1000.0
    
    # Filter out zero/invalid RTT values
    tcp_stats_df = tcp_stats_df[tcp_stats_df['rtt_ms'] > 0]
    
    if tcp_stats_df.empty:
        logger.warning("⚠ No valid RTT data after filtering")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: RTT over time
    ax = axes[0]
    algorithms = tcp_stats_df['cc_algorithm'].unique()
    
    for algo in algorithms:
        algo_data = tcp_stats_df[tcp_stats_df['cc_algorithm'] == algo]
        
        # Plot average RTT across runs
        avg_by_time = (
            algo_data.assign(t_bin=_time_bin(algo_data['ts']))
            .groupby('t_bin')['rtt_ms']
            .mean()
        )
        ax.plot(avg_by_time.index, avg_by_time.values, 
               label=algo.upper(), linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('RTT (ms)', fontsize=12)
    ax.set_title('Round-Trip Time Over Time by Congestion Control Algorithm', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RTT distribution
    ax = axes[1]
    
    box_data = []
    labels = []
    for algo in algorithms:
        algo_data = tcp_stats_df[tcp_stats_df['cc_algorithm'] == algo]['rtt_ms']
        box_data.append(algo_data)
        labels.append(algo.upper())
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('RTT (ms)', fontsize=12)
    ax.set_title('RTT Distribution by Congestion Control Algorithm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "rtt_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"✓ Saved RTT comparison: {output_file}")
    plt.close()


def plot_loss_comparison(tcp_stats_df: pd.DataFrame, output_dir: Path):
    """Plot packet loss comparison across algorithms."""
    
    if tcp_stats_df.empty:
        logger.warning("⚠ No TCP stats data to plot loss")
        return
    
    # Calculate loss rate from retransmissions
    algorithms = tcp_stats_df['cc_algorithm'].unique()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Total retransmissions over time
    ax = axes[0]
    
    for algo in algorithms:
        algo_data = tcp_stats_df[tcp_stats_df['cc_algorithm'] == algo]
        
        if 'total_retrans' in algo_data.columns:
            avg_by_time = (
                algo_data.assign(t_bin=_time_bin(algo_data['ts']))
                .groupby('t_bin')['total_retrans']
                .mean()
            )
            ax.plot(avg_by_time.index, avg_by_time.values, 
                   label=algo.upper(), linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Retransmissions', fontsize=12)
    ax.set_title('Packet Retransmissions Over Time by Congestion Control Algorithm', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss comparison bar chart
    ax = axes[1]
    
    loss_summary = []
    for algo in algorithms:
        algo_data = tcp_stats_df[tcp_stats_df['cc_algorithm'] == algo]
        
        # Get final values per run
        final_losses = []
        for run_id in algo_data['run_id'].unique():
            run_data = algo_data[algo_data['run_id'] == run_id].sort_values('ts')
            if 'total_retrans' in run_data.columns and len(run_data) > 0:
                final_loss = run_data['total_retrans'].iloc[-1]
                final_losses.append(final_loss)
        
        if final_losses:
            loss_summary.append({
                'algorithm': algo.upper(),
                'avg_loss': np.mean(final_losses),
                'std_loss': np.std(final_losses)
            })
    
    if loss_summary:
        loss_df = pd.DataFrame(loss_summary)
        x = np.arange(len(loss_df))
        
        bars = ax.bar(x, loss_df['avg_loss'], yerr=loss_df['std_loss'],
                     capsize=5, alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(loss_df))))
        
        ax.set_xticks(x)
        ax.set_xticklabels(loss_df['algorithm'])
        ax.set_ylabel('Average Total Retransmissions', fontsize=12)
        ax.set_title('Average Packet Loss by Congestion Control Algorithm', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "loss_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"✓ Saved loss comparison: {output_file}")
    plt.close()


def plot_cwnd_comparison(tcp_stats_df: pd.DataFrame, output_dir: Path):
    """Plot congestion window comparison across algorithms."""
    
    if tcp_stats_df.empty or 'snd_cwnd' not in tcp_stats_df.columns:
        logger.warning("⚠ No congestion window data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    algorithms = tcp_stats_df['cc_algorithm'].unique()
    
    for algo in algorithms:
        algo_data = tcp_stats_df[tcp_stats_df['cc_algorithm'] == algo]
        
        # Plot average cwnd across runs
        avg_by_time = (
            algo_data.assign(t_bin=_time_bin(algo_data['ts']))
            .groupby('t_bin')['snd_cwnd']
            .mean()
        )
        ax.plot(avg_by_time.index, avg_by_time.values, 
               label=algo.upper(), linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Congestion Window (packets)', fontsize=12)
    ax.set_title('Congestion Window Over Time by Algorithm', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "cwnd_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"✓ Saved congestion window comparison: {output_file}")
    plt.close()


def generate_summary_table(summary_df: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive summary table."""
    
    if summary_df.empty:
        logger.warning("⚠ No summary data available")
        return
    
    # Group by algorithm and compute statistics
    summary_stats = summary_df.groupby('algorithm').agg({
        'avg': ['mean', 'std'],
        'median': 'mean',
        'min': 'mean',
        'p95': 'mean'
    }).round(2)
    
    # Convert to Mbps
    for col in summary_stats.columns:
        summary_stats[col] = summary_stats[col] / 1e6
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.rename(columns={
        'avg_mean': 'Avg Throughput (Mbps)',
        'avg_std': 'Std Dev (Mbps)',
        'median_mean': 'Median Throughput (Mbps)',
        'min_mean': 'Min Throughput (Mbps)',
        'p95_mean': 'P95 Throughput (Mbps)'
    })
    
    # Reset index to make algorithm a column
    summary_stats = summary_stats.reset_index()
    summary_stats['algorithm'] = summary_stats['algorithm'].str.upper()
    
    # Save to CSV
    output_file = output_dir / "analysis_summary_table.csv"
    summary_stats.to_csv(output_file, index=False)
    logger.success(f"✓ Saved summary table: {output_file}")
    
    # Print to console
    logger.info("\n" + "="*90)
    logger.info("SUMMARY TABLE: Throughput Comparison (Mbps)")
    logger.info("="*90)
    logger.info("\n" + summary_stats.to_string(index=False))
    logger.info("\n" + "="*90 + "\n")
    
    return summary_stats


def main(
    results_dir: Optional[str] = None,
    generate_plots: bool = True
):
    """
    Analyze congestion control test results and generate visualizations.
    
    Args:
        results_dir: Path to results directory (default: assignment_3/results)
        generate_plots: Whether to generate plots (default: True)
    """
    
    results_path: Path
    if results_dir is None:
        results_path = ASSIGNMENT_3_PATH / "results"
    else:
        results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Error: Results directory not found: {results_path}")
        logger.info("Please run the tests first using: PYTHONPATH=src python -m cs536.assignment_3.run_tests")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("Analyzing Congestion Control Test Results")
    logger.info(f"{'='*60}")
    logger.info(f"Results directory: {results_path}\n")
    
    # Load results
    results = load_results(results_path)
    
    if all(df.empty for df in results.values()):
        logger.error("\nError: No valid results found!")
        return
    
    logger.info("")
    
    # Generate visualizations
    if generate_plots:
        logger.info("Generating plots...")
        
        if not results['goodput'].empty:
            plot_throughput_comparison(results['goodput'], results_path)
        
        if not results['tcp_stats'].empty:
            plot_rtt_comparison(results['tcp_stats'], results_path)
            plot_loss_comparison(results['tcp_stats'], results_path)
            plot_cwnd_comparison(results['tcp_stats'], results_path)
        
        logger.info("")
    
    # Generate summary table
    if not results['summary'].empty:
        generate_summary_table(results['summary'], results_path)
    
    logger.success(f"\n✓ Analysis complete! Results saved to: {results_path}")


if __name__ == "__main__":
    tyro.cli(main)
