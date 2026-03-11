"""
Utility script to check available TCP congestion control algorithms on the system.

Usage:
    python -m cs536.assignment_3.check_cc_algorithms
"""

import subprocess
import socket
from loguru import logger


def check_available_algorithms():
    """Check which TCP congestion control algorithms are available."""
    
    logger.info("="*60)
    logger.info("TCP Congestion Control Algorithms - System Check")
    logger.info("="*60)
    
    # Method 1: Check via sysctl
    logger.info("\n1. Available algorithms (via sysctl):")
    try:
        result = subprocess.run(
            ['sysctl', 'net.ipv4.tcp_available_congestion_control'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if '=' in output:
                algos = output.split('=')[1].strip()
                logger.info(f"   {algos}")
                algo_list = algos.split()
                logger.info(f"\n   Found {len(algo_list)} algorithms:")
                for algo in algo_list:
                    logger.info(f"   - {algo}")
        else:
            logger.warning("   Unable to read (may need elevated privileges)")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    # Method 2: Check allowed algorithms
    logger.info("\n2. Allowed algorithms (via sysctl):")
    try:
        result = subprocess.run(
            ['sysctl', 'net.ipv4.tcp_allowed_congestion_control'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if '=' in output:
                algos = output.split('=')[1].strip()
                logger.info(f"   {algos}")
        else:
            logger.warning("   Unable to read")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    # Method 3: Check current default
    logger.info("\n3. Current default algorithm:")
    try:
        result = subprocess.run(
            ['sysctl', 'net.ipv4.tcp_congestion_control'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if '=' in output:
                default_algo = output.split('=')[1].strip()
                logger.info(f"   {default_algo}")
        else:
            logger.warning("   Unable to read")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    # Method 4: Test setting congestion control on a socket
    logger.info("\n4. Testing socket configuration:")
    common_algos = ['cubic', 'reno', 'custom', 'bbr', 'vegas', 'westwood']
    
    for algo in common_algos:
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            TCP_CONGESTION = getattr(socket, 'TCP_CONGESTION', 13)
            test_sock.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, algo.encode('utf-8'))
            
            # Verify it was set
            actual = test_sock.getsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, 16)
            actual_algo = actual.rstrip(b'\x00').decode('utf-8')
            
            test_sock.close()
            
            if actual_algo.lower() == algo.lower():
                logger.success(f"   ✓ {algo.upper()}: Available and can be set")
            else:
                logger.warning(f"   ⚠ {algo.upper()}: Set but returned '{actual_algo}'")
                
        except OSError as e:
            logger.error(f"   ✗ {algo.upper()}: Not available ({e})")
        except Exception as e:
            logger.warning(f"   ? {algo.upper()}: Error testing ({e})")
    
    logger.info("\n" + "="*60)
    logger.info("\nRecommendations:")
    logger.info("- For testing: Use 'cubic' and 'reno' (most widely available)")
    logger.info("- If BBR is available: Consider testing it for comparison")
    logger.info("- To enable more algorithms: sudo modprobe tcp_<algorithm>")
    logger.info("="*60)


if __name__ == "__main__":
    check_available_algorithms()
