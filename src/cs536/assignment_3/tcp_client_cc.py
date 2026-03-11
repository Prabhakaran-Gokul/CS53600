"""
TCP Client with Congestion Control Algorithm Selection
This module extends the assignment 2 TCP client to support setting specific congestion control algorithms.
"""

import socket
import threading
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from cs536.assignment_2.iperf3_tcp_client import (
    generate_cookie, resolve_target, set_common_sockopts,
    recv_state, send_state, json_write, DataSender,
    PARAM_EXCHANGE, CREATE_STREAMS, TEST_START, TEST_RUNNING,
    TEST_END, EXCHANGE_RESULTS, IPERF_DONE, DEFAULT_PACING_TIMER_MS,
    sample_goodput_bytes_acked
)
from cs536.assignment_2.tcp_stats import sample_tcp_info


def set_congestion_control(sock: socket.socket, algorithm: str) -> bool:
    """
    Set the TCP congestion control algorithm for a socket.
    
    Args:
        sock: The socket to configure
        algorithm: Congestion control algorithm name (e.g., 'cubic', 'reno', 'bbr')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # TCP_CONGESTION is typically 13 on Linux
        TCP_CONGESTION = getattr(socket, 'TCP_CONGESTION', 13)
        sock.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, algorithm.encode('utf-8'))
        return True
    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to set congestion control to {algorithm}: {e}")
        return False


def get_congestion_control(sock: socket.socket) -> Optional[str]:
    """
    Get the current TCP congestion control algorithm for a socket.
    
    Args:
        sock: The socket to query
    
    Returns:
        The congestion control algorithm name, or None if unavailable
    """
    try:
        TCP_CONGESTION = getattr(socket, 'TCP_CONGESTION', 13)
        algo_bytes = sock.getsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, 16)
        # Remove null bytes and decode
        algo = algo_bytes.rstrip(b'\x00').decode('utf-8')
        return algo
    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to get congestion control: {e}")
        return None


def run_with_congestion_control(
    host: str,
    port: int,
    duration: float,
    interval: float,
    cc_algorithm: str,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Run TCP test with specified congestion control algorithm.
    
    Args:
        host: Target server hostname or IP
        port: Target server port
        duration: Test duration in seconds
        interval: Sampling interval in seconds
        cc_algorithm: Congestion control algorithm ('cubic', 'reno', etc.)
        verbose: Enable verbose logging
    
    Returns:
        Tuple of (goodput_df, goodput_stats, tcp_stats_df)
    """
    # ----- control connection -----
    cookie = generate_cookie()
    ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl.settimeout(8.0)
    ctrl.connect(resolve_target(host, port))
    set_common_sockopts(ctrl)
    ctrl.sendall(cookie)
    
    if verbose:
        logger.info(f"[control] connected to {host}:{port}, cookie len={len(cookie)}")

    def await_state(expected: List[int], context: str) -> int:
        st = recv_state(ctrl)
        if verbose:
            logger.debug(f"[control] state {st} ({context})")
        if st in (-1, -2):
            raise RuntimeError("ACCESS_DENIED" if st == -1 else "SERVER_ERROR")
        return st

    st = await_state([PARAM_EXCHANGE], "await PARAM_EXCHANGE")
    if st != PARAM_EXCHANGE and verbose:
        logger.warning("[warn] unexpected state; proceeding to send parameters")

    params = {
        "tcp": True,
        "time": max(1, int(duration)),
        "num": 0,
        "blockcount": 0,
        "pacing_timer": DEFAULT_PACING_TIMER_MS,
    }
    if verbose:
        logger.debug(f"[control] send_parameters: {params}")
    json_write(ctrl, params)

    st = await_state([CREATE_STREAMS], "await CREATE_STREAMS")

    # ----- data stream with congestion control -----
    payload = b"\x00" * 131072  # 128 KiB blocks
    ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Set congestion control BEFORE connect
    cc_set = set_congestion_control(ds, cc_algorithm)
    if not cc_set:
        raise RuntimeError(f"Failed to set congestion control to '{cc_algorithm}'")
    if verbose:
        logger.info(f"[data] congestion control set to: {cc_algorithm}")
    
    ds.settimeout(8.0)
    ds.connect(resolve_target(host, port))
    set_common_sockopts(ds)
    
    # Verify congestion control was set
    actual_cc = get_congestion_control(ds)
    if verbose:
        logger.debug(f"[data] actual congestion control: {actual_cc}")
    if actual_cc is None:
        raise RuntimeError("Unable to verify active congestion control algorithm")
    if actual_cc.lower() != cc_algorithm.lower():
        raise RuntimeError(f"Requested congestion control '{cc_algorithm}' but socket is using '{actual_cc}'")

    stop_event = threading.Event()
    sender = DataSender(ds, cookie=cookie, payload=payload, stop_event=stop_event)
    sender.start()
    
    if verbose:
        logger.info("[data] started 1 TCP data stream")

    # ----- test start/run -----
    st = await_state([TEST_START, TEST_RUNNING], "await TEST_START/TEST_RUNNING")
    if st == TEST_START:
        st = await_state([TEST_RUNNING], "await TEST_RUNNING")

    # ----- sample TCP stats and goodput concurrently -----
    tcp_stats_holder: dict = {}

    def _tcp_stats_worker():
        try:
            tcp_stats_holder["df"] = sample_tcp_info(
                sock=sender.sock,
                sample_interval=interval,
                duration=duration,
            )
        except Exception as e:
            tcp_stats_holder["error"] = e
            tcp_stats_holder["df"] = pd.DataFrame()

    tcp_thread = threading.Thread(target=_tcp_stats_worker, daemon=True)
    tcp_thread.start()

    # ----- sampling loop: bytes_acked every 'interval' seconds -----
    samples = sample_goodput_bytes_acked(
        sender.sock,
        sample_interval=interval,
        duration=duration,
        is_alive_fn=sender.is_alive,
    )

    # ----- stop sender and close -----
    stop_event.set()
    sender.join(timeout=5.0)
    tcp_thread.join(timeout=2.0)

    if "error" in tcp_stats_holder:
        err = tcp_stats_holder["error"]
        err_name = type(err).__name__
        raise RuntimeError(f"TCP stats sampling failed ({err_name}: {err})")
    
    if sender.error is not None:
        err_name = type(sender.error).__name__
        msg = str(sender.error)
        raise RuntimeError(f"data socket ended early ({err_name}: {msg})")

    send_state(ctrl, TEST_END)
    st = await_state([EXCHANGE_RESULTS], "await EXCHANGE_RESULTS")
    
    # Send minimal results back
    results = {
        "cpu_util_total": 0.0,
        "cpu_util_user": 0.0,
        "cpu_util_system": 0.0,
        "sender_has_retransmits": 0,
        "streams": [{
            "id": 1,
            "bytes": sender.bytes_sent,
            "retransmits": -1,
            "jitter": 0,
            "errors": 0,
            "packets": 0
        }],
    }
    json_write(ctrl, results)

    # Finalize connection
    try:
        st2 = recv_state(ctrl)
        if verbose:
            logger.debug(f"[control] post-results state {st2}")
    except Exception:
        pass
    send_state(ctrl, IPERF_DONE)
    
    try:
        ctrl.close()
    except Exception:
        pass

    # ----- compute per-interval goodput from samples -----
    rows = []
    for i in range(1, len(samples)):
        t_prev, b_prev = samples[i - 1].t, samples[i - 1].bytes_acked
        t_curr, b_curr = samples[i].t, samples[i].bytes_acked
        db = max(0, b_curr - b_prev)
        dt = max(t_curr - t_prev, 1e-9)
        bps = (db * 8.0) / dt
        t_mid = 0.5 * (t_prev + t_curr)
        rows.append({"t_mid": t_mid, "goodput_bps": bps})

    df = pd.DataFrame(rows)
    df["destination"] = f"{host}:{port}"
    df["cc_algorithm"] = cc_algorithm

    if len(df) == 0:
        stats = {"min": 0.0, "median": 0.0, "avg": 0.0, "p95": 0.0}
    else:
        vals = df["goodput_bps"].astype(float).values
        stats = {
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "avg": float(np.mean(vals)),
            "p95": float(np.percentile(vals, 95)),
        }

    tcp_stats = tcp_stats_holder.get("df", pd.DataFrame())
    if not tcp_stats.empty:
        tcp_stats["cc_algorithm"] = cc_algorithm
        tcp_stats["destination"] = f"{host}:{port}"
    
    return df, stats, tcp_stats
