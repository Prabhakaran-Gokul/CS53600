# iperf3_tcp_client.py
# A minimal-but-robust iperf3-compatible TCP client (control+data) for public servers.
# Implements: control handshake, JSON param exchange, data streams, timed send, results exchange, clean termination.

import json
import os
import random
import socket
import struct
import string
import threading
import time
from typing import List, Tuple, Optional
from typing import Literal
import tyro
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
import math


# ----- Protocol constants (from Wireshark's iperf3 dissector) -----
TEST_START        = 1
TEST_RUNNING      = 2
TEST_END          = 4
PARAM_EXCHANGE    = 9
CREATE_STREAMS    = 10
SERVER_TERMINATE  = 11
CLIENT_TERMINATE  = 12
EXCHANGE_RESULTS  = 13
DISPLAY_RESULTS   = 14
IPERF_START       = 15
IPERF_DONE        = 16

ACCESS_DENIED     = 0xFF  # -1 in signed byte
SERVER_ERROR      = 0xFE  # -2 in signed byte

COOKIE_SIZE       = 37    # bytes total on the wire (usually 36 chars + terminating NUL)

# ----- Defaults mirroring typical iperf3 behavior -----
DEFAULT_PORT = 5201
DEFAULT_PACING_TIMER_MS = 1000

def recv_state_or_json(sock: socket.socket):
    KNOWN_STATES = {
        TEST_START, TEST_RUNNING, TEST_END, PARAM_EXCHANGE, CREATE_STREAMS,
        SERVER_TERMINATE, CLIENT_TERMINATE, EXCHANGE_RESULTS, DISPLAY_RESULTS,
        IPERF_DONE, ACCESS_DENIED, SERVER_ERROR
    }
    # Peek without consuming
    hdr = sock.recv(4, socket.MSG_PEEK)
    if len(hdr) < 1:
        raise ConnectionError("control socket closed")

    first = hdr[0]
    if first in KNOWN_STATES:
        return 0 # 0 is ok, a state
    
    return -1 # -1 is directly json

    '''
    # Otherwise treat as JSON length-prefixed blob
    if len(hdr) < 4:
        raise TimeoutError("need 4 bytes for JSON length prefix")
    (n,) = struct.unpack("!I", hdr)
    if n <= 0 or n > 50_000_000:
        raise RuntimeError(f"unexpected control framing (first bytes={hdr!r}, len={n})")
    return ("json", json_read(sock))         # consumes 4+n bytes
    '''

def set_common_sockopts(sock: socket.socket, *, nodelay=True, keepalive=True, timeout_sec=10.0):
    sock.settimeout(timeout_sec)
    if nodelay:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    if keepalive:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass

def readn(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise TimeoutError/ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        buf += chunk
    return bytes(buf)

def send_state(sock: socket.socket, state_code: int):
    sock.sendall(struct.pack("!B", state_code & 0xFF))

def recv_state(sock: socket.socket) -> int:
    b = readn(sock, 1)[0]
    # Interpret 0xFF and 0xFE as ACCESS_DENIED / SERVER_ERROR
    if b == ACCESS_DENIED:
        return -1
    if b == SERVER_ERROR:
        return -2
    return b

def json_write(sock: socket.socket, obj: dict):
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

def json_read(sock: socket.socket) -> dict:
    (length,) = struct.unpack("!I", readn(sock, 4))
    data = readn(sock, length)
    return json.loads(data.decode("utf-8", errors="replace"))

def generate_cookie() -> bytes:
    # Produce 36 printable characters, then a trailing NUL to total 37 bytes.
    # iperf3 itself generates opaque-looking ASCII; server expects 37 bytes on the wire.
    s = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(36))
    return s.encode("ascii") + b"\x00"

class DataSender(threading.Thread):
    def __init__(self, sock: socket.socket, cookie: bytes, payload: bytes, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.sock = sock
        self.cookie = cookie
        self.payload = payload
        self.stop_event = stop_event
        self.bytes_sent = 0
        self.error: Optional[Exception] = None

    def run(self):
        try:
            # Each data connection must send the cookie first
            self.sock.sendall(self.cookie)
            # Then send payload blocks until told to stop
            while not self.stop_event.is_set():
                self.sock.sendall(self.payload)
                self.bytes_sent += len(self.payload)
        except Exception as e:
            self.error = e
        finally:
            try:
                self.sock.shutdown(socket.SHUT_WR)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass

def resolve_target(host: str, port: int) -> Tuple[str, int]:
    # Let the OS do the right thing (IPv4/IPv6), but we return the host as given.
    return (host, port)

#def run_iperf3_tcp_client(server: str, port: int, duration: int, parallel: int, blksize: int,
#                          omit: int, connect_timeout: float, read_timeout: float, verbose: bool) -> int:
def run_iperf3_tcp_client(server: str, port: int, duration: int, connect_timeout: float, verbose: bool) -> int:

    ## some constants: 


    cookie = generate_cookie()

    # Prepare control connection
    ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl.settimeout(connect_timeout)
    try:
        ctrl.connect(resolve_target(server, port))
    except Exception as e:
        print(f"[control] connect failed: {e}")
        return 1
    #set_common_sockopts(ctrl, timeout_sec=read_timeout)
    set_common_sockopts(ctrl)

    # Send cookie immediately on the control channel
    try:
        ctrl.sendall(cookie)
    except Exception as e:
        print(f"[control] failed to send cookie: {e}")
        ctrl.close()
        return 1

    if verbose:
        print(f"[control] Connected to {server}:{port}, cookie len={len(cookie)}")

    # Helper to await a specific state from server with basic validation
    def await_state(expected: List[int], context: str) -> int:
        st = recv_state(ctrl)
        if verbose:
            print(f"[control] state {st} ({context})")
        if st in (-1, -2):
            if st == -1:
                raise RuntimeError("ACCESS_DENIED (server busy or rejected)")
            else:
                raise RuntimeError("SERVER_ERROR (server-side failure)")
        if expected and st not in expected:
            # Some servers may send intermediate states (TEST_START, etc.). Accept if reasonable.
            # We'll not be overly strict; just inform on mismatch.
            if verbose:
                print(f"[warn] unexpected state {st}, expected one of {expected}")
        return st

    try:
        # Expect server to move to PARAM_EXCHANGE after we connect
        st = await_state([PARAM_EXCHANGE], "await PARAM_EXCHANGE")
        if st != PARAM_EXCHANGE:
            # Some servers send IPERF_START then PARAM_EXCHANGE. If so, read one more state.
            if st == IPERF_START:
                st = await_state([PARAM_EXCHANGE], "await PARAM_EXCHANGE (after IPERF_START)")
            else:
                # Still try sending params; server will error if wrong.
                if verbose:
                    print("[warn] proceeding to send parameters despite unexpected state")

        # Send parameter JSON
        params = {
            "tcp": True,
            #"omit": max(0, int(omit)),
            "time": max(1, int(duration)),
            "num": 0,
            "blockcount": 0,
            #"parallel": max(1, int(parallel)),
            #"len": max(1024, int(blksize)),
            "pacing_timer": DEFAULT_PACING_TIMER_MS,
            #"client_version": "3.18+ (custom-python)",  # purely informational
        }
        if verbose:
            print(f"[control] send_parameters: {params}")
        json_write(ctrl, params)

        # Next state: CREATE_STREAMS
        st = await_state([CREATE_STREAMS], "await CREATE_STREAMS")

        # Open N data connections; each must send the cookie first (handled by worker)
        data_threads: List[DataSender] = []
        data_socks: List[socket.socket] = []
        stop_event = threading.Event()
        '''
        payload = os.urandom(blksize)  # iperf3 can send zeroes or random; servers don't inspect payload
        for i in range(parallel):
            ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ds.settimeout(connect_timeout)
            ds.connect(resolve_target(server, port))
            set_common_sockopts(ds, timeout_sec=read_timeout)
            sender = DataSender(ds, cookie=cookie, payload=payload, stop_event=stop_event)
            data_threads.append(sender)
            data_socks.append(ds)
            
        if verbose:
            print(f"[data] {parallel} TCP data stream(s) connected")
        '''
        payload = os.urandom(131072)  ## default block size 128 kb
        ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ds.settimeout(connect_timeout)
        ds.connect(resolve_target(server, port))
        set_common_sockopts(ds)
        sender = DataSender(ds, cookie=cookie, payload=payload, stop_event=stop_event)
        data_threads.append(sender)
        #data_socks.append(ds)            

        if verbose:
            print(f"[data] 1 TCP data stream(s) connected")

        sender.start()
        # After server is happy with streams, it will progress through TEST_START → TEST_RUNNING
        st = await_state([TEST_START, TEST_RUNNING], "await TEST_START/TEST_RUNNING")

        if verbose:
            print(f"[info] finished await")

        # Start sending workers once we know the test is starting/running
        '''
        for t in data_threads:
            t.start()
        '''



        # Keep reading states; some servers emit TEST_START then TEST_RUNNING
        if st == TEST_START:
            st = await_state([TEST_RUNNING], "await TEST_RUNNING")

        # Send for the configured duration
        if verbose:
            print(f"[data] transmitting for {duration}s ...")
        t0 = time.time()
        while time.time() - t0 < duration:
            # Also poll control channel in case of early termination or server errors
            ctrl.settimeout(0.0)
            try:
                maybe = ctrl.recv(1, socket.MSG_PEEK)
                if maybe:
                    st2 = recv_state(ctrl)
                    if verbose:
                        print(f"[control] mid-run state {st2}")
                    if st2 in (SERVER_TERMINATE, CLIENT_TERMINATE, EXCHANGE_RESULTS, DISPLAY_RESULTS, IPERF_DONE, ACCESS_DENIED, SERVER_ERROR):
                        break
                else:
                    time.sleep(0.05)
            except (BlockingIOError, TimeoutError):
                time.sleep(0.05)
            #finally:
                ctrl.settimeout(8.0)

        # Stop the data senders and send TEST_END
        stop_event.set()
        sender.join(timeout=5.0)
        '''
        for t in data_threads:
            t.join(timeout=5.0)
        '''
            
        # Aggregate bytes sent
        #total_bytes = sum(t.bytes_sent for t in data_threads)
        total_bytes = sender.bytes_sent
        if verbose:
            print(f"[data] total bytes sent: {total_bytes}")

        send_state(ctrl, TEST_END)
        if verbose:
            print("[control] sent TEST_END")

        # Expect EXCHANGE_RESULTS from server
        st = await_state([EXCHANGE_RESULTS], "await EXCHANGE_RESULTS")

        # Send a minimal results JSON
        # Note: iperf3's own client includes CPU/jitter fields; server is tolerant to a small subset.
        results = {
            "cpu_util_total": 0.0,
            "cpu_util_user": 0.0,
            "cpu_util_system": 0.0,
            "sender_has_retransmits": 0,
            "streams": [
                {"id": 1, "bytes":sender.bytes_sent, "retransmits": -1, "jitter": 0, "errors": 0, "packets": 0}
            ]
            
            #"streams": [
            #    {"id": i + 1, "bytes": data_threads[i].bytes_sent, "retransmits": -1, "jitter": 0, "errors": 0, "packets": 0}
            #    for i in range(len(data_threads))
            #],
        }
        if verbose:
            print(f"[control] send_results: {results}")
        json_write(ctrl, results)


        kind = recv_state_or_json(ctrl)

        if kind == 0:
            st = await_state([DISPLAY_RESULTS, IPERF_DONE], "await DISPLAY_RESULTS/IPERF_DONE")
            # Next state: DISPLAY_RESULTS (typically), then we can send IPERF_DONE.
            if st == DISPLAY_RESULTS:
                # Some servers also send a second JSON (server_output_json)
                try:
                    srv_json2 = json_read(ctrl)
                    if verbose:
                        print(f"[server-results-2] {srv_json2}")
                except Exception:
                    pass

        elif kind == "json":
                        # Some servers then go to DISPLAY_RESULTS automatically.
            try:
                # First results blob
                srv_json1 = json_read(ctrl)
                if verbose:
                    print(f"[server-results-1] {srv_json1}")
            except Exception:
                pass
            st = await_state([DISPLAY_RESULTS, IPERF_DONE], "await DISPLAY_RESULTS/IPERF_DONE")


        # Tell server we're done and close
        send_state(ctrl, IPERF_DONE)
        if verbose:
            print("[control] sent IPERF_DONE")

        ctrl.close()
        print("[ok] Test completed successfully.")
        # Print a quick local summary
        mbps = (total_bytes * 8) / duration / 1e6 if duration > 0 else 0.0
        print(f"[summary] sent {total_bytes} bytes in {duration}s ≈ {mbps:.2f} Mbit/s (client-side)")
        return 0

    except (TimeoutError, ConnectionError, OSError, RuntimeError) as e:
        print(f"[error] {e}")
        try:
            send_state(ctrl, CLIENT_TERMINATE)
        except Exception:
            pass
        try:
            ctrl.close()
        except Exception:
            pass
        return 2

def _get_bytes_acked_linux(sock: socket.socket) -> int:
    """
    Read tcpi_bytes_acked from TCP_INFO via getsockopt on Linux.

    We use a defensive approach: request a reasonably large buffer and scan a few
    candidate offsets for an unsigned 64-bit counter. This is robust across
    minor kernel layout variations.

    Returns:
        int: Current bytes_acked on this socket (monotonic).

    Raises:
        OSError if TCP_INFO is not available or bytes_acked cannot be located.
    """
    TCP_INFO = getattr(socket, "TCP_INFO", None)
    if TCP_INFO is None:
        raise OSError("TCP_INFO not supported on this platform")

    buf = sock.getsockopt(socket.IPPROTO_TCP, TCP_INFO, 512)
    (bytes_acked,) = struct.unpack_from("Q", buf, 120)
    return bytes_acked

    #return int(best_val)


@dataclass
class Sample:
    t: float
    bytes_acked: int


def sample_goodput_bytes_acked(sock: socket.socket, sample_interval: float, duration: float) -> List[Sample]:
    """
    Periodically sample tcpi_bytes_acked for 'duration' seconds at 'sample_interval' frequency.
    Returns a list of time-stamped samples.

    We align the first sample at t=0 (baseline), then collect ~floor(duration / interval) additional samples.
    """
    t0 = time.time()
    samples: List[Sample] = []

    # Initial baseline (t=0)
    try:
        b0 = _get_bytes_acked_linux(sock)
    except OSError as e:
        raise OSError(f"TCP_INFO sampling failed: {e}. This feature requires Linux.") from e
    samples.append(Sample(t=0.0, bytes_acked=b0))

    # Periodic samples
    next_t = t0 + sample_interval
    while True:
        now = time.time()
        remaining = (t0 + duration) - now
        if remaining <= 0:
            break
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue
        # Take a sample
        try:
            bi = _get_bytes_acked_linux(sock)
        except OSError as e:
            # Propagate to caller; better to fail fast than silently degrade
            raise
        t_rel = now - t0
        samples.append(Sample(t=t_rel, bytes_acked=bi))
        next_t += sample_interval

    # Final boundary sample at t=duration (if we didn't land exactly on it)
    last_t = samples[-1].t if samples else 0.0
    if duration - last_t > (sample_interval / 4.0):
        bi = _get_bytes_acked_linux(sock)
        samples.append(Sample(t=duration, bytes_acked=bi))

    return samples


# ---------------- iperf3 one-stream runner with TCP_INFO sampling ----------------

def run_one_destination_with_sampling(host: str, port: int, duration: float, interval: float, verbose: bool = False
                                     ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    This is the one to use for 1c) where results are returned by interval
    Connect to the iperf3 server (control channel), create 1 TCP data stream,
    send for 'duration' seconds, and in parallel sample bytes_acked at 'interval'.
    Returns:
        df: DataFrame with columns [t_mid, goodput_bps, destination]
        stats: dict with min/median/avg/p95 (bits/s)
    """
    # ----- control connection -----
    cookie = generate_cookie()
    ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl.settimeout(8.0)
    ctrl.connect(resolve_target(host, port))
    set_common_sockopts(ctrl)
    ctrl.sendall(cookie)
    if verbose:
        print(f"[control] connected to {host}:{port}, cookie len={len(cookie)}")

    def await_state(expected: List[int], context: str) -> int:
        st = recv_state(ctrl)
        if verbose:
            print(f"[control] state {st} ({context})")
        if st in (-1, -2):
            raise RuntimeError("ACCESS_DENIED" if st == -1 else "SERVER_ERROR")
        return st

    st = await_state([PARAM_EXCHANGE], "await PARAM_EXCHANGE")
    if st != PARAM_EXCHANGE and verbose:
        print("[warn] unexpected state; proceeding to send parameters")

    params = {
        "tcp": True,
        "time": max(1, int(math.ceil(duration))),
        "num": 0,
        "blockcount": 0,
        "pacing_timer": DEFAULT_PACING_TIMER_MS,
    }
    if verbose:
        print(f"[control] send_parameters: {params}")
    json_write(ctrl, params)

    st = await_state([CREATE_STREAMS], "await CREATE_STREAMS")

    # ----- data stream -----
    payload = b"\x00" * 131072  # 128 KiB blocks (payload contents are irrelevant to server)
    ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ds.settimeout(8.0)
    ds.connect(resolve_target(host, port))
    set_common_sockopts(ds)

    stop_event = __import__("threading").Event()
    sender = DataSender(ds, cookie=cookie, payload=payload, stop_event=stop_event)
    sender.start()
    if verbose:
        print("[data] started 1 TCP data stream")

    # ----- test start/run -----
    st = await_state([TEST_START, TEST_RUNNING], "await TEST_START/TEST_RUNNING")
    if st == TEST_START:
        st = await_state([TEST_RUNNING], "await TEST_RUNNING")

    # ----- sampling loop: bytes_acked every 'interval' seconds -----
    samples = sample_goodput_bytes_acked(sender.sock, sample_interval=interval, duration=duration)

    # ----- stop sender and close -----
    stop_event.set()
    sender.join(timeout=5.0)

    send_state(ctrl, TEST_END)
    st = await_state([EXCHANGE_RESULTS], "await EXCHANGE_RESULTS")
    # Send minimal results back (iperf3 server is tolerant)
    results = {
        "cpu_util_total": 0.0,
        "cpu_util_user": 0.0,
        "cpu_util_system": 0.0,
        "sender_has_retransmits": 0,
        "streams": [{"id": 1, "bytes": sender.bytes_sent, "retransmits": -1, "jitter": 0, "errors": 0, "packets": 0}],
    }
    json_write(ctrl, results)

    # Server may send DISPLAY_RESULTS and/or JSON; finalize with IPERF_DONE
    try:
        st2 = recv_state(ctrl)
        if verbose:
            print(f"[control] post-results state {st2}")
    except Exception:
        pass
    send_state(ctrl, IPERF_DONE)
    try:
        ctrl.close()
    except Exception:
        pass

    # ----- compute per-interval goodput from samples -----
    # Use consecutive deltas: goodput_bps = (Δbytes_acked * 8) / Δt
    rows = []
    for i in range(1, len(samples)):
        t_prev, b_prev = samples[i - 1].t, samples[i - 1].bytes_acked
        t_curr, b_curr = samples[i].t, samples[i].bytes_acked
        #dt = max(1e-9, t_curr - t_prev)
        db = max(0, b_curr - b_prev)  # ensure non-negative (monotonic guard)
        bps = (db * 8.0) / interval
        t_mid = 0.5 * (t_prev + t_curr)
        rows.append({"t_mid": t_mid, "goodput_bps": bps})

    df = pd.DataFrame(rows)
    df["destination"] = f"{host}:{port}"

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

    return df, stats


def main(    server: str,
    port: int = DEFAULT_PORT,
    duration: int = 60,
    connect_timeout: float = 8.0,
    verbose: bool = False,):


    random.seed()  # cookie
    rc = run_iperf3_tcp_client(
        server=server,
        port=port,
        duration=duration,
        #parallel=args.parallel,
        #blksize=args.blksize,
        #omit=args.omit,
        connect_timeout=connect_timeout,
        #read_timeout=args.read_timeout,
        verbose=verbose,
    )
    raise SystemExit(rc)

if __name__ == "__main__":
    tyro.cli(main)

    '''
    Example: PYTHONPATH=src python3 -m cs536.assignment_2.iperf3_tcp_client --server 185.93.1.65 --duration 15 --verbose
    '''