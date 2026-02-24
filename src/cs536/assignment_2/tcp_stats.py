# ===== TCP_INFO helpers (Linux) =====
import sys
import socket 
from dataclasses import dataclass
import time 
from typing import List
import pandas as pd

class TCPInfoError(Exception):
    pass

def _tcpinfo_linux_parse(buf: bytes) -> dict:
    import struct
    def u32(off):
        return struct.unpack_from("I", buf, off)[0]


    def u64(off): 
        return struct.unpack_from('Q', buf, off)[0]


    out = {}
    '''
    rtt_us = struct.unpack_from("I", buf, 68)[0]
    rttvar_us = struct.unpack_from("I", buf, 72)[0]
    snd_cwnd = struct.unpack_from("I", buf, 80)[0]
    total_retrans = struct.unpack_from("I", buf, 100)[0]
    retrans_pkts = struct.unpack_from("I", buf, 36)[0]
    lost_pkts = struct.unpack("I", buf, 32)[0]
    pacing_rate_Bps = struct.unpack("I", buf, 104)[0]
    bytes_acked = struct.unpack("I", buf, 120)[0]
    byts_recevied = struct.unpack("I", buf, 128)[0]
    '''


    out['rtt_us']            = u32(68)
    out['rttvar_us']         = u32(72)
    out['snd_cwnd']          = u32(80)
    out['total_retrans']     = u32(100)
    out['lost_pkts']         = u32(32)  # optional loss signal
    out['retrans_pkts']      = u32(36)
    out['pacing_rate_Bps']   = u64(104)
    out['bytes_acked']       = u64(120)
    out['bytes_received']    = u64(128)
    out["bytes_sent"]        = u64(200)

    return out

def get_tcp_info(sock: socket.socket) -> dict:
    if sys.platform.startswith('linux'):
        TCP_INFO = getattr(socket, 'TCP_INFO', None)
        if TCP_INFO is None:
            raise TCPInfoError('TCP_INFO not available')
        buf = sock.getsockopt(socket.IPPROTO_TCP, TCP_INFO, 256)
        return _tcpinfo_linux_parse(buf)
    else:
        raise TCPInfoError(f'Unsupported platform: {sys.platform!r}')

@dataclass
class TCPSample:
    t: float
    snd_cwnd_pkts: int
    rtt_us: int
    loss_signal: int
    rttvar_us: int = 0
    pacing_rate_Bps: int = 0
    bytes_acked: int = 0
    bytes_sent: int = 0
    delivery_rate_Bps: int = 0

#def sample_tcp_info(sock: socket.socket, sample_interval: float, duration: float) -> List[TCPSample]:
def sample_tcp_info(sock: socket.socket, sample_interval: float, duration: float) -> pd.DataFrame:
    t0 = time.time()
    #samples: List[TCPSample] = []
    #samples: List[dict[str, float]] = []

    samples: dict[str, list[float]] = {
        "ts": [],
        "snd_cwnd": [],
        "rtt_us": [],
        #"loss_signal": [],
        "total_retrans": [],
        "lost_pkts": [],
        "retrans_pkts": [],
        "rttvar_us": [],
        "pacing_rate_Bps": [],
        "bytes_acked": [],
        "bytes_sent": [],
        "delivery_rate_Bps": [],
    }

    ## TODO samples should be a df
    next_t = t0
    while True:
        now = time.time()
        if now - t0 > duration:
            break
        if now < next_t:
            time.sleep(min(0.002, next_t - now))
            continue
        try:
            info = get_tcp_info(sock)
        except Exception:
            break
        t_rel = now - t0
        #loss = info.get('total_retrans', info.get('lost_pkts', info.get('retrans_pkts', 0)))

        ## store results in dictionary
        samples["ts"].append(float(t_rel))
        samples["snd_cwnd"].append(float(info.get("snd_cwnd", 0)))
        samples["rtt_us"].append(float(info.get("rtt_us", 0)))
        samples["total_retrans"].append(float(info.get("total_retrans", 0)))
        samples["lost_pkts"].append(float(info.get("lost_pkts", 0)))
        samples["retrans_pkts"].append(float(info.get("retrans_pkts", 0)))
        samples["rttvar_us"].append(float(info.get("rttvar_us", 0)))
        samples["pacing_rate_Bps"].append(float(info.get("pacing_rate_Bps", 0)))
        samples["bytes_acked"].append(float(info.get("bytes_acked", 0)))
        samples["bytes_sent"].append(float(info.get('bytes_sent', 0)))
        samples["delivery_rate_Bps"].append(float(info.get("delivery_rate_Bps", 0)))
        next_t += sample_interval
    return pd.DataFrame(samples)


