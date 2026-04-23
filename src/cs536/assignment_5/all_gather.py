import math
import os
import socket
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro

DATA_DTYPE = torch.int32
VALIDATE_ENV = "CS536_VALIDATE_ALLGATHER"
INIT_TIMEOUT = timedelta(seconds=120)


def is_power_of_two(world_size: int) -> bool:
    return world_size > 0 and (world_size & (world_size - 1)) == 0


def _require_power_of_two(world_size: int, algorithm_name: str) -> None:
    if not is_power_of_two(world_size):
        raise ValueError(
            f"{algorithm_name} requires a power-of-two world_size, got {world_size}."
        )


def _expected_gathered_tensor(world_size: int, msg_size: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.full((msg_size,), rank, dtype=DATA_DTYPE)
            for rank in range(world_size)
        ],
        dim=0,
    )


def _maybe_validate_allgather(
    local_tensor: torch.Tensor, world_size: int, msg_size: int, rank: int
) -> None:
    if os.environ.get(VALIDATE_ENV, "0") != "1":
        return
    expected = _expected_gathered_tensor(world_size, msg_size)
    if not torch.equal(local_tensor, expected):
        raise RuntimeError(
            f"AllGather validation failed on rank {rank}. "
            f"Expected rank-ordered concatenation of 0..{world_size - 1}."
        )


def _pack_ranked_chunks(
    rank_to_chunk: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_ranks = sorted(rank_to_chunk.keys())
    ranks_tensor = torch.tensor(ordered_ranks, dtype=torch.int64)
    payload = torch.cat([rank_to_chunk[r] for r in ordered_ranks], dim=0)
    return ranks_tensor, payload


def _merge_received_chunks(
    rank_to_chunk: dict[int, torch.Tensor],
    recv_ranks: torch.Tensor,
    recv_payload: torch.Tensor,
    msg_size: int,
) -> None:
    for idx, owner_rank in enumerate(recv_ranks.tolist()):
        start = idx * msg_size
        end = start + msg_size
        rank_to_chunk[owner_rank] = recv_payload[start:end].clone()


def _init_process_group(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
        timeout=INIT_TIMEOUT,
    )


def _destroy_process_group_safely() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _publish_timing(result_target, value: float) -> None:
    if result_target is None:
        return
    if hasattr(result_target, "put"):
        result_target.put(value)
        return
    Path(str(result_target)).write_text(f"{value:.17g}", encoding="utf-8")


def _read_timing(result_path: str, fn_name: str) -> float:
    path = Path(result_path)
    if not path.exists():
        raise RuntimeError(f"{fn_name} finished but produced no timing result file.")
    contents = path.read_text(encoding="utf-8").strip()
    if not contents:
        raise RuntimeError(f"{fn_name} produced an empty timing result file.")
    try:
        return float(contents)
    except ValueError as exc:
        raise RuntimeError(
            f"{fn_name} produced an invalid timing result: {contents!r}"
        ) from exc


def ring(
    rank,
    world_size,
    msg_size,
    result_queue,
    master_addr: str = "127.0.0.1",
    master_port: int = 29512,
):
    _init_process_group(rank, world_size, master_addr, master_port)

    try:
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size

        local_chunk = torch.full((msg_size,), rank, dtype=DATA_DTYPE)
        chunks_by_rank = [torch.zeros(msg_size, dtype=DATA_DTYPE) for _ in range(world_size)]
        chunks_by_rank[rank] = local_chunk.clone()

        send_tensor = local_chunk.clone()
        dist.barrier()
        if rank == 0:
            start_time = time.perf_counter()

        for step in range(world_size - 1):
            recv_tensor = torch.zeros(msg_size, dtype=DATA_DTYPE)
            send_req = dist.isend(tensor=send_tensor, dst=send_to, tag=step)
            recv_req = dist.irecv(tensor=recv_tensor, src=recv_from, tag=step)
            send_req.wait()
            recv_req.wait()

            recv_owner = (rank - step - 1 + world_size) % world_size
            chunks_by_rank[recv_owner] = recv_tensor
            send_tensor = recv_tensor

        local_tensor = torch.cat(chunks_by_rank, dim=0)

        dist.barrier()
        if rank == 0:
            end_time = time.perf_counter()
            _publish_timing(result_queue, end_time - start_time)

        _maybe_validate_allgather(local_tensor, world_size, msg_size, rank)
    finally:
        _destroy_process_group_safely()


def recursive_doubling(
    rank,
    world_size,
    msg_size,
    result_queue,
    master_addr: str = "127.0.0.1",
    master_port: int = 29512,
):
    def flip_nth_bit(x: int, n: int) -> int:
        return x ^ (1 << n)

    _require_power_of_two(world_size, "Recursive Doubling")
    _init_process_group(rank, world_size, master_addr, master_port)

    try:
        steps = int(math.log2(world_size))
        local_chunk = torch.full((msg_size,), rank, dtype=DATA_DTYPE)
        rank_to_chunk = {rank: local_chunk}

        dist.barrier()
        if rank == 0:
            start_time = time.perf_counter()

        for step in range(steps):
            partner = flip_nth_bit(rank, step)

            send_ranks, send_payload = _pack_ranked_chunks(rank_to_chunk)
            recv_ranks = torch.empty_like(send_ranks)
            recv_payload = torch.empty_like(send_payload)

            rank_tag = step * 2
            payload_tag = rank_tag + 1
            reqs = [
                dist.isend(tensor=send_ranks, dst=partner, tag=rank_tag),
                dist.irecv(tensor=recv_ranks, src=partner, tag=rank_tag),
                dist.isend(tensor=send_payload, dst=partner, tag=payload_tag),
                dist.irecv(tensor=recv_payload, src=partner, tag=payload_tag),
            ]
            for req in reqs:
                req.wait()

            _merge_received_chunks(rank_to_chunk, recv_ranks, recv_payload, msg_size)

        local_tensor = torch.cat([rank_to_chunk[r] for r in range(world_size)], dim=0)

        dist.barrier()
        if rank == 0:
            end_time = time.perf_counter()
            _publish_timing(result_queue, end_time - start_time)

        _maybe_validate_allgather(local_tensor, world_size, msg_size, rank)
    finally:
        _destroy_process_group_safely()


def swing(
    rank,
    world_size,
    msg_size,
    result_queue,
    master_addr: str = "127.0.0.1",
    master_port: int = 29512,
):
    def peer_function(s: int) -> int:
        total = 0
        for i in range(s + 1):
            total += (-2) ** i
        return total

    _require_power_of_two(world_size, "Swing")
    _init_process_group(rank, world_size, master_addr, master_port)

    try:
        steps = int(math.log2(world_size))
        local_chunk = torch.full((msg_size,), rank, dtype=DATA_DTYPE)
        rank_to_chunk = {rank: local_chunk}

        dist.barrier()
        if rank == 0:
            start_time = time.perf_counter()

        for step in range(steps):
            s = steps - 1 - step
            offset = peer_function(s) % world_size
            partner = (rank + offset) % world_size if rank % 2 == 0 else (rank - offset) % world_size

            send_ranks, send_payload = _pack_ranked_chunks(rank_to_chunk)
            recv_ranks = torch.empty_like(send_ranks)
            recv_payload = torch.empty_like(send_payload)

            rank_tag = step * 2
            payload_tag = rank_tag + 1
            reqs = [
                dist.isend(tensor=send_ranks, dst=partner, tag=rank_tag),
                dist.irecv(tensor=recv_ranks, src=partner, tag=rank_tag),
                dist.isend(tensor=send_payload, dst=partner, tag=payload_tag),
                dist.irecv(tensor=recv_payload, src=partner, tag=payload_tag),
            ]
            for req in reqs:
                req.wait()

            _merge_received_chunks(rank_to_chunk, recv_ranks, recv_payload, msg_size)

        local_tensor = torch.cat([rank_to_chunk[r] for r in range(world_size)], dim=0)

        dist.barrier()
        if rank == 0:
            end_time = time.perf_counter()
            _publish_timing(result_queue, end_time - start_time)

        _maybe_validate_allgather(local_tensor, world_size, msg_size, rank)
    finally:
        _destroy_process_group_safely()


def run_once(fn, world_size, msg_size):
    if fn in (recursive_doubling, swing):
        _require_power_of_two(world_size, fn.__name__.replace("_", " ").title())

    retry_error = None
    for _ in range(5):
        master_addr = "127.0.0.1"
        master_port = _find_free_port()

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"cs536_ag_{fn.__name__}_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            result_path = tmp_file.name
        Path(result_path).unlink(missing_ok=True)

        try:
            mp.spawn(
                fn,
                args=(world_size, msg_size, result_path, master_addr, master_port),
                nprocs=world_size,
                join=True,
                start_method="spawn",
            )
        except Exception as exc:
            message = str(exc).lower()
            if "address already in use" in message or "eaddrinuse" in message:
                retry_error = exc
                Path(result_path).unlink(missing_ok=True)
                continue
            Path(result_path).unlink(missing_ok=True)
            raise

        try:
            return _read_timing(result_path, fn.__name__)
        finally:
            Path(result_path).unlink(missing_ok=True)

    raise RuntimeError(
        f"Unable to launch {fn.__name__} after retries due to rendezvous-port conflicts."
    ) from retry_error


def _bytes_to_nelems(nb):
    return max(1, nb // 4)


def msg_size_benchmark(ax, size_bytes: list[int] | None = None, world_size: int = 4):
    if size_bytes is None:
        size_bytes = [1 << s for s in range(10, 20, 2)]

    ring_completion_time = []
    recursive_doubling_completion_time = []
    swing_completion_time = []

    for nb in size_bytes:
        msg_size = _bytes_to_nelems(nb)
        ring_t = run_once(ring, world_size, msg_size)
        recursive_doubling_t = run_once(recursive_doubling, world_size, msg_size)
        swing_t = run_once(swing, world_size, msg_size)
        ring_completion_time.append(ring_t * 1e3)
        recursive_doubling_completion_time.append(recursive_doubling_t * 1e3)
        swing_completion_time.append(swing_t * 1e3)

    ax.scatter(size_bytes, ring_completion_time, c="orange", marker="o", label="Ring")
    ax.scatter(
        size_bytes,
        recursive_doubling_completion_time,
        c="blue",
        marker="o",
        label="Recursive Doubling",
    )
    ax.scatter(size_bytes, swing_completion_time, c="red", marker="o", label="Swing")
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"AllGather - Varying Message Size (world size={world_size})")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax


def rank_benchmark(ax, world_sizes: list[int] | None = None, msg_size: int = 1024):
    if world_sizes is None:
        world_sizes = [2, 4, 8, 16]

    ring_completion_time = []
    recursive_doubling_completion_time = []
    swing_completion_time = []
    msg_elems = _bytes_to_nelems(msg_size)

    for world_size in world_sizes:
        ring_t = run_once(ring, world_size, msg_elems)
        recursive_doubling_t = run_once(recursive_doubling, world_size, msg_elems)
        swing_t = run_once(swing, world_size, msg_elems)
        ring_completion_time.append(ring_t * 1e3)
        recursive_doubling_completion_time.append(recursive_doubling_t * 1e3)
        swing_completion_time.append(swing_t * 1e3)

    ax.scatter(world_sizes, ring_completion_time, c="orange", marker="o", label="Ring")
    ax.scatter(
        world_sizes,
        recursive_doubling_completion_time,
        c="blue",
        marker="o",
        label="Recursive Doubling",
    )
    ax.scatter(world_sizes, swing_completion_time, c="red", marker="o", label="Swing")
    ax.set_xlabel("World Size")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"AllGather - Varying World Size (msg size={msg_size} bytes)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax


def main():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)
    size_bytes = [1 << s for s in range(10, 20, 2)]
    world_sizes = [2, 4, 8, 16]

    axes[0] = msg_size_benchmark(axes[0], size_bytes=size_bytes)
    axes[1] = rank_benchmark(axes[1], world_sizes=world_sizes)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
