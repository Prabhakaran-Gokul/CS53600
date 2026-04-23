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

VALIDATE_ENV = "CS536_VALIDATE_BROADCAST"
ROOT_VALUE = 420.0
INIT_TIMEOUT = timedelta(seconds=120)


def _expected_broadcast_tensor(msg_size: int) -> torch.Tensor:
    return torch.full((msg_size,), ROOT_VALUE, dtype=torch.float32)


def _maybe_validate_broadcast(recv_tensor: torch.Tensor, msg_size: int, rank: int) -> None:
    if os.environ.get(VALIDATE_ENV, "0") != "1":
        return
    expected = _expected_broadcast_tensor(msg_size)
    if not torch.equal(recv_tensor, expected):
        raise RuntimeError(
            f"Broadcast validation failed on rank {rank}. "
            f"Expected every element to equal {ROOT_VALUE}."
        )


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


def binary_tree(
    rank,
    world_size,
    msg_size,
    result_queue,
    master_addr: str = "127.0.0.1",
    master_port: int = 29512,
):
    _init_process_group(rank, world_size, master_addr, master_port)

    try:
        left = 2 * rank + 1
        right = 2 * rank + 2

        recv_tensor = torch.zeros(msg_size, dtype=torch.float32)

        dist.barrier()
        if rank == 0:
            start_time = time.perf_counter()
            recv_tensor = _expected_broadcast_tensor(msg_size)
        else:
            parent = math.floor((rank - 1) / 2)
            dist.recv(tensor=recv_tensor, src=parent)

        if left < world_size:
            dist.send(tensor=recv_tensor, dst=left)
        if right < world_size:
            dist.send(tensor=recv_tensor, dst=right)

        dist.barrier()
        if rank == 0:
            end_time = time.perf_counter()
            _publish_timing(result_queue, end_time - start_time)

        _maybe_validate_broadcast(recv_tensor, msg_size, rank)
    finally:
        _destroy_process_group_safely()


def binomial_tree(
    rank,
    world_size,
    msg_size,
    result_queue,
    master_addr: str = "127.0.0.1",
    master_port: int = 29512,
):
    _init_process_group(rank, world_size, master_addr, master_port)

    try:
        root = 0
        virtual_rank = (rank - root + world_size) % world_size

        recv_tensor = torch.zeros(msg_size, dtype=torch.float32)
        if virtual_rank == 0:
            recv_tensor = _expected_broadcast_tensor(msg_size)

        dist.barrier()
        if rank == 0:
            start_time = time.perf_counter()

        step = 1
        while step < world_size:
            if virtual_rank < step:
                dst = virtual_rank + step
                if dst < world_size:
                    dist.send(tensor=recv_tensor, dst=(dst + root) % world_size)
            elif virtual_rank < 2 * step:
                src = virtual_rank - step
                dist.recv(tensor=recv_tensor, src=(src + root) % world_size)
            step *= 2

        dist.barrier()
        if rank == 0:
            end_time = time.perf_counter()
            _publish_timing(result_queue, end_time - start_time)

        _maybe_validate_broadcast(recv_tensor, msg_size, rank)
    finally:
        _destroy_process_group_safely()


def run_once(fn, world_size, msg_size):
    retry_error = None
    for _ in range(5):
        master_addr = "127.0.0.1"
        master_port = _find_free_port()

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"cs536_bc_{fn.__name__}_",
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


def msg_size_benchmark(ax, size_bytes: list[int], world_size: int = 4):
    binary_completion_time = []
    binomial_completion_time = []

    for nb in size_bytes:
        msg_size = _bytes_to_nelems(nb)
        binary_t = run_once(binary_tree, world_size, msg_size)
        binomial_t = run_once(binomial_tree, world_size, msg_size)
        binary_completion_time.append(binary_t * 1e3)
        binomial_completion_time.append(binomial_t * 1e3)

    ax.scatter(size_bytes, binary_completion_time, c="orange", marker="o", label="Binary Tree")
    ax.scatter(size_bytes, binomial_completion_time, c="blue", marker="o", label="Binomial Tree")
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"Broadcast - Varying Message Size (world size={world_size})")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax


def rank_benchmark(ax, world_sizes: list[int], msg_size: int = 1024):
    binary_completion_time = []
    binomial_completion_time = []
    msg_elems = _bytes_to_nelems(msg_size)

    for world_size in world_sizes:
        binary_t = run_once(binary_tree, world_size, msg_elems)
        binomial_t = run_once(binomial_tree, world_size, msg_elems)
        binary_completion_time.append(binary_t * 1e3)
        binomial_completion_time.append(binomial_t * 1e3)

    ax.scatter(world_sizes, binary_completion_time, c="orange", marker="o", label="Binary Tree")
    ax.scatter(world_sizes, binomial_completion_time, c="blue", marker="o", label="Binomial Tree")
    ax.set_xlabel("World Size")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"Broadcast - Varying World Size (msg size={msg_size} bytes)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax


def main():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)
    size_bytes = [1 << s for s in range(10, 20, 2)]
    world_sizes = [2, 4, 8, 16]

    axes[0] = msg_size_benchmark(axes[0], size_bytes)
    axes[1] = rank_benchmark(axes[1], world_sizes)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
