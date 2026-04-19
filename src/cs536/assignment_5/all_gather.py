import torch 
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import math
from typing import Tuple
import time 
import queue
import matplotlib.pyplot as plt
import tyro

## all gather prcess is gatering all N values from k ranks 
# AllGather operation: each rank receives the aggregation of data from all ranks in the order of the ranks.


def ring(rank, world_size, msg_size, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )

    #tensor = torch.tensor([rank], dtype=torch.float32)
    #print(f"Rank {rank} starting with tensor: {tensor.item()}")

    if rank == world_size - 1:
        send_to = 0 
    else: 
        send_to = rank + 1
    
    if rank == 0:
        recv_from = world_size - 1
    else:
        recv_from = rank - 1
    

    local_tensor = torch.full((msg_size,), float(rank), dtype=torch.int)
    send_tensor = local_tensor.clone()
    dist.barrier()
    if rank == 0:
        start_time = time.perf_counter()
    
    for i in range(world_size - 1):

        recv_tensor = torch.zeros(msg_size, dtype=torch.int)
        send_req = dist.isend(tensor=send_tensor, dst=send_to)
        recv_req = dist.irecv(tensor=recv_tensor, src=recv_from)  
        send_req.wait()
        recv_req.wait()
        local_tensor = torch.cat((local_tensor, recv_tensor))  
        send_tensor = recv_tensor

    dist.barrier()
    if rank == 0:
        end_time = time.perf_counter()
        result_queue.put(end_time - start_time)

    #print(f"Rank {rank}: {local_tensor}")
    dist.destroy_process_group()

def recursive_doubling(rank, world_size, msg_size, result_queue):

    def flip_nth_bit(x: int, n: int) -> int:
        return x ^ (1 << n)
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )

    #tensor = torch.full((msg_size,), float(rank), dtype=torch.float32)
    #print(f"Rank {rank} starting with tensor: {tensor.item()}")

    steps = int(math.log2(world_size))
    local_tensor = torch.full((msg_size,), float(rank), dtype=torch.float32)
    dist.barrier()
    if rank == 0:
        start_time = time.perf_counter()

    for i in range(steps):
        partner  = flip_nth_bit(rank, i)
        recv_tensor = torch.zeros(len(local_tensor), dtype=torch.int) 
        send_req = dist.isend(tensor=local_tensor, dst=partner)
        recv_req = dist.irecv(tensor=recv_tensor, src=partner)  
        send_req.wait()
        recv_req.wait()
        local_tensor = torch.cat((local_tensor, recv_tensor))

    dist.barrier()
    if rank == 0:
        end_time = time.perf_counter()
        result_queue.put(end_time - start_time)

    #print(f"Rank {rank}: {local_tensor}")
    dist.destroy_process_group()

def swing(rank, world_size, msg_size, result_queue):
    
    def peer_function(s) -> int:
        total = 0
        for i in range(s + 1):
            total += (-2) ** i
        return total

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )


    steps = int(math.log2(world_size))
    local_tensor = torch.full((msg_size,), float(rank), dtype=torch.int)

    dist.barrier()
    if rank == 0:
        start_time = time.perf_counter()

    for step in range(steps):
        s = int(math.log2(world_size)) - 1 - step 

        if rank % 2 == 0: 
            partner = rank + peer_function(s) % world_size
        else: 
            partner = rank - peer_function(s) % world_size

        if partner >= world_size:
            partner = partner - world_size
        if partner < 0:

            partner = world_size + partner

        #print(f"rank {rank} partner {partner}")
        recv_tensor = torch.zeros(len(local_tensor), dtype=torch.int) 
        send_req = dist.isend(tensor=local_tensor, dst=partner)
        recv_req = dist.irecv(tensor=recv_tensor, src=partner)  
        send_req.wait()
        recv_req.wait()
        local_tensor = torch.cat((local_tensor, recv_tensor))

    dist.barrier()
    if rank == 0:
        end_time = time.perf_counter()
        result_queue.put(end_time - start_time)

    #print(f"Rank {rank}: {local_tensor}")
    dist.destroy_process_group()

def run_once(fn, world_size, msg_size):

    result_queue = mp.Queue()
    mp.spawn(fn, args=(world_size, msg_size, result_queue), nprocs=world_size, join=True)
    return result_queue.get()

def _bytes_to_nelems(nb):
    return max(1, nb // 4) 

def msg_size_benchmark( ax : plt.axes, size_bytes: list[int], world_size : int = 4):
    ring_completion_time = []
    recursive_doubling_completion_time = []
    swing_completion_time = []

    for idx, nb in enumerate(size_bytes):
        msg_size =  _bytes_to_nelems(nb)
        ring_t = run_once(ring, world_size, msg_size)
        recursive_doubling_t = run_once(recursive_doubling, world_size, msg_size)
        swing_t = run_once(swing, world_size, msg_size)
        ring_completion_time.append(ring_t * 1e3)
        recursive_doubling_completion_time.append(recursive_doubling_t * 1e3)
        swing_completion_time.append(swing_t * 1e3)

    ax.scatter(size_bytes, ring_completion_time, c = "orange", marker = "o", label = "Ring")
    ax.scatter(size_bytes, recursive_doubling_completion_time, c = "blue", marker = "o", label = "Recursive Doubling")
    ax.scatter(size_bytes, swing_completion_time, c = "red", marker = "o", label = "Swing")
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"AllGather — Varying Message Size (world size={world_size})")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax 


def rank_benchmark( ax : plt.axes, world_sizes: list[int], msg_size : int = 1024):
    ring_completion_time = []
    recursive_doubling_completion_time = []
    swing_completion_time = []

    for idx, world_size in enumerate(world_sizes):
        print(f"World size {world_size}")
        ring_t = run_once(ring, world_size, _bytes_to_nelems(msg_size))
        recursive_doubling_t = run_once(recursive_doubling, world_size, _bytes_to_nelems(msg_size))
        swing_t = run_once(swing, world_size, _bytes_to_nelems(msg_size))
        ring_completion_time.append(ring_t * 1e3)
        recursive_doubling_completion_time.append(recursive_doubling_t * 1e3)
        swing_completion_time.append(swing_t * 1e3)

    ax.scatter(world_sizes, ring_completion_time, c = "orange", marker = "o", label = "Ring")
    ax.scatter(world_sizes, recursive_doubling_completion_time, c = "blue", marker = "o", label = "Recursive Doubling")
    ax.scatter(world_sizes, swing_completion_time, c = "red", marker = "o", label = "Swing")
    ax.set_xlabel("World Size")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"AllGather — Varying World Size (msg  size={msg_size} bytes)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax 

def main():

    fig, axes = plt.subplots(2, 1)


    axes[0] = msg_size_benchmark(axes[0])
    axes[1] = rank_benchmark(axes[1])
    plt.tight_layout()
    plt.show()

'''
python -m cs536.assignment_5.all_gather

'''
if __name__ == "__main__":
    tyro.cli(main)

