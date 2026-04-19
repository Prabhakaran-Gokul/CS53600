import torch 
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import math
from typing import Tuple
import matplotlib.pyplot as plt
import time 
import tyro


def binary_tree(rank, world_size, msg_size, start_queue, end_queue):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )



    left = 2 * rank + 1
    right = 2 * rank + 2

    recv_tensor = torch.zeros((msg_size), dtype=torch.float32)
    ## every one except root must reeive 
    dist.barrier()
    if rank == 0:
        start_queue.put(time.perf_counter())
        recv_tensor = torch.full((msg_size,), 420.0, dtype=torch.float32)
    else:
        parent = math.floor((rank - 1 )/2)
        dist.recv(tensor=recv_tensor, src=parent)  

    ## send them out
    if left < world_size:
        dist.send(tensor=recv_tensor, dst=left)
    if right < world_size: 
        dist.send(tensor=recv_tensor, dst=right)

    dist.barrier()
    if rank == world_size - 1:
        end_queue.put(time.perf_counter())
    
    dist.destroy_process_group()





def binomial_tree(rank, world_size,  msg_size, start_queue, end_queue):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29512"
    )

    root = 0
    virtual_rank = (rank - root + world_size) % world_size

    recv_tensor = torch.zeros(msg_size, dtype=torch.float32)
    if virtual_rank == 0:
        recv_tensor = torch.full((msg_size,), 420.0, dtype=torch.float32)

    dist.barrier()
    if rank == 0:
        start_queue.put(time.perf_counter())

    step = 1
    while step < world_size:
        if virtual_rank < step:
            # This rank already has data; send to virtual_rank + step
            dst = virtual_rank + step
            if dst < world_size:
                dist.send(tensor=recv_tensor, dst=(dst + root) % world_size)
                #print(f"Rank {rank} sent to rank {(dst + root) % world_size}")
        elif virtual_rank < 2 * step:
            src = virtual_rank - step
            dist.recv(tensor=recv_tensor, src=(src + root) % world_size)
            print(f"Rank {rank} received from rank {(src + root) % world_size}")
        step *= 2

    dist.barrier()
    
    if rank == world_size - 1:
        end_queue.put(time.perf_counter())
    
    dist.destroy_process_group()

def run_once(fn, world_size, msg_size):

    start_queue = mp.Queue()
    end_queue = mp.Queue()
    mp.spawn(fn, args=(world_size, msg_size, start_queue, end_queue), nprocs=world_size, join=True)
    return end_queue.get() - start_queue.get()


def _bytes_to_nelems(nb):
    return max(1, nb // 4) 

def msg_size_benchmark( ax : plt.axes, size_bytes: list[int], world_size : int = 4):
    binary_completion_time = []
    binomial_completion_time = []

    for idx, nb in enumerate(size_bytes):
        msg_size =  _bytes_to_nelems(nb)
        bnary_t = run_once(binary_tree, world_size, msg_size)
        binomial_t = run_once(binomial_tree, world_size, msg_size)
        binary_completion_time.append(bnary_t * 1e3)
        binomial_completion_time.append(binomial_t * 1e3)

    ax.scatter(size_bytes, binary_completion_time, c = "orange", marker = "o", label = "Binary Tree")
    ax.scatter(size_bytes, binomial_completion_time, c = "blue", marker = "o", label = "Binomial Tree")
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"Broadcast — Varying Message Size (world size={world_size})")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax 

def rank_benchmark( ax : plt.axes, world_sizes: list[int], msg_size : int = 1024):
    
    binary_completion_time = []
    binomial_completion_time = []

    for idx, world_size in enumerate(world_sizes):
        print(f"World size {world_size}")
        bnary_t = run_once(binary_tree, world_size, msg_size)
        binomial_t = run_once(binomial_tree, world_size, msg_size)
        binary_completion_time.append(bnary_t * 1e3)
        binomial_completion_time.append(binomial_t * 1e3)

    ax.scatter(world_sizes, binary_completion_time, c = "orange", marker = "o", label = "Binary Tree")
    ax.scatter(world_sizes, binomial_completion_time, c = "blue", marker = "o", label = "Binomial Tree")
    ax.set_xlabel("World Size")
    ax.set_ylabel("Completion Time (ms)")
    ax.set_title(f"Broadcast — Varying World Size (msg size={msg_size} bytes)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return ax 


def main():

    ## Binary Tree
    # world_size = 16
    # mp.spawn(binary_tree, args=(world_size,), nprocs=world_size, join=True)


    #world_size = 16
    #mp.spawn(binomial_tree, args=(world_size,), nprocs=world_size, join=True)
    fig, axes = plt.subplots(2, 1)
    size_bytes = [1 << s for s in range(10, 20, 2)]
    world_sizes = [2, 4, 8, 16]

    axes[0] = msg_size_benchmark(axes[0], size_bytes)
    axes[1] = rank_benchmark(axes[1], world_sizes)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tyro.cli(main)
