import torch 
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import math
from typing import Tuple
import tyro

## all gather prcess is gatering all N values from k ranks 
# AllGather operation: each rank receives the aggregation of data from all ranks in the order of the ranks.


def ring(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )

    tensor = torch.tensor([rank], dtype=torch.float32)
    print(f"Rank {rank} starting with tensor: {tensor.item()}")

    if rank == world_size - 1:
        send_to = 0 
    else: 
        send_to = rank + 1
    
    if rank == 0:
        recv_from = world_size - 1
    else:
        recv_from = rank - 1
    

    '''
    rank_tensor = torch.tensor([rank], dtype=torch.int)
    send_tensor = torch.tensor([rank], dtype=torch.int)
    for i in range(world_size - 1):

        recv_tensor = torch.zeros((i + 1), dtype=torch.int)
        send_req = dist.isend(tensor=send_tensor, dst=send_to)
        recv_req = dist.irecv(tensor=recv_tensor, src=recv_from)

        # Wait for communication to complete
        send_req.wait()
        recv_req.wait()
        send_tensor = torch.cat((recv_tensor, rank_tensor))
    print(f"Rank {rank}: {send_tensor}")
    '''
    local_tensor = torch.tensor([rank], dtype=torch.int)
    send_tensor = torch.tensor([rank], dtype=torch.int)

    for i in range(world_size - 1):

        recv_tensor = torch.zeros(1, dtype=torch.int)
        send_req = dist.isend(tensor=send_tensor, dst=send_to)
        recv_req = dist.irecv(tensor=recv_tensor, src=recv_from)  
        send_req.wait()
        recv_req.wait()
        local_tensor = torch.cat((local_tensor, recv_tensor))  
        send_tensor = recv_tensor

    print(f"Rank {rank}: {local_tensor}")
    dist.destroy_process_group()

def recursive_doubling(rank, world_size):

    def flip_nth_bit(x: int, n: int) -> int:
        return x ^ (1 << n)
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )

    tensor = torch.tensor([rank], dtype=torch.float32)
    print(f"Rank {rank} starting with tensor: {tensor.item()}")

    steps = int(math.log2(world_size))
    rank_binary = bin(rank)

    local_tensor = torch.tensor([rank], dtype=torch.int)

    for i in range(steps):
        partner  = flip_nth_bit(rank, i)
        recv_tensor = torch.zeros((2**i), dtype=torch.int) 
        send_req = dist.isend(tensor=local_tensor, dst=partner)
        recv_req = dist.irecv(tensor=recv_tensor, src=partner)  
        send_req.wait()
        recv_req.wait()
        local_tensor = torch.cat((local_tensor, recv_tensor))

    print(f"Rank {rank}: {local_tensor}")
    dist.destroy_process_group()

def swing(rank, world_size):
    
    def peer_function(s) -> int:
        total = 0
        for i in range(s + 1):
            total += (-2) ** i
        return total

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method = "tcp://127.0.0.1:29512"
    )

    tensor = torch.tensor([rank], dtype=torch.float32)

    steps = int(math.log2(world_size))
    local_tensor = torch.tensor([rank], dtype=torch.int)

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
        recv_tensor = torch.zeros((2**step), dtype=torch.int) 
        send_req = dist.isend(tensor=local_tensor, dst=partner)
        recv_req = dist.irecv(tensor=recv_tensor, src=partner)  
        send_req.wait()
        recv_req.wait()
        local_tensor = torch.cat((local_tensor, recv_tensor))

    print(f"Rank {rank}: {local_tensor}")
    dist.destroy_process_group()

def main():

    ## Ring
    #world_size = 4
    #mp.spawn(ring, args=(world_size,), nprocs=world_size, join=True)

    ## Recursive Doubling
    #world_size = 8
    #mp.spawn(recursive_doubling, args=(world_size,), nprocs=world_size, join=True)

    ## Swing AllGather
    world_size = 8
    mp.spawn(swing, args=(world_size,), nprocs=world_size, join=True)

'''
python -m cs536.assignment_5.all_gather

'''
if __name__ == "__main__":
    tyro.cli(main)

