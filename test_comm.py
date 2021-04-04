import torch
import torch.distributed as dist
import os
from torch.multiprocessing import Process
""" All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    group1 = dist.new_group([2, 3])
    tensor = torch.ones(5).cuda(rank)
    tensor1 = torch.ones(5).cuda(rank)
    dist.all_reduce(tensor, group=group)
    dist.all_reduce(tensor1, group=group1)
    torch.cuda.synchronize()
    print('Rank ', rank, ' has data ', tensor[0],tensor1)

def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()