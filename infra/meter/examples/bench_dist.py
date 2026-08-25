'''
Distributed measurement example: a toy "train step" (matmul + all_reduce),
benchmarked per rank with an imbalance report.

    # cpu/gloo on macOS: pin loopback + IPv4, otherwise the c10d rendezvous
    # hangs forever on a failing IPv6 reverse-DNS lookup of ::1
    GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node 2 \
        --master_addr 127.0.0.1 --master_port 29517 infra/meter/examples/bench_dist.py

    torchrun --nproc_per_node 8 infra/meter/examples/bench_dist.py --nccl     # GPU box
'''

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from infra.meter.dist import dist_bench, render_dist_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nccl', action='store_true', help='use nccl + one GPU per rank')
    p.add_argument('--size', type=int, default=1024, help='square matrix side')
    args = p.parse_args()

    if args.nccl:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    else:
        device = torch.device('cpu')
        dist.init_process_group('gloo')

    w = torch.randn(args.size, args.size, device=device)
    x = torch.randn(args.size, args.size, device=device)

    def train_step(x):
        # compute + gradient-sync stand-in: the timed unit of DDP training
        y = x @ w
        dist.all_reduce(y)
        return y

    results = dist_bench(train_step, (x,), name='matmul+all_reduce',
                         warmup=5, iters=20)
    if results is not None:
        print(render_dist_table(results))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
