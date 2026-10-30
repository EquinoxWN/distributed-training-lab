"""Collectives for data parallelism: a ring all-reduce built on send/recv, and gradient buckets."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn


def ring_all_reduce(tensor: Tensor) -> None:
    """Sum a 1-D tensor across ranks in place: reduce-scatter then all-gather around a ring."""
    world = dist.get_world_size()
    if world == 1 or tensor.numel() == 0:
        return
    if tensor.dim() != 1 or not tensor.is_contiguous():
        raise ValueError("ring_all_reduce expects a contiguous 1-D tensor")
    rank = dist.get_rank()
    n = tensor.numel()
    per = -(-n // world)
    buf = torch.zeros(per * world, dtype=tensor.dtype, device=tensor.device)
    buf[:n].copy_(tensor)
    chunks = list(buf.chunk(world))
    right, left = (rank + 1) % world, (rank - 1) % world
    incoming = torch.empty(per, dtype=tensor.dtype, device=tensor.device)
    # Reduce-scatter: after world-1 hops, rank r holds the full sum of chunk (r + 1) % world.
    for s in range(world - 1):
        send_idx, recv_idx = (rank - s) % world, (rank - s - 1) % world
        _exchange(chunks[send_idx], incoming, right, left)
        chunks[recv_idx].add_(incoming)
    # All-gather: pass the finished chunks around so every rank has every sum.
    for s in range(world - 1):
        send_idx, recv_idx = (rank - s + 1) % world, (rank - s) % world
        _exchange(chunks[send_idx], incoming, right, left)
        chunks[recv_idx].copy_(incoming)
    tensor.copy_(buf[:n])


def _exchange(send: Tensor, recv: Tensor, right: int, left: int) -> None:
    """Send to the right neighbour while receiving from the left one."""
    reqs = [dist.isend(send, right), dist.irecv(recv, left)]
    for req in reqs:
        if req is None:
            raise RuntimeError("point-to-point request was not created")
        req.wait()


def ring_bytes_sent_per_rank(numel: int, world: int, element_size: int = 4) -> int:
    """Bytes one rank sends in a ring all-reduce of numel elements: 2 (N-1) / N of the payload."""
    if world == 1:
        return 0
    per = -(-numel // world)
    return 2 * (world - 1) * per * element_size


def all_reduce_sum(tensor: Tensor, algorithm: Literal["ring", "builtin"]) -> None:
    """Sum across ranks with the chosen algorithm."""
    if algorithm == "ring":
        ring_all_reduce(tensor)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Copy parameters and buffers from rank src to every rank (via CPU for gloo)."""
    for t in [*module.parameters(), *module.buffers()]:
        cpu = t.detach().to("cpu").contiguous()
        dist.broadcast(cpu, src=src)
        with torch.no_grad():
            t.copy_(cpu.to(t.device))


def make_buckets(params: list[nn.Parameter], bucket_bytes: int) -> list[list[nn.Parameter]]:
    """Group parameters into buckets of at most bucket_bytes, last layers first (as in DDP)."""
    if bucket_bytes < 1:
        raise ValueError("bucket_bytes must be positive")
    buckets: list[list[nn.Parameter]] = []
    current: list[nn.Parameter] = []
    size = 0
    for p in reversed(params):
        nbytes = p.numel() * p.element_size()
        if current and size + nbytes > bucket_bytes:
            buckets.append(current)
            current, size = [], 0
        current.append(p)
        size += nbytes
    if current:
        buckets.append(current)
    return buckets


@dataclass
class CommStats:
    """What gradient synchronisation cost on this rank."""

    seconds: float = 0.0
    calls: int = 0
    elements: int = 0
    per_step: list[float] = field(default_factory=list)


class GradientAllReducer:
    """Average gradients across ranks: flatten each bucket, all-reduce, divide by world size."""

    def __init__(
        self,
        module: nn.Module,
        algorithm: Literal["ring", "builtin"] = "ring",
        bucket_bytes: int = 4 << 20,
    ) -> None:
        """Broadcast rank 0's weights so every replica starts identical, then plan buckets."""
        self.world = dist.get_world_size()
        self.algorithm = algorithm
        self.params = [p for p in module.parameters() if p.requires_grad]
        self.buckets = make_buckets(self.params, bucket_bytes)
        self.stats = CommStats()
        broadcast_module(module)

    def sync(self) -> None:
        """All-reduce and average every gradient; missing gradients count as zero."""
        start = time.perf_counter()
        if self.world == 1:
            self.stats.per_step.append(0.0)
            return
        for bucket in self.buckets:
            grads = [
                p.grad.detach().reshape(-1)
                if p.grad is not None
                else torch.zeros_like(p).reshape(-1)
                for p in bucket
            ]
            # gloo reduces CPU tensors; CUDA gradients are staged through host memory.
            flat = torch.cat(grads).to("cpu")
            all_reduce_sum(flat, self.algorithm)
            flat /= self.world
            offset = 0
            for p in bucket:
                n = p.numel()
                piece = flat[offset : offset + n].view_as(p).to(p.device)
                if p.grad is None:
                    p.grad = piece.clone()
                else:
                    p.grad.copy_(piece)
                offset += n
            self.stats.calls += 1
            self.stats.elements += offset
        if self.params and self.params[0].is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.stats.seconds += elapsed
        self.stats.per_step.append(elapsed)


def flat_params(module: nn.Module) -> Tensor:
    """All parameters as one CPU float tensor."""
    return torch.cat([p.detach().reshape(-1).float().cpu() for p in module.parameters()])


def max_replica_divergence(module: nn.Module) -> float:
    """Largest absolute difference between any rank's parameters and rank 0's."""
    mine = flat_params(module)
    ref = mine.clone()
    dist.broadcast(ref, src=0)
    diff = torch.tensor([float((mine - ref).abs().max())])
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    return float(diff.item())
