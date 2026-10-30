# ADR 0003: gloo CPU ranks stand in for 2 and 4 GPUs in M1

- **Status:** Accepted

## Context

The backlog asks for training on 1, 2 and 4 GPUs. The development machine has one RTX 4050
Laptop GPU (6 GB) and Windows 11, where PyTorch ships no NCCL backend. Renting a multi-GPU node is
possible but costs money for every change and cannot run in CI.

## Decision

- 1, 2 and 4 ranks run as real separate processes with the gloo backend on the CPU, each rank
  limited to the same thread budget (`--threads-per-rank`, default 2), so a rank behaves like one
  fixed-size device. The same global batch is split across ranks (strong scaling).
- The single GPU is used for a 1-rank CUDA run and for 2 ranks sharing that GPU, which exercises
  the CUDA code path but says nothing about multi-GPU scaling.
- The code takes a device per rank (`cuda:rank % device_count`), so on a machine with several GPUs
  the same command uses one GPU per rank; the backend would be NCCL there, which M1 does not test.

## Consequences

- Every claim in M1 is measured on this machine or in CI: multi-process DDP correctness, replica
  equality, the all-reduce cost, and CPU strong-scaling efficiency.
- CPU ranks share one memory bus and the same laptop CPU as other work, so scaling numbers are
  noisy and lower than on separate GPUs; they show the method, not GPU speedups.
- Rejected alternatives: threads in one process (no real process group, no real communication),
  and renting GPUs (cost per change, not reproducible in CI). Real multi-GPU numbers stay open.
