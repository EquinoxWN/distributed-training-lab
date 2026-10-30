# RFC 0001: distributed-training-lab design

- **Status:** Accepted for M1
- **Author:** EquinoxWN
- **Created:** 2026-10

## Problem (Define)

A model that is too slow or too large for one GPU has to be trained across several. The usual
answer is "use DDP" (later FSDP or pipeline parallelism), but the cost of each strategy is rarely
measured: what is communicated, how long it takes compared with compute, and whether the
distributed run still computes the same thing as one device. This lab builds the strategies one at
a time and measures each one. M1 covers data parallelism: a scaled-up version of the
tiny-transformer-scratch model trained on 1, 2 and 4 ranks with DDP.

**Users:** engineers and students who want to understand and check data-parallel training
before paying for a multi-GPU node.

**Critical-to-quality measures (M1):**

| CTQ | Target | How it is checked |
|---|---|---|
| Replicas stay identical | max abs parameter difference across ranks = 0 after training | `replica_max_abs_diff` in every run, asserted in tests and CI |
| DDP computes the same update as one process | per-step losses equal within relative 1e-4 for 2 and 4 ranks | `test_ddp_matches_single_process_training` |
| Ring all-reduce is a correct sum | error below 1e-9 (float64) against the expected sum and gloo | `dtlab verify`, `test_ring_all_reduce_matches_expected_sum_and_gloo` |
| Hand-written DDP agrees with PyTorch DDP | same per-step losses within relative 1e-4 | `test_manual_ddp_matches_pytorch_ddp` |
| The model learns | final training loss below the first; held-out loss reported | every run summary |
| Cost is visible | step time, all-reduce time, tokens/s and scaling efficiency per world size | `dtlab scale` |

## Goals

- A GPT-style decoder (the tiny-transformer-scratch architecture: causal multi-head attention,
  GELU MLP, pre-LayerNorm, residuals, tied output head) in PyTorch, with a `tiny` preset and a
  `scaled` preset of 4.84M parameters.
- Training on 1, 2 and 4 ranks as separate processes in a real `torch.distributed` process group.
- DDP: replicate the model (broadcast from rank 0), shard each global batch across ranks, average
  gradients with all-reduce, step the same optimizer on every rank.
- Measure every run and keep the result as JSON.

## Non-goals (M1)

- FSDP / ZeRO sharding and pipeline parallelism (M2).
- Activation checkpointing, mixed precision, a profiler split of communication and compute (M3).
- Multi-node runs and the NCCL backend; renting GPUs.
- A hosted service.

## Design

```
 dtlab scale/train ──► run(cfg) ──► mp.spawn(N ranks) ──► gloo process group (TCP, localhost)
                                         │
             each rank:  GPT ─► broadcast rank 0 weights
                         rank_batch(step, rank)  (its slice of one seeded global batch)
                         forward + backward      (timed: compute)
                         GradientAllReducer.sync (timed: all-reduce; buckets, ring, / N)
                         AdamW step              (same on every rank)
             rank 0:  replica check, held-out loss, JSON summary
```

- **Data.** Byte-level tokens (vocabulary 256) from the TinyStories validation file at a pinned
  dataset commit (19,447,282 bytes, SHA-256 checked, HTTPS only, size-capped), or a generated
  corpus for tests and CI. The last 5% is held out. For step `s`, one generator seeded with
  `(seed, s)` draws the start offsets of the whole global batch; rank `r` takes the `r`-th
  contiguous slice, so the union over ranks is exactly the single-process batch.
- **Gradient averaging.** Each rank's loss is the mean over its slice; with equal slice sizes the
  average of the rank gradients is the gradient of the global-batch mean. Gradients are bucketed
  (default 4 MB, last layers first), flattened, summed with a ring all-reduce built on
  `isend`/`irecv` (or gloo's `all_reduce`), and divided by N.
- **Replica check.** After training, rank 0's flattened parameters are broadcast and every rank
  reports its maximum absolute difference; the maximum over ranks must be 0.
- **Devices.** gloo on the CPU for 1, 2 and 4 ranks, each with a fixed thread budget; CUDA for one
  rank on the RTX 4050, and 2 ranks sharing it (gradients staged through host memory).

## Alternatives considered

| Option | Why not (for M1) |
|---|---|
| Use only `torch.nn.parallel.DistributedDataParallel` | Correct and faster (overlaps all-reduce with backward), but the mechanism stays hidden and communication cannot be timed apart from compute. Kept as `--strategy torch` and used as a reference in tests. |
| Rent a 2- or 4-GPU node with NCCL | Real GPU scaling numbers, but costs money for every change and cannot run in CI. Planned once the strategies exist (see ADR 0003). |
| Simulate ranks with threads in one process | Cheap, but no real process group, no real communication and no separate memory; it would not test DDP. |
| Parameter server (ranks push gradients to one server) | Simple, but the server link carries N times the payload; ring all-reduce sends 2 (N-1)/N per rank, which is why DDP uses it. |

## Measurement plan (Measure)

- `make test`: 48 tests, including real 2-, 3- and 4-process runs.
- `dtlab verify --world-size 4`: ring all-reduce error.
- `make scale`: `scaled` model on TinyStories, global batch 32 x 128 tokens, 15 steps (3 warm-up),
  on 1, 2 and 4 gloo ranks with 2 threads each; reports mean step time, mean all-reduce time,
  tokens/s, speedup and efficiency against 1 rank, and held-out loss.
- `make gpu`: the same model for 300 steps on the RTX 4050 (1 rank), then 2 ranks sharing it.
- Machine: Intel Core i7-13700H, 13.7 GB RAM, RTX 4050 Laptop 6 GB, Windows 11, PyTorch 2.14.1.
- Results and their analysis: [docs/results/m1.md](../results/m1.md).

## Control

CI (ubuntu, Python 3.12, CPU PyTorch) runs ruff, mypy `--strict`, all tests, the 4-rank all-reduce
check and a 1/2/4-rank run on generated data whose replica equality and loss equality are asserted
and printed as `::notice` annotations, then `pip-audit`. Configuration is validated up front
(uneven batch splits, sizes and paths outside the run folder are refused).

## Milestones

- **M1:** A scaled-up version of tiny-transformer-scratch is trained on 1, 2 and 4 ranks. DDP replicates the model and averages gradients with all-reduce.
- **M2:** FSDP shards parameters, gradients and optimizer state (ZeRO-style), so larger models fit per GPU. Pipeline parallelism splits layers across GPUs and uses micro-batches to reduce idle time.
- **M3:** Activation checkpointing and mixed precision trade compute for memory. The profiler separates communication from compute, and scaling efficiency is reported per strategy.

## Risks and open questions

- CPU ranks share the laptop's memory bus and run beside other work, so scaling numbers are noisy
  and say little about GPU clusters.
- The all-reduce is not overlapped with backward; on real GPUs that overlap is a large part of
  DDP's efficiency and is not measured here.
- PyTorch DDP over gloo with CUDA tensors crashed on Windows (access violation), so the PyTorch DDP
  comparison is CPU-only; the CLI refuses that combination.
