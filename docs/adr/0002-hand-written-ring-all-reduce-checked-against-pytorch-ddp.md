# ADR 0002: Hand-written bucketed ring all-reduce, checked against PyTorch DDP

- **Status:** Accepted

## Context

M1 has to show that data parallelism replicates the model and averages gradients with all-reduce.
`torch.nn.parallel.DistributedDataParallel` does this, but it hides the mechanism: bucketing,
the all-reduce algorithm and the overlap with backward all happen inside C++. The point of the lab
is to see and measure each part.

## Decision

The default strategy (`--strategy manual`) is our own `GradientAllReducer`:

- rank 0's weights are broadcast at start, so replicas are identical even though each rank
  initialises with a different seed (the tests rely on this);
- gradients are grouped into buckets of at most `--bucket-mb` (last layers first, as in DDP),
  flattened, summed with a ring all-reduce built only on `isend`/`irecv` (reduce-scatter, then
  all-gather; each rank sends 2 (N-1)/N of the payload), and divided by the world size;
- `--algorithm builtin` swaps the ring for gloo's `all_reduce`, and `--strategy torch` runs
  PyTorch DDP itself, so all three can be compared on the same data.

## Consequences

- Correctness is checked, not assumed: the ring equals both the locally computed sum and gloo's
  all-reduce (float64, error below 1e-9, sizes 1 to 100,003 on 2, 3 and 4 ranks), and training with
  2 or 4 ranks gives the same per-step losses as one process on the whole batch (relative 1e-4).
- The all-reduce runs after backward, not overlapped with it. Communication time is therefore
  easy to separate from compute (`mean_allreduce_s`), but the step is slower than PyTorch DDP,
  which overlaps the two. Overlap and a real profiler split are M3 work.
- gloo reduces host tensors, so CUDA gradients are staged through CPU memory. On one GPU with
  one rank no all-reduce is needed and none is done.
