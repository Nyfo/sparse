# Benchmark Overview

This folder is organized around two benchmark roles:

1. Structured benchmarks
   - regular sparsity patterns such as `banded5` and `stencil`
   - used for the main dense-vs-sparse and CPU-vs-GPU comparisons

2. Irregular benchmarks
   - the `spiky` family
   - used for pipeline breakdowns and harder irregular sparsity cases

## Files

- `bench_cases.fut`
  - Shared dataset generators and helper functions.

- `bench_dense_jacobian.fut`
  - Dense JVP/VJP baseline benchmarks.
  - Used as the dense reference point for the project.

- `bench_jvp_structured.fut`
  - Fair CSR-output JVP benchmarks on `banded5` and `stencil`.
  - Compares dense-to-CSR, sparse BGPC, and sparse D2.

- `bench_vjp_structured.fut`
  - Fair CSR-output VJP benchmarks on `banded5` and `stencil`.
  - Compares dense-to-CSR, sparse BGPC, and sparse D2.

- `bench_jvp_spiky.fut`
  - Irregular JVP deep-dive on the synthetic `spiky` family.
  - Contains total, coloring-only, and precolored benchmarks.

## Typical commands

- CPU benchmarks:
  - `make bench`

- GPU benchmarks:
  - `make bench-gpu`

- Single-file runs:
  - `futhark bench benchmark/<file>.fut`
  - `futhark bench --backend=cuda benchmark/<file>.fut`
