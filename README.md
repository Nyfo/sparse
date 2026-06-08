# sparse

Bachelor thesis project on sparse Jacobian computation in Futhark.

The project computes Jacobians from a known sparsity pattern using graph coloring and compressed JVP/VJP evaluations. The benchmarks compare dense baselines, a greedy partial distance-2 coloring pipeline, and a BGPC-based coloring pipeline.

## Structure

- `src/`: library implementation
- `test/`: correctness tests
- `benchmark/`: benchmark programs
- `results/`: benchmark output notes
- `lib/`: external Futhark dependencies
- `futhark.pkg`: Futhark package file
- `Makefile`: test and benchmark commands

## Main Modules

- `src/dense_jacobian.fut`: dense JVP/VJP baselines
- `src/pattern_csr.fut`: CSR construction from sparsity patterns
- `src/partial_d2_coloring.fut`: greedy partial distance-2 coloring
- `src/bgpc_vv_coloring.fut`: BGPC-style coloring
- `src/sparse_jacobian_jvp.fut`: sparse Jacobian computation using JVPs
- `src/sparse_jacobian_vjp.fut`: sparse Jacobian computation using VJPs
- `src/sparse_jacobian_auto.fut`: direction-selecting wrapper

## Benchmarks

The benchmarks cover:

- structured `Banded5` and `Stencil` problems
- `BA` bundle adjustment
- `HT` hand tracking
- coloring and precolored pipeline breakdowns
- ADBench `calculate_jacobian` comparisons

## Commands

Run all tests:

    make test

Run CPU-only tests:

    make test-cpu

Run GPU-only tests:

    make test-gpu

Run all benchmark groups:

    make bench

CUDA tests and GPU benchmarks require CUDA and an NVIDIA GPU.