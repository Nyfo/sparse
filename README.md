# sparse

This repository contains my bachelor thesis project on sparse Jacobian computation in Futhark.

The project looks at how to compute Jacobians more efficiently when the sparsity pattern is known in advance. The main comparison is between dense Jacobian computation and sparse, coloring-based approaches that reduce the number of JVP or VJP evaluations.

## Project Structure

- `src/`: core library code
- `test/`: tests for the library modules
- `benchmark/`: benchmark programs used during the thesis work
- `results/`: saved benchmark outputs and short notes
- `lib/`: external Futhark dependencies

## `src/` Overview

- `src/dense_jacobian.fut`
  Dense reference implementations based on repeated JVP and VJP calls. This is mainly used as a baseline and for correctness checking.

- `src/pattern_csr.fut`
  Utilities for converting dense boolean sparsity patterns into CSR-style sparse representations. This is the connection between a dense pattern description and the sparse pipelines.

- `src/bgpc_vv_coloring.fut`
  A BGPC-style coloring implementation for bipartite row/column sparsity graphs. This is the main GPU-oriented coloring approach in the project.

- `src/partial_d2_coloring.fut`
  A greedy partial distance-2 coloring implementation. This is kept as an alternative coloring strategy and as an important comparison point in the experiments.

- `src/sparse_jacobian_jvp.fut`
  Sparse Jacobian construction using compressed forward-mode evaluations. It supports compressed output, CSR output, and dense reconstruction.

- `src/sparse_jacobian_vjp.fut`
  Sparse Jacobian construction using compressed reverse-mode evaluations. Like the JVP module, it supports compressed output, CSR output, and dense reconstruction.

## Benchmark Structure

The benchmark folder is currently split into two main parts:

- structured benchmarks, mainly based on `banded5` and `stencil`, used for the main dense-vs-sparse and CPU-vs-GPU comparisons
- an irregular benchmark, `spiky`, used more as a deeper stress test and pipeline breakdown case

The benchmark setup is still being refined, so the most stable part of the repository right now is still the `src/` library together with the tests and saved results.

## Basic Commands

Run CPU tests:

```bash
make test
```

Run CUDA tests:

```bash
make test-gpu
```

Run CPU benchmarks:

```bash
make bench
```

Run CUDA benchmarks:

```bash
make bench-gpu
```
