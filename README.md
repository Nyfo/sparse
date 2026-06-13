# sparse

Bachelor thesis project on sparse Jacobian computation in Futhark.

The project computes Jacobians from a known sparsity pattern using graph coloring and compressed JVP/VJP evaluations. The benchmarks compare dense baselines, a greedy partial distance-2 coloring pipeline, and a BGPC-based coloring pipeline.

## Requirements to run all tests and benchmarks

- Futhark (run `module load futhark` on the Hendrix cluster)
- `make`
- CUDA and an NVIDIA GPU (run `module load cuda` on the Hendrix cluster)

On the Hendrix cluster, first load Futhark and CUDA:

    module load futhark
    module load cuda

Then install the Futhark package dependencies:

    futhark pkg sync
    cd benchmark/ba && futhark pkg sync
    cd ../ht && futhark pkg sync
    cd ../..

## Structure:

- `src/`: library implementation
- `test/`: correctness tests
- `benchmark/`: benchmark programs
- `results/`: benchmark output notes
- `lib/`: external Futhark dependencies
- `futhark.pkg`: Futhark package file
- `Makefile`: test and benchmark commands

## Main Modules:

- `src/dense_jacobian.fut`: dense JVP/VJP baselines
- `src/pattern_csr.fut`: CSR construction from sparsity patterns
- `src/partial_d2_coloring.fut`: greedy partial distance-2 coloring
- `src/bgpc_vv_coloring.fut`: BGPC-style coloring
- `src/sparse_jacobian_jvp.fut`: sparse Jacobian computation using JVPs
- `src/sparse_jacobian_vjp.fut`: sparse Jacobian computation using VJPs
- `src/sparse_jacobian_auto.fut`: direction-selecting wrapper

## Benchmarks:

The benchmarks cover:

- structured `Banded5` and `Stencil` problems
- `BA` bundle adjustment
- `HT` hand tracking
- coloring and precolored pipeline breakdowns
- ADBench `calculate_jacobian` comparisons

## Commands to run tests and benchmarks:

Run all tests. This runs both CPU tests and CUDA tests:

    make test

Run only the CPU tests:

    make test-cpu

Run only the CUDA tests:

    make test-gpu

Run all benchmark groups:

    make bench

The full benchmark suite is intentionally large and can take a long time. The
main benchmark groups can also be run individually:

    make bench-structured-cpu
    make bench-vjp-structured-cpu
    make bench-ba-cpu
    make bench-ht-cpu
    make bench-structured-gpu
    make bench-vjp-structured-gpu
    make bench-ba-gpu
    make bench-ht-gpu
    make bench-coloring
    make bench-precolored
    make bench-adbench-cpu
    make bench-adbench-gpu

Generated benchmark output is written to the `results/` folder.
