# Sparse Jacobian Computation in Futhark

Bachelor's thesis project on sparse Jacobian computation in Futhark.

The project computes Jacobians from a known sparsity pattern using graph coloring and compressed JVP and VJP evaluations. It includes dense baselines, a greedy partial distance-2 coloring pipeline, a BGPC-based coloring pipeline, CSR representations, reusable preprocessing, and automatic selection between forward and reverse mode based on the number of colors.

## Thesis

- **[Read the bachelor's thesis](thesis/Elias_Smedegaard_BSc_Thesis.pdf)**
- [View the thesis defense slides](thesis/Elias_Smedegaard_BSc_Thesis_Defense_Slides.pdf)

The thesis was completed at the University of Copenhagen under the supervision of Troels Henriksen.

## Selected Performance Results

On the largest tested GradBench instances, the greedy partial distance-2 coloring pipeline on the Futhark multicore backend achieved end-to-end speedups of:

- **29.4×** on Bundle Adjustment
- **15.2×** on Hand Tracking

The speedups are relative to the dense multicore baseline. These are selected performance results. See the thesis for the complete results and methodology.

## Requirements to run all tests and benchmarks

- Futhark
- `make`
- CUDA and an NVIDIA GPU for the CUDA tests and benchmarks

On the Hendrix cluster, first load Futhark and CUDA:

```bash
module load futhark
module load cuda
```

Then install the Futhark package dependencies:

```bash
futhark pkg sync
cd benchmark/ba && futhark pkg sync
cd ../ht && futhark pkg sync
cd ../..
```

## Structure

- `src/`: library implementation
- `test/`: correctness tests
- `benchmark/`: benchmark programs
- `results/`: directory for generated benchmark output
- `lib/`: external Futhark dependencies
- `thesis/`: completed bachelor's thesis and thesis defense slides
- `futhark.pkg`: Futhark package file
- `Makefile`: test and benchmark commands

## Main Modules

- `src/dense_jacobian.fut`: dense JVP and VJP baselines
- `src/pattern_csr.fut`: CSR construction from sparsity patterns
- `src/partial_d2_coloring.fut`: greedy partial distance-2 coloring
- `src/bgpc_vv_coloring.fut`: BGPC-style coloring
- `src/sparse_jacobian_jvp.fut`: sparse Jacobian computation using JVPs
- `src/sparse_jacobian_vjp.fut`: sparse Jacobian computation using VJPs
- `src/sparse_jacobian_auto.fut`: automatic selection between the JVP and VJP pipelines

## Benchmarks

The benchmarks cover:

- structured `Banded5` and `Stencil` problems
- `BA` bundle adjustment
- `HT` hand tracking
- coloring and precolored pipeline breakdowns
- ADBench `calculate_jacobian` comparisons

## Commands to run tests and benchmarks

Run all CPU and CUDA tests:

```bash
make test
```

Run only the CPU tests:

```bash
make test-cpu
```

Run only the CUDA tests:

```bash
make test-gpu
```

Run the full benchmark suite:

```bash
make bench
```

The full benchmark suite is intentionally large and can take a long time. The main benchmark groups can also be run individually:

```bash
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
```

Generated benchmark output is written to the `results/` folder.