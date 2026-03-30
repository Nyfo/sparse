# Benchmark Notes

## JVP Structured
- CPU:
  - D2 + CPU is clearly the fastest method on both banded5 and stencil.
  - BGPC + CPU is much faster than dense, but still clearly behind D2.
- GPU:
  - BGPC + GPU beats dense + GPU on banded5.
  - Dense + GPU beats BGPC + GPU on stencil.
  - D2 + GPU is excluded from routine GPU benchmarking because it is already known to be noncompetitive.
- Main takeaway:
  - On structured JVP benchmarks, D2 + CPU is the strongest overall method, while BGPC + GPU is the only sparse GPU method that remains competitive.

## VJP Structured
- CPU:
  - D2 + CPU is clearly the fastest method on both banded5 and stencil.
  - BGPC + CPU is strong, but still behind D2.
- GPU:
  - BGPC + GPU slightly beats or matches dense + GPU on banded5.
  - Dense + GPU beats BGPC + GPU on stencil.
  - D2 + GPU is excluded from routine GPU benchmarking because it is already known to be noncompetitive.
- Main takeaway:
  - The VJP structured benchmarks show the same pattern as JVP: D2 + CPU dominates on structured cases, while BGPC + GPU is the only meaningful sparse GPU alternative.

## JVP Spiky
- CPU:
  - D2 + CPU total is 670.7 ms at 8192x131072 and 4057.8 ms at 16384x262144.
  - D2 coloring on CPU is only 11.9 ms and 74.9 ms on the same sizes.
  - This means D2 coloring is only about 1.8 percent of the full D2 CPU pipeline on the measured spiky cases.
- GPU:
  - BGPC + GPU total is 918.1 ms, 2909.1 ms, and 12370.2 ms on the measured spiky sizes.
  - BGPC coloring on GPU is 316.5 ms, 1593.6 ms, and 9604.9 ms.
  - This means GPU coloring grows from about 34 percent of total at the smallest measured spiky size to about 78 percent at the largest measured spiky size.
  - Dense GPU was only benchmarked on smaller spiky sizes, so it is useful as context but not directly comparable to the largest sparse runs.
- Main takeaway:
  - On the irregular spiky family, GPU coloring is the main bottleneck in the BGPC pipeline.
  - D2 + CPU is faster at the smaller measured spiky size, but BGPC + GPU overtakes it as the problem grows.
  - This suggests that the crossover between CPU and GPU sparse pipelines depends strongly on problem size and irregularity.

## Cross-cutting observations
- Dense output vs CSR output:
  - Dense GPU is very strong when the target output is a dense Jacobian.
  - Fair CSR-output benchmarks give sparse methods a more meaningful and often much stronger comparison.
- Structured vs irregular:
  - Structured cases favor the ordered D2 coloring on CPU.
  - Irregular cases make GPU coloring much more important and can shift the balance toward BGPC + GPU at larger sizes.
- BGPC vs D2:
  - D2 appears to be the stronger coloring algorithm on CPU for structured cases and has very cheap preprocessing cost on the measured spiky cases.
  - BGPC appears to be the stronger single-backend GPU pipeline among the sparse methods tested so far, but its GPU coloring cost is the main weakness on irregular cases.
