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
- GPU:
- Main takeaway:

## Cross-cutting observations
- Dense output vs CSR output:
  - Dense GPU is very strong when the target output is a dense Jacobian.
  - Fair CSR-output benchmarks give sparse methods a more meaningful and often much stronger comparison.
- Structured vs irregular:
  - Structured cases favor the ordered D2 coloring on CPU.
  - GPU performance depends strongly on the sparsity structure and the output format.
- BGPC vs D2:
  - D2 appears to be the stronger coloring algorithm on CPU for structured cases.
  - BGPC appears to be the stronger single-backend GPU pipeline among the sparse methods tested so far.
