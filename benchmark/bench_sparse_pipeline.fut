-- Bench the sparse *pipeline overhead* only:
--  1) Build CSR from a dense boolean pattern
--  2) Greedy partial distance-2 coloring from CSR
--
-- Note (based on your measurements):
-- - CSR build from dense pattern: best on CUDA
-- - Greedy coloring: best on C (CUDA is very slow here)

module CSR   = import "../src/pattern_csr"
module Col   = import "../src/partial_d2_coloring"
module Cases = import "./bench_cases"

-- Datasets:

-- For CSR-build benchmark: return (m,n,pat) so entries can take explicit sizes.
--
-- ==
-- entry: mk_pat_test
entry mk_pat_test (m:i64) (n:i64) : (i64, i64, [m][n]bool) =
  let pat : [m][n]bool = Cases.mk_pat_banded5 m n
  in (m, n, pat)

-- Benchmarks:

-- ==
-- entry: mk_csr_test
entry mk_csr_test (m:i64) (n:i64)
  : (i64, i64, [m+1]i64, []i64, [n+1]i64, []i64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_banded5 m n
  in (m, n, row_offs, row_idx, col_offs, col_idx)

-- section: (1) build CSR from dense pattern
-- Best backend: CUDA
--
-- ==
-- entry: bench_build_csr_from_dense_pattern
-- script input { mk_pat_test 256 2048 }
-- script input { mk_pat_test 256 4096 }
-- script input { mk_pat_test 256 8192 }
entry bench_build_csr_from_dense_pattern (m:i64) (n:i64) (pat:[m][n]bool) : i64 =
  let ((row_offs, _row_idx), (_col_offs, _col_idx)) =
    CSR.csr_bipartite_from_pattern pat
  in row_offs[m]

-- section: (2) greedy coloring from CSR
-- Best backend: C
--
-- ==
-- entry: bench_color_from_csr
-- script input { mk_csr_test 256 2048 }
-- script input { mk_csr_test 256 4096 }
-- script input { mk_csr_test 256 8192 }
entry bench_color_from_csr (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [n]i64 =
  Col.partial_d2_color_cols row_offs row_idx col_offs col_idx