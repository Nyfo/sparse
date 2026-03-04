-- Sparse compressed-VJP benchmarks on the banded-5 case.

module Sparse = import "../src/sparse_jacobian_vjp"
module Cases  = import "./bench_cases"

-- Datasets:

entry mk_banded_test (m:i64) (n:i64)
  : (i64, i64, [m][n]bool, [m]i64, [n]f64) =
  let (pat, rows, x) = Cases.mk_banded5_inputs m n
  in (m, n, pat, rows, x)

entry mk_banded_test_with_row_colors (m:i64) (n:i64)
  : (i64, i64, [m][n]bool, [m]i64, [m]i64, [n]f64) =
  let (pat, row_colors, rows, x) = Cases.mk_banded5_inputs_with_row_colors m n
  in (m, n, pat, row_colors, rows, x)

-- Benchmarks:

-- ==
-- entry: bench_sparse_vjp_banded5_total
-- script input { mk_banded_test 256 2048 }
-- script input { mk_banded_test 256 4096 }
-- script input { mk_banded_test 256 8192 }
entry bench_sparse_vjp_banded5_total (m:i64) (n:i64)
  (pat:[m][n]bool) (rows:[m]i64) (x:[n]f64)
  : [m][n]f64 =
  Sparse.jac_compressed_vjp (\x0 -> Cases.f_banded5 rows x0) pat x

-- ==
-- entry: bench_sparse_vjp_banded5_precolored
-- script input { mk_banded_test_with_row_colors 256 2048 }
-- script input { mk_banded_test_with_row_colors 256 4096 }
-- script input { mk_banded_test_with_row_colors 256 8192 }
entry bench_sparse_vjp_banded5_precolored (m:i64) (n:i64)
  (pat:[m][n]bool) (row_colors:[m]i64) (rows:[m]i64) (x:[n]f64)
  : [m][n]f64 =
  Sparse.jac_compressed_vjp_with_row_colors
    (\x0 -> Cases.f_banded5 rows x0) pat row_colors x