-- Sparse compressed-JVP benchmarks on the banded-5 case

module Sparse = import "../src/sparse_jacobian_jvp"
module Cases  = import "./bench_cases"

-- Datasets:

entry mk_banded_test (m:i64) (n:i64)
  : (i64, i64, [m][n]bool, [m]i64, [n]f64) =
  let (pat, rows, x) = Cases.mk_banded5_inputs m n
  in (m, n, pat, rows, x)

entry mk_banded_test_with_colors (m:i64) (n:i64)
  : (i64, i64, [m][n]bool, [n]i64, [m]i64, [n]f64) =
  let (pat, colors, rows, x) = Cases.mk_banded5_inputs_with_colors m n
  in (m, n, pat, colors, rows, x)

-- Benchmarks:

-- ==
-- entry: bench_sparse_jvp_banded5_total
-- script input { mk_banded_test 256 2048 }
-- script input { mk_banded_test 256 4096 }
-- script input { mk_banded_test 256 8192 }
entry bench_sparse_jvp_banded5_total (m:i64) (n:i64)
  (pat:[m][n]bool) (rows:[m]i64) (x:[n]f64)
  : [m][n]f64 =
  Sparse.jac_jvp_dense (\x0 -> Cases.f_banded5 rows x0) pat x

-- ==
-- entry: bench_sparse_jvp_banded5_precolored
-- script input { mk_banded_test_with_colors 256 2048 }
-- script input { mk_banded_test_with_colors 256 4096 }
-- script input { mk_banded_test_with_colors 256 8192 }
entry bench_sparse_jvp_banded5_precolored (m:i64) (n:i64)
  (pat:[m][n]bool) (colors:[n]i64) (rows:[m]i64) (x:[n]f64)
  : [m][n]f64 =
  Sparse.jac_jvp_dense_with_colors (\x0 -> Cases.f_banded5 rows x0) pat colors x

-- section: Dataset generators for stencil

entry mk_stencil_test (h:i64) (w:i64)
  : (i64, i64, [h*w][h*w]bool, [h*w]f64) =
  let (pat, x) = Cases.mk_stencil_inputs h w
  in (h, w, pat, x)

entry mk_stencil_test_with_colors (h:i64) (w:i64)
  : (i64, i64, [h*w][h*w]bool, [h*w]i64, [h*w]f64) =
  let (pat, colors, x) = Cases.mk_stencil_inputs_with_colors h w
  in (h, w, pat, colors, x)

-- section: Sparse JVP on stencil

-- ==
-- entry: bench_sparse_jvp_stencil_total
-- script input { mk_stencil_test 16 16 }
-- script input { mk_stencil_test 32 32 }
-- script input { mk_stencil_test 64 64 }
entry bench_sparse_jvp_stencil_total (h:i64) (w:i64)
  (pat:[h*w][h*w]bool) (x:[h*w]f64)
  : [h*w][h*w]f64 =
  Sparse.jac_jvp_dense (\x0 -> Cases.stencil2d x0) pat x

-- ==
-- entry: bench_sparse_jvp_stencil_precolored
-- script input { mk_stencil_test_with_colors 16 16 }
-- script input { mk_stencil_test_with_colors 32 32 }
-- script input { mk_stencil_test_with_colors 64 64 }
entry bench_sparse_jvp_stencil_precolored (h:i64) (w:i64)
  (pat:[h*w][h*w]bool) (colors:[h*w]i64) (x:[h*w]f64)
  : [h*w][h*w]f64 =
  Sparse.jac_jvp_dense_with_colors (\x0 -> Cases.stencil2d x0) pat colors x