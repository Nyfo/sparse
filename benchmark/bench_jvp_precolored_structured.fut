-- These separate compressed JVP evaluation and CSR reconstruction from
-- the coloring step.

module Sparse = import "../src/sparse_jacobian_jvp"
module D2 = import "../src/partial_d2_coloring"
module Cases = import "./bench_cases"

entry mk_banded_csr_test_with_d2_colors (m:i64) (n:i64)
  : (i64, i64, [m+1]i64, []i64, [n]i64, [m]i64, [n]f64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_banded5 m n
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let rows : [m]i64 = iota m
  let x : [n]f64 = Cases.rand_vec 42i64
  in (m, n, row_offs, row_idx, colors, rows, x)

entry mk_stencil_csr_test_with_d2_colors (h:i64) (w:i64)
  : (i64, i64, [h*w+1]i64, []i64, [h*w]i64, [h*w]f64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_stencil h w
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let x : [h*w]f64 = Cases.rand_vec 42i64
  in (h, w, row_offs, row_idx, colors, x)

-- ==
-- entry: bench_precolored_jvp_banded5_d2_compressed
-- script input { mk_banded_csr_test_with_d2_colors 512 16384 }
-- script input { mk_banded_csr_test_with_d2_colors 1024 32768 }
-- script input { mk_banded_csr_test_with_d2_colors 2048 65536 }
entry bench_precolored_jvp_banded5_d2_compressed (m:i64) (n:i64)
  (_row_offs:[m+1]i64) (_row_idx:[]i64)
  (colors:[n]i64)
  (rows:[m]i64) (x:[n]f64)
  : [][m]f64 =
  Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x

-- ==
-- entry: bench_precolored_jvp_banded5_d2_csr
-- script input { mk_banded_csr_test_with_d2_colors 512 16384 }
-- script input { mk_banded_csr_test_with_d2_colors 1024 32768 }
-- script input { mk_banded_csr_test_with_d2_colors 2048 65536 }
entry bench_precolored_jvp_banded5_d2_csr (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (colors:[n]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_precolored_jvp_stencil_d2_compressed
-- script input { mk_stencil_csr_test_with_d2_colors 64 64 }
-- script input { mk_stencil_csr_test_with_d2_colors 96 96 }
-- script input { mk_stencil_csr_test_with_d2_colors 128 128 }
entry bench_precolored_jvp_stencil_d2_compressed (h:i64) (w:i64)
  (_row_offs:[h*w+1]i64) (_row_idx:[]i64)
  (colors:[h*w]i64)
  (x:[h*w]f64)
  : [][h*w]f64 =
  Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x

-- ==
-- entry: bench_precolored_jvp_stencil_d2_csr
-- script input { mk_stencil_csr_test_with_d2_colors 64 64 }
-- script input { mk_stencil_csr_test_with_d2_colors 96 96 }
-- script input { mk_stencil_csr_test_with_d2_colors 128 128 }
entry bench_precolored_jvp_stencil_d2_csr (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (colors:[h*w]i64)
  (x:[h*w]f64)
  : []f64 =
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
