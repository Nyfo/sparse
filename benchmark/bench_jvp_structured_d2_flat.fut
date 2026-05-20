module Sparse = import "../src/sparse_jacobian_jvp"
module D2 = import "../src/partial_d2_coloring_flattened"
module Cases = import "./bench_cases"

entry mk_banded_csr_test (m:i64) (n:i64)
  : (i64, i64, [m+1]i64, []i64, [n+1]i64, []i64, [m]i64, [n]f64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_banded5 m n
  let rows : [m]i64 = iota m
  let x : [n]f64 = Cases.rand_vec 42i64
  in (m, n, row_offs, row_idx, col_offs, col_idx, rows, x)

entry mk_stencil_csr_test (h:i64) (w:i64)
  : (i64, i64, [h*w+1]i64, []i64, [h*w+1]i64, []i64, [h*w]f64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_stencil h w
  let x : [h*w]f64 = Cases.rand_vec 42i64
  in (h, w, row_offs, row_idx, col_offs, col_idx, x)

-- ==
-- entry: bench_sparse_jvp_to_csr_banded5_d2_flat
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_sparse_jvp_to_csr_banded5_d2_flat (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys =
    Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_to_csr_stencil_d2_flat
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_sparse_jvp_to_csr_stencil_d2_flat (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : []f64 =
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys =
    Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
