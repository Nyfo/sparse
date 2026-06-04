-- Structured JVP benchmarks.
--
-- Dense:
--   dense JVP -> dense Jacobian
--
-- Sparse Compressed:
--   CSR pattern -> coloring -> compressed JVP
--
-- Sparse CSR:
--   CSR pattern -> coloring -> compressed JVP -> CSR vals

module Dense = import "../src/dense_jacobian"
module Sparse = import "../src/sparse_jacobian_jvp"
module BGPC = import "../src/bgpc_vv_coloring"
module D2 = import "../src/partial_d2_coloring"
module Cases = import "./bench_cases"

entry mk_banded_csr_test (m:i64) (n:i64)
  : (i64, i64, [m+1]i64, []i64, [n+1]i64, []i64, [m]i64, [n]f64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_banded5 m n
  let rows : [m]i64 = iota m
  let x : [n]f64 = Cases.rand_vec 42i64
  in (m, n, row_offs, row_idx, col_offs, col_idx, rows, x)

entry mk_stencil_csr_test (h:i64) (w:i64)
  : (i64, i64, [h*w+1]i64, []i64, [h*w+1]i64, []i64, [h*w]f64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_stencil h w
  let x : [h*w]f64 = Cases.rand_vec 42i64
  in (h, w, row_offs, row_idx, col_offs, col_idx, x)

-- ==
-- entry: bench_dense_jvp_banded5
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
-- script input { mk_banded_csr_test 3072 98304 }
-- script input { mk_banded_csr_test 4096 131072 }
entry bench_dense_jvp_banded5 (m:i64) (n:i64)
  (_row_offs:[m+1]i64) (_row_idx:[]i64)
  (_col_offs:[n+1]i64) (_col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : [m][n]f64 =
  Dense.jac_dense_jvp (\x0 -> Cases.f_banded5 rows x0) x

-- ==
-- entry: bench_sparse_jvp_banded5_d2_compressed
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
-- script input { mk_banded_csr_test 3072 98304 }
-- script input { mk_banded_csr_test 4096 131072 }
entry bench_sparse_jvp_banded5_d2_compressed (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : ([n]i64, [][m]f64) =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x
  in (colors, ys)

-- ==
-- entry: bench_sparse_jvp_to_csr_banded5_d2
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
-- script input { mk_banded_csr_test 3072 98304 }
-- script input { mk_banded_csr_test 4096 131072 }
entry bench_sparse_jvp_to_csr_banded5_d2 (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_banded5_bgpc_compressed
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
-- script input { mk_banded_csr_test 3072 98304 }
-- script input { mk_banded_csr_test 4096 131072 }
entry bench_sparse_jvp_banded5_bgpc_compressed (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : ([n]i64, [][m]f64) =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x
  in (colors, ys)

-- ==
-- entry: bench_sparse_jvp_to_csr_banded5_bgpc
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
-- script input { mk_banded_csr_test 3072 98304 }
-- script input { mk_banded_csr_test 4096 131072 }
entry bench_sparse_jvp_to_csr_banded5_bgpc (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.f_banded5 rows x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_dense_jvp_stencil
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
-- script input { mk_stencil_csr_test 160 160 }
-- script input { mk_stencil_csr_test 192 192 }
entry bench_dense_jvp_stencil (h:i64) (w:i64)
  (_row_offs:[h*w+1]i64) (_row_idx:[]i64)
  (_col_offs:[h*w+1]i64) (_col_idx:[]i64)
  (x:[h*w]f64)
  : [h*w][h*w]f64 =
  Dense.jac_dense_jvp (\x0 -> Cases.stencil2d x0) x

-- ==
-- entry: bench_sparse_jvp_stencil_d2_compressed
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
-- script input { mk_stencil_csr_test 160 160 }
-- script input { mk_stencil_csr_test 192 192 }
entry bench_sparse_jvp_stencil_d2_compressed (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : ([h*w]i64, [][h*w]f64) =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x
  in (colors, ys)

-- ==
-- entry: bench_sparse_jvp_to_csr_stencil_d2
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
-- script input { mk_stencil_csr_test 160 160 }
-- script input { mk_stencil_csr_test 192 192 }
entry bench_sparse_jvp_to_csr_stencil_d2 (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : []f64 =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_stencil_bgpc_compressed
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
-- script input { mk_stencil_csr_test 160 160 }
-- script input { mk_stencil_csr_test 192 192 }
entry bench_sparse_jvp_stencil_bgpc_compressed (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : ([h*w]i64, [][h*w]f64) =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x
  in (colors, ys)

-- ==
-- entry: bench_sparse_jvp_to_csr_stencil_bgpc
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
-- script input { mk_stencil_csr_test 160 160 }
-- script input { mk_stencil_csr_test 192 192 }
entry bench_sparse_jvp_to_csr_stencil_bgpc (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : []f64 =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (\x0 -> Cases.stencil2d x0) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
