-- Fairer VJP benchmark:
-- Compare pipelines that all produce CSR Jacobian values.
--
-- Sparse path:
--   CSR pattern -> coloring -> compressed VJP -> CSR vals
--
-- Dense path:
--   dense VJP -> extract only needed CSR vals
--
-- Active benchmark sizes:
--   banded5: scale both m and n to give the GPU more real work
--   stencil: scale cautiously to avoid dense-memory blowups

module Dense = import "../src/dense_jacobian"
module Sparse = import "../src/sparse_jacobian_vjp"
module BGPC = import "../src/bgpc_vv_coloring"
module D2 = import "../src/partial_d2_coloring"
module Cases = import "./bench_cases"

def dense_to_csr_vals [m][n]
  (row_offs:[m+1]i64)
  (row_idx:[]i64)
  (j:[m][n]f64)
  : []f64 =
  let nnz = length row_idx
  let vals0 = replicate nnz 0.0f64

  let (vals_final, _i) =
    loop (vals, i) = (vals0, 0i64)
    while i < m do
      let s = row_offs[i]
      let e = row_offs[i+1]
      let cols = row_idx[s:e]
      let jrow = j[i]
      let seg = map (\col -> jrow[col]) cols
      let vals' = vals with [s:e] = seg
      in (vals', i + 1i64)

  in vals_final

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
-- entry: bench_dense_vjp_to_csr_banded5
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_dense_vjp_to_csr_banded5 (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (_col_offs:[n+1]i64) (_col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let j = Dense.jac_dense_vjp (\x0 -> Cases.f_banded5 rows x0) x
  in dense_to_csr_vals row_offs row_idx j

-- ==
-- entry: bench_sparse_vjp_to_csr_banded5_bgpc
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_sparse_vjp_to_csr_banded5_bgpc (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let row_colors = BGPC.vv_color_rows row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_vjp (\x0 -> Cases.f_banded5 rows x0) row_colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx row_colors ys

-- ==
-- entry: bench_sparse_vjp_to_csr_banded5_d2
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_sparse_vjp_to_csr_banded5_d2 (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (rows:[m]i64) (x:[n]f64)
  : []f64 =
  let row_colors = D2.partial_d2_color_rows row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_vjp (\x0 -> Cases.f_banded5 rows x0) row_colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx row_colors ys

-- ==
-- entry: bench_dense_vjp_to_csr_stencil
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_dense_vjp_to_csr_stencil (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (_col_offs:[h*w+1]i64) (_col_idx:[]i64)
  (x:[h*w]f64)
  : []f64 =
  let j = Dense.jac_dense_vjp (\x0 -> Cases.stencil2d x0) x
  in dense_to_csr_vals row_offs row_idx j

-- ==
-- entry: bench_sparse_vjp_to_csr_stencil_bgpc
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_sparse_vjp_to_csr_stencil_bgpc (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : []f64 =
  let row_colors = BGPC.vv_color_rows row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_vjp (\x0 -> Cases.stencil2d x0) row_colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx row_colors ys

-- ==
-- entry: bench_sparse_vjp_to_csr_stencil_d2
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_sparse_vjp_to_csr_stencil_d2 (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  (x:[h*w]f64)
  : []f64 =
  let row_colors = D2.partial_d2_color_rows row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_vjp (\x0 -> Cases.stencil2d x0) row_colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx row_colors ys
