module Dense = import "../../src/dense_jacobian"
module Sparse = import "../../src/sparse_jacobian_jvp"
module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ba_cases"


def csr_row_ids [m][nnz]
  (row_offs:[m+1]i64)
  (_row_idx:[nnz]i64)
  : [nnz]i64 =
  let rows0 = replicate nnz 0i64

  let (rows, _i) =
    loop (acc, i) = (rows0, 0i64)
    while i < m do
      let s = row_offs[i]
      let e = row_offs[i+1]
      let len = e - s
      let seg = replicate len i
      let acc' = acc with [s:e] = seg
      in (acc', i + 1i64)

  in rows

def dense_to_csr_vals [m][n][nnz]
  (row_ids:[nnz]i64)
  (row_idx:[nnz]i64)
  (j:[m][n]f64)
  : [nnz]f64 =
  map2 (\i col -> j[i, col]) row_ids row_idx

entry mk_ba_csr_test_with_row_ids (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64, []i64,
     [11*num_cams + 3*num_points + num_obs + 1]i64, []i64,
     [num_obs][2]i32, [num_obs][2]f64,
     [11*num_cams + 3*num_points + num_obs]f64) =
  let (num_cams', num_points', num_obs',
       row_offs, row_idx, col_offs, col_idx,
       obs, feat, x) =
    Cases.mk_ba_csr_data num_cams num_points num_obs

  let row_ids =
    csr_row_ids row_offs row_idx

  in (num_cams', num_points', num_obs',
      row_offs, row_ids, row_idx,
      col_offs, col_idx,
      obs, feat, x)


entry mk_ba_csr_test (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64,
     [11*num_cams + 3*num_points + num_obs + 1]i64, []i64,
     [num_obs][2]i32, [num_obs][2]f64,
     [11*num_cams + 3*num_points + num_obs]f64) =
  Cases.mk_ba_csr_data num_cams num_points num_obs

-- ==
-- entry: bench_dense_jvp_to_csr_ba
-- script input { mk_ba_csr_test_with_row_ids 64 256 8192 }
-- script input { mk_ba_csr_test_with_row_ids 96 384 16384 }
-- script input { mk_ba_csr_test_with_row_ids 128 512 32768 }
-- script input { mk_ba_csr_test_with_row_ids 160 640 40960 }
entry bench_dense_jvp_to_csr_ba [nnz]
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (_row_offs:[3*num_obs+1]i64) (row_ids:[nnz]i64) (row_idx:[nnz]i64)
  (_col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (_col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : [nnz]f64 =
  let j =
    Dense.jac_dense_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      x
  in dense_to_csr_vals row_ids row_idx j


-- ==
-- entry: bench_sparse_jvp_to_csr_ba_d2
-- script input { mk_ba_csr_test 64 256 8192 }
-- script input { mk_ba_csr_test 96 384 16384 }
-- script input { mk_ba_csr_test 128 512 32768 }
-- script input { mk_ba_csr_test 160 640 40960 }
entry bench_sparse_jvp_to_csr_ba_d2 (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : []f64 =
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_to_csr_ba_bgpc
-- script input { mk_ba_csr_test 64 256 8192 }
-- script input { mk_ba_csr_test 96 384 16384 }
-- script input { mk_ba_csr_test 128 512 32768 }
-- script input { mk_ba_csr_test 160 640 40960 }
entry bench_sparse_jvp_to_csr_ba_bgpc (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : []f64 =
  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

