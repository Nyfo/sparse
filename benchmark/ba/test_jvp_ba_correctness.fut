module Dense = import "../../src/dense_jacobian"
module Sparse = import "../../src/sparse_jacobian_jvp"
module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ba_cases"

def approx_eq_mat [m][n] (a:[m][n]f64) (b:[m][n]f64) (eps:f64) : bool =
  let row_ok (ra:[n]f64) (rb:[n]f64) : bool =
    reduce (&&) true (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in reduce (&&) true (map2 row_ok a b)

entry mk_ba_csr_test (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64,
     [11*num_cams + 3*num_points + num_obs + 1]i64, []i64,
     [num_obs][2]i32, [num_obs][2]f64,
     [11*num_cams + 3*num_points + num_obs]f64) =
  Cases.mk_ba_csr_data num_cams num_points num_obs

-- ==
-- entry: test_ba_d2_compressed_matches_dense
-- script input { mk_ba_csr_test 4 16 64 }
-- output { true }
-- script input { mk_ba_csr_test 8 32 128 }
-- output { true }
entry test_ba_d2_compressed_matches_dense
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : bool =
  let eps = 1.0e-8f64

  let jd : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Dense.jac_dense_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      x

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  let vals =
    Sparse.compressed_to_csr_vals row_offs row_idx colors ys

  let js : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Sparse.csr_to_dense row_offs row_idx vals

  in approx_eq_mat js jd eps

-- ==
-- entry: test_ba_d2_csr_matches_dense
-- script input { mk_ba_csr_test 4 16 64 }
-- output { true }
-- script input { mk_ba_csr_test 8 32 128 }
-- output { true }
entry test_ba_d2_csr_matches_dense
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : bool =
  let eps = 1.0e-8f64

  let jd : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Dense.jac_dense_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      x

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  let vals =
    Sparse.compressed_to_csr_vals row_offs row_idx colors ys

  let js : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Sparse.csr_to_dense row_offs row_idx vals

  in approx_eq_mat js jd eps

-- ==
-- entry: test_ba_bgpc_compressed_matches_dense
-- script input { mk_ba_csr_test 4 16 64 }
-- output { true }
-- script input { mk_ba_csr_test 8 32 128 }
-- output { true }
entry test_ba_bgpc_compressed_matches_dense
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : bool =
  let eps = 1.0e-8f64

  let jd : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Dense.jac_dense_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      x

  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  let vals =
    Sparse.compressed_to_csr_vals row_offs row_idx colors ys

  let js : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Sparse.csr_to_dense row_offs row_idx vals

  in approx_eq_mat js jd eps

-- ==
-- entry: test_ba_bgpc_csr_matches_dense
-- script input { mk_ba_csr_test 4 16 64 }
-- output { true }
-- script input { mk_ba_csr_test 8 32 128 }
-- output { true }
entry test_ba_bgpc_csr_matches_dense
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : bool =
  let eps = 1.0e-8f64

  let jd : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Dense.jac_dense_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      x

  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  let vals =
    Sparse.compressed_to_csr_vals row_offs row_idx colors ys

  let js : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
    Sparse.csr_to_dense row_offs row_idx vals

  in approx_eq_mat js jd eps
