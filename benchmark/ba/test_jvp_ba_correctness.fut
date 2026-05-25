-- End-to-end BA correctness tests for the JVP sparse pipelines.
--
-- These tests compare the sparse BA pipelines against a dense Jacobian,
-- restricted to the structurally nonzero entries of the BA sparsity pattern.

module Dense = import "../../src/dense_jacobian"
module Sparse = import "../../src/sparse_jacobian_jvp"
module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ba_cases"

def approx_eq_mat [m][n] (a:[m][n]f64) (b:[m][n]f64) (eps:f64) : bool =
  let row_ok (ra:[n]f64) (rb:[n]f64) : bool =
    reduce (&&) true (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in reduce (&&) true (map2 row_ok a b)

def mask_with_pattern [m][n] (pat:[m][n]bool) (j:[m][n]f64) : [m][n]f64 =
  map2 (\prow jrow ->
          map2 (\p x -> if p then x else 0.0f64) prow jrow)
       pat j

entry mk_ba_csr_test (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64,
     [11*num_cams + 3*num_points + num_obs + 1]i64, []i64,
     [num_obs][2]i32, [num_obs][2]f64,
     [11*num_cams + 3*num_points + num_obs]f64) =
  Cases.mk_ba_csr_data num_cams num_points num_obs

def ba_dense_masked [num_obs]
  (num_cams:i64) (num_points:i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
  let jd =
    Dense.jac_dense_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      x

  let pat =
    Cases.pat_ba num_cams num_points obs

  in mask_with_pattern pat jd

def ba_sparse_from_colors [num_obs]
  (num_cams:i64) (num_points:i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (colors:[11*num_cams + 3*num_points + num_obs]i64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : [3*num_obs][11*num_cams + 3*num_points + num_obs]f64 =
  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  let vals =
    Sparse.compressed_to_csr_vals row_offs row_idx colors ys

  in Sparse.csr_to_dense row_offs row_idx vals

-- D2 on BA: checks coloring, compressed JVP, CSR reconstruction, and values.
-- ==
-- entry: test_ba_d2_jvp_matches_dense
-- script input { mk_ba_csr_test 3 7 20 }
-- output { true }
-- script input { mk_ba_csr_test 4 16 64 }
-- output { true }
-- script input { mk_ba_csr_test 8 32 128 }
-- output { true }
entry test_ba_d2_jvp_matches_dense
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : bool =
  let eps = 1.0e-9f64

  let jd =
    ba_dense_masked num_cams num_points obs feat x

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let js =
    ba_sparse_from_colors num_cams num_points obs feat row_offs row_idx colors x

  in approx_eq_mat js jd eps

-- BGPC on BA: same end-to-end check with the alternative coloring algorithm.
-- ==
-- entry: test_ba_bgpc_jvp_matches_dense
-- script input { mk_ba_csr_test 3 7 20 }
-- output { true }
-- script input { mk_ba_csr_test 4 16 64 }
-- output { true }
-- script input { mk_ba_csr_test 8 32 128 }
-- output { true }
entry test_ba_bgpc_jvp_matches_dense
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : bool =
  let eps = 1.0e-9f64

  let jd =
    ba_dense_masked num_cams num_points obs feat x

  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let js =
    ba_sparse_from_colors num_cams num_points obs feat row_offs row_idx colors x

  in approx_eq_mat js jd eps