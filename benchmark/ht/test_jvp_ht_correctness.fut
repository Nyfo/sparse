module Dense = import "../../src/dense_jacobian"
module Sparse = import "../../src/sparse_jacobian_jvp"
module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ht_cases"
module HT = import "./ht_gradbench_original"

def approx_eq_mat [m][n] (a:[m][n]f64) (b:[m][n]f64) (eps:f64) : bool =
  let row_ok (ra:[n]f64) (rb:[n]f64) : bool =
    reduce (&&) true (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in reduce (&&) true (map2 row_ok a b)

def csr_to_dense_ht [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (vals: []f64)
  : [m][n]f64 =
  Sparse.csr_to_dense row_offs row_idx vals

def mask_ht_pattern [num_obs]
  (j: [3*num_obs][HT.theta_count + 2*num_obs]f64)
  : [3*num_obs][HT.theta_count + 2*num_obs]f64 =
  tabulate (3i64*num_obs) (\r ->
    tabulate (HT.theta_count + 2i64*num_obs) (\c ->
      let q = r / 3i64
      in if c < HT.theta_count ||
            c == HT.theta_count + 2i64*q ||
            c == HT.theta_count + 2i64*q + 1i64
         then j[r, c]
         else 0.0f64))

entry mk_ht_csr_test (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  : (i64, i64, i64,
     [3*num_obs + 1]i64, []i64,
     [HT.theta_count + 2*num_obs + 1]i64, []i64,
     [num_bones]i32,
     [num_bones][4][4]f64,
     [num_bones][4][4]f64,
     [num_bones][num_vertices]f64,
     [4][num_vertices]f64,
     [num_vertices][3]i32,
     bool,
     [num_obs]i32,
     [3][num_obs]f64,
     [HT.theta_count + 2*num_obs]f64) =
  Cases.mk_ht_csr_data num_bones num_obs num_vertices

def dense_ht_jac [num_bones][num_obs][num_vertices]
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (x:[HT.theta_count + 2*num_obs]f64)
  : [3*num_obs][HT.theta_count + 2*num_obs]f64 =
  Dense.jac_dense_jvp
    (\x0 -> Cases.ht_residual_flat
              parents
              base_relatives
              inverse_base_absolutes
              weights
              base_positions
              triangles
              is_mirrored
              correspondences
              points
              x0)
    x

def d2_ht_csr_vals [num_bones][num_obs][num_vertices]
  (row_offs:[3*num_obs + 1]i64)
  (row_idx:[]i64)
  (col_offs:[HT.theta_count + 2*num_obs + 1]i64)
  (col_idx:[]i64)
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (x:[HT.theta_count + 2*num_obs]f64)
  : []f64 =
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ht_residual_flat
                parents
                base_relatives
                inverse_base_absolutes
                weights
                base_positions
                triangles
                is_mirrored
                correspondences
                points
                x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

def bgpc_ht_csr_vals [num_bones][num_obs][num_vertices]
  (row_offs:[3*num_obs + 1]i64)
  (row_idx:[]i64)
  (col_offs:[HT.theta_count + 2*num_obs + 1]i64)
  (col_idx:[]i64)
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (x:[HT.theta_count + 2*num_obs]f64)
  : []f64 =
  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ht_residual_flat
                parents
                base_relatives
                inverse_base_absolutes
                weights
                base_positions
                triangles
                is_mirrored
                correspondences
                points
                x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: test_ht_d2_csr_matches_dense
-- script input { mk_ht_csr_test 22 8 32 }
-- output { true }
-- script input { mk_ht_csr_test 22 16 64 }
-- output { true }
entry test_ht_d2_csr_matches_dense
  (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  (row_offs:[3*num_obs + 1]i64) (row_idx:[]i64)
  (col_offs:[HT.theta_count + 2*num_obs + 1]i64) (col_idx:[]i64)
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (x:[HT.theta_count + 2*num_obs]f64)
  : bool =
  let eps = 1.0e-8f64

  let dense =
    mask_ht_pattern (dense_ht_jac
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      x)

  let vals =
    d2_ht_csr_vals
      row_offs
      row_idx
      col_offs
      col_idx
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      x

  let sparse =
    csr_to_dense_ht row_offs row_idx vals

  in approx_eq_mat dense sparse eps

-- ==
-- entry: test_ht_bgpc_csr_matches_dense
-- script input { mk_ht_csr_test 22 8 32 }
-- output { true }
-- script input { mk_ht_csr_test 22 16 64 }
-- output { true }
entry test_ht_bgpc_csr_matches_dense
  (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  (row_offs:[3*num_obs + 1]i64) (row_idx:[]i64)
  (col_offs:[HT.theta_count + 2*num_obs + 1]i64) (col_idx:[]i64)
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (x:[HT.theta_count + 2*num_obs]f64)
  : bool =
  let eps = 1.0e-8f64

  let dense =
    mask_ht_pattern (dense_ht_jac
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      x)

  let vals =
    bgpc_ht_csr_vals
      row_offs
      row_idx
      col_offs
      col_idx
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      x

  let sparse =
    csr_to_dense_ht row_offs row_idx vals

  in approx_eq_mat dense sparse eps
