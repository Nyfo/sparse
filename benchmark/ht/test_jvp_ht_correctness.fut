-- End-to-end HT correctness tests for the JVP sparse pipelines.
--
-- These tests compare the sparse HT pipelines against a dense Jacobian,
-- restricted to the structurally nonzero entries of the HT sparsity pattern.

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

def has (xs:[]i64) (x:i64) : bool =
  reduce (||) false (map (== x) xs)

def mask_with_csr_pattern [m][n]
  (row_offs:[m+1]i64)
  (row_idx:[]i64)
  (j:[m][n]f64)
  : [m][n]f64 =
  tabulate m (\r ->
    let s = row_offs[r]
    let e = row_offs[r+1]
    let cols = row_idx[s:e]
    in tabulate n (\c ->
         if has cols c then j[r,c] else 0.0f64))

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

def ht_dense_masked [num_bones][num_obs][num_vertices]
  (row_offs:[3*num_obs + 1]i64)
  (row_idx:[]i64)
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
  let jd =
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

  in mask_with_csr_pattern row_offs row_idx jd

def ht_sparse_from_colors [num_bones][num_obs][num_vertices]
  (row_offs:[3*num_obs + 1]i64)
  (row_idx:[]i64)
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (colors:[HT.theta_count + 2*num_obs]i64)
  (x:[HT.theta_count + 2*num_obs]f64)
  : [3*num_obs][HT.theta_count + 2*num_obs]f64 =
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

  let vals =
    Sparse.compressed_to_csr_vals row_offs row_idx colors ys

  in Sparse.csr_to_dense row_offs row_idx vals

-- D2 on HT: checks coloring, compressed JVP, CSR reconstruction, and values.
-- ==
-- entry: test_ht_d2_jvp_matches_dense
-- script input { mk_ht_csr_test 22 7 29 }
-- output { true }
-- script input { mk_ht_csr_test 22 8 32 }
-- output { true }
-- script input { mk_ht_csr_test 22 16 64 }
-- output { true }
entry test_ht_d2_jvp_matches_dense
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
  let eps = 1.0e-9f64

  let jd =
    ht_dense_masked
      row_offs
      row_idx
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

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let js =
    ht_sparse_from_colors
      row_offs
      row_idx
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      colors
      x

  in approx_eq_mat js jd eps

-- BGPC on HT: same end-to-end check with the alternative coloring algorithm.
-- ==
-- entry: test_ht_bgpc_jvp_matches_dense
-- script input { mk_ht_csr_test 22 7 29 }
-- output { true }
-- script input { mk_ht_csr_test 22 8 32 }
-- output { true }
-- script input { mk_ht_csr_test 22 16 64 }
-- output { true }
entry test_ht_bgpc_jvp_matches_dense
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
  let eps = 1.0e-9f64

  let jd =
    ht_dense_masked
      row_offs
      row_idx
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

  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let js =
    ht_sparse_from_colors
      row_offs
      row_idx
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      colors
      x

  in approx_eq_mat js jd eps