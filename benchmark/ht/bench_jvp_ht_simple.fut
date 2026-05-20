module Dense = import "../../src/dense_jacobian"
module Sparse = import "../../src/sparse_jacobian_jvp"
module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ht_cases"
module HT = import "./ht_gradbench_original"

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

-- ==
-- entry: bench_dense_jvp_ht_raw
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
entry bench_dense_jvp_ht_raw
  (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  (_row_offs:[3*num_obs + 1]i64) (_row_idx:[]i64)
  (_col_offs:[HT.theta_count + 2*num_obs + 1]i64) (_col_idx:[]i64)
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

-- ==
-- entry: bench_sparse_jvp_ht_d2_compressed
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
-- script input { mk_ht_csr_test 22 4096 16384 }
-- script input { mk_ht_csr_test 22 8192 32768 }
entry bench_sparse_jvp_ht_d2_compressed
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
  : ([HT.theta_count + 2*num_obs]i64, [][3*num_obs]f64) =
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

  in (colors, ys)

-- ==
-- entry: bench_sparse_jvp_ht_d2_csr
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
-- script input { mk_ht_csr_test 22 4096 16384 }
-- script input { mk_ht_csr_test 22 8192 32768 }
entry bench_sparse_jvp_ht_d2_csr
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

-- ==
-- entry: bench_sparse_jvp_ht_bgpc_compressed
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
-- script input { mk_ht_csr_test 22 4096 16384 }
-- script input { mk_ht_csr_test 22 8192 32768 }
entry bench_sparse_jvp_ht_bgpc_compressed
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
  : ([HT.theta_count + 2*num_obs]i64, [][3*num_obs]f64) =
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

  in (colors, ys)

-- ==
-- entry: bench_sparse_jvp_ht_bgpc_csr
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
-- script input { mk_ht_csr_test 22 4096 16384 }
-- script input { mk_ht_csr_test 22 8192 32768 }
entry bench_sparse_jvp_ht_bgpc_csr
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
