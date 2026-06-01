module HT = import "./ht_gradbench_original"
module Cases = import "./ht_cases"

entry mk_ht_adbench_test (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  : (i64, i64, i64,
     [num_bones]i32,
     [num_bones][4][4]f64,
     [num_bones][4][4]f64,
     [num_bones][num_vertices]f64,
     [4][num_vertices]f64,
     [num_vertices][3]i32,
     bool,
     [num_obs]i32,
     [3][num_obs]f64,
     [HT.theta_count]f64,
     [2*num_obs]f64) =
  let (_, _, _,
       _row_offs, _row_idx,
       _col_offs, _col_idx,
       parents,
       base_relatives,
       inverse_base_absolutes,
       weights,
       base_positions,
       triangles,
       is_mirrored,
       correspondences,
       points,
       x) =
    Cases.mk_ht_csr_data num_bones num_obs num_vertices

  let (theta, us) =
    Cases.unpack_ht_x x

  in (num_bones, num_obs, num_vertices,
      parents,
      base_relatives,
      inverse_base_absolutes,
      weights,
      base_positions,
      triangles,
      is_mirrored,
      correspondences,
      points,
      theta,
      us)

-- ==
-- entry: bench_adbench_ht_calculate_jacobian
-- script input { mk_ht_adbench_test 22 512 2048 }
-- script input { mk_ht_adbench_test 22 1024 4096 }
-- script input { mk_ht_adbench_test 22 2048 8192 }
-- script input { mk_ht_adbench_test 22 4096 16384 }
entry bench_adbench_ht_calculate_jacobian
  (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  (parents:[num_bones]i32)
  (base_relatives:[num_bones][4][4]f64)
  (inverse_base_absolutes:[num_bones][4][4]f64)
  (weights:[num_bones][num_vertices]f64)
  (base_positions:[4][num_vertices]f64)
  (triangles:[num_vertices][3]i32)
  (is_mirrored:bool)
  (correspondences:[num_obs]i32)
  (points:[3][num_obs]f64)
  (theta:[HT.theta_count]f64)
  (us:[num_us]f64) =
  HT.calculate_jacobian
    parents
    base_relatives
    inverse_base_absolutes
    weights
    base_positions
    triangles
    is_mirrored
    correspondences
    points
    theta
    us
