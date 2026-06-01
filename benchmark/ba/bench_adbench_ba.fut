module BA = import "./ba_gradbench_original"
module Cases = import "./ba_cases"

entry mk_ba_adbench_test (num_cams:i64) (num_points:i64) (num_obs:i64) =
  let (_, _, _,
       _row_offs, _row_idx,
       _col_offs, _col_idx,
       obs, feat, x) =
    Cases.mk_ba_csr_data num_cams num_points num_obs

  let (cams, points, weights) =
    Cases.unpack_ba_x num_cams num_points num_obs x

  in (cams, points, weights, obs, feat)

-- ==
-- entry: bench_adbench_ba_calculate_jacobian
-- script input { mk_ba_adbench_test 32 128 2048 }
-- script input { mk_ba_adbench_test 48 192 4096 }
-- script input { mk_ba_adbench_test 64 256 8192 }
-- script input { mk_ba_adbench_test 80 320 12288 }
entry bench_adbench_ba_calculate_jacobian [num_cams][num_points][num_obs]
  (cams:[num_cams][11]f64)
  (points:[num_points][3]f64)
  (weights:[num_obs]f64)
  (obs:[num_obs][2]i32)
  (feat:[num_obs][2]f64) =
  BA.calculate_jacobian cams points weights obs feat