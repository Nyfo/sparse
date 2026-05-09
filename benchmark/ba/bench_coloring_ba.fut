module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ba_cases"

entry mk_ba_csr_test (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64,
     [11*num_cams + 3*num_points + num_obs + 1]i64, []i64) =
  let (_num_cams, _num_points, _num_obs,
       row_offs, row_idx, col_offs, col_idx,
       _obs, _feat, _x) =
    Cases.mk_ba_csr_data num_cams num_points num_obs

  in (num_cams, num_points, num_obs,
      row_offs, row_idx, col_offs, col_idx)

-- ==
-- entry: bench_color_ba_d2
-- script input { mk_ba_csr_test 64 256 8192 }
-- script input { mk_ba_csr_test 96 384 16384 }
-- script input { mk_ba_csr_test 128 512 32768 }
-- script input { mk_ba_csr_test 160 640 40960 }
entry bench_color_ba_d2
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  : [11*num_cams + 3*num_points + num_obs]i64 =
  D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

-- ==
-- entry: bench_color_ba_bgpc
-- script input { mk_ba_csr_test 64 256 8192 }
-- script input { mk_ba_csr_test 96 384 16384 }
-- script input { mk_ba_csr_test 128 512 32768 }
-- script input { mk_ba_csr_test 160 640 40960 }
entry bench_color_ba_bgpc
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  : [11*num_cams + 3*num_points + num_obs]i64 =
  BGPC.vv_color_cols row_offs row_idx col_offs col_idx
