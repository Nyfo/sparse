module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ht_cases"
module HT = import "./ht_gradbench_original"

entry mk_ht_csr_test (num_bones:i64) (num_obs:i64) (num_vertices:i64)
  : (i64, i64, i64,
     [3*num_obs + 1]i64, []i64,
     [HT.theta_count + 2*num_obs + 1]i64, []i64) =
  let (_num_bones, _num_obs, _num_vertices,
       row_offs, row_idx, col_offs, col_idx,
       _parents,
       _base_relatives,
       _inverse_base_absolutes,
       _weights,
       _base_positions,
       _triangles,
       _is_mirrored,
       _correspondences,
       _points,
       _x) =
    Cases.mk_ht_csr_data num_bones num_obs num_vertices

  in (num_bones, num_obs, num_vertices,
      row_offs, row_idx, col_offs, col_idx)

-- ==
-- entry: bench_color_ht_d2
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
-- script input { mk_ht_csr_test 22 4096 16384 }
-- script input { mk_ht_csr_test 22 8192 32768 }
entry bench_color_ht_d2
  (_num_bones:i64) (num_obs:i64) (_num_vertices:i64)
  (row_offs:[3*num_obs + 1]i64) (row_idx:[]i64)
  (col_offs:[HT.theta_count + 2*num_obs + 1]i64) (col_idx:[]i64)
  : [HT.theta_count + 2*num_obs]i64 =
  D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

-- ==
-- entry: bench_color_ht_bgpc
-- script input { mk_ht_csr_test 22 512 2048 }
-- script input { mk_ht_csr_test 22 2048 8192 }
-- script input { mk_ht_csr_test 22 4096 16384 }
-- script input { mk_ht_csr_test 22 8192 32768 }
entry bench_color_ht_bgpc
  (_num_bones:i64) (num_obs:i64) (_num_vertices:i64)
  (row_offs:[3*num_obs + 1]i64) (row_idx:[]i64)
  (col_offs:[HT.theta_count + 2*num_obs + 1]i64) (col_idx:[]i64)
  : [HT.theta_count + 2*num_obs]i64 =
  BGPC.vv_color_cols row_offs row_idx col_offs col_idx
