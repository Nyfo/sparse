-- BA JVP benchmarks with precomputed D2 colors.
--
-- These entries separate compressed JVP evaluation and CSR reconstruction from
-- the coloring step.

module Sparse = import "../../src/sparse_jacobian_jvp"
module D2 = import "../../src/partial_d2_coloring"
module Cases = import "./ba_cases"

entry mk_ba_csr_test_with_d2_colors (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64,
     [11*num_cams + 3*num_points + num_obs]i64,
     [num_obs][2]i32, [num_obs][2]f64,
     [11*num_cams + 3*num_points + num_obs]f64) =
  let (num_cams', num_points', num_obs',
       row_offs, row_idx, col_offs, col_idx,
       obs, feat, x) =
    Cases.mk_ba_csr_data num_cams num_points num_obs

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  in (num_cams', num_points', num_obs',
      row_offs, row_idx, colors,
      obs, feat, x)

-- ==
-- entry: bench_precolored_jvp_ba_d2_compressed
-- script input { mk_ba_csr_test_with_d2_colors 32 128 2048 }
-- script input { mk_ba_csr_test_with_d2_colors 48 192 4096 }
-- script input { mk_ba_csr_test_with_d2_colors 64 256 8192 }
-- script input { mk_ba_csr_test_with_d2_colors 80 320 12288 }
entry bench_precolored_jvp_ba_d2_compressed
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (_row_offs:[3*num_obs+1]i64) (_row_idx:[]i64)
  (colors:[11*num_cams + 3*num_points + num_obs]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : [][3*num_obs]f64 =
  Sparse.compressed_ys_jvp
    (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
    colors
    x

-- ==
-- entry: bench_precolored_jvp_ba_d2_csr
-- script input { mk_ba_csr_test_with_d2_colors 32 128 2048 }
-- script input { mk_ba_csr_test_with_d2_colors 48 192 4096 }
-- script input { mk_ba_csr_test_with_d2_colors 64 256 8192 }
-- script input { mk_ba_csr_test_with_d2_colors 80 320 12288 }
entry bench_precolored_jvp_ba_d2_csr
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (colors:[11*num_cams + 3*num_points + num_obs]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : []f64 =
  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> Cases.ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
