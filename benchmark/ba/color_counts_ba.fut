module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ba_cases"

def num_colors_of [n] (colors:[n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def ba_d2_colors (num_cams:i64) (num_points:i64) (num_obs:i64) : i64 =
  let (_num_cams, _num_points, _num_obs,
       row_offs, row_idx, col_offs, col_idx,
       _obs, _feat, _x) =
    Cases.mk_ba_csr_data num_cams num_points num_obs

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  in num_colors_of colors

def ba_bgpc_colors (num_cams:i64) (num_points:i64) (num_obs:i64) : i64 =
  let (_num_cams, _num_points, _num_obs,
       row_offs, row_idx, col_offs, col_idx,
       _obs, _feat, _x) =
    Cases.mk_ba_csr_data num_cams num_points num_obs

  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  in num_colors_of colors

entry color_counts_ba : []i64 =
  [ ba_d2_colors 64i64 256i64 8192i64
  , ba_bgpc_colors 64i64 256i64 8192i64

  , ba_d2_colors 96i64 384i64 16384i64
  , ba_bgpc_colors 96i64 384i64 16384i64

  , ba_d2_colors 128i64 512i64 32768i64
  , ba_bgpc_colors 128i64 512i64 32768i64

  , ba_d2_colors 160i64 640i64 40960i64
  , ba_bgpc_colors 160i64 640i64 40960i64
  ]
