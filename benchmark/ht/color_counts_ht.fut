module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module Cases = import "./ht_cases"

def num_colors_of [n] (colors:[n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def ht_d2_colors (num_bones:i64) (num_obs:i64) (num_vertices:i64) : i64 =
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

  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  in num_colors_of colors

def ht_bgpc_colors (num_bones:i64) (num_obs:i64) (num_vertices:i64) : i64 =
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

  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  in num_colors_of colors

entry color_counts_ht : []i64 =
  [ ht_d2_colors 22i64 512i64 2048i64
  , ht_bgpc_colors 22i64 512i64 2048i64

  , ht_d2_colors 22i64 1024i64 4096i64
  , ht_bgpc_colors 22i64 1024i64 4096i64

  , ht_d2_colors 22i64 2048i64 8192i64
  , ht_bgpc_colors 22i64 2048i64 8192i64

  , ht_d2_colors 22i64 4096i64 16384i64
  , ht_bgpc_colors 22i64 4096i64 16384i64
  ]
