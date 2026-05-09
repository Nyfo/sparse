-- Color counts for the structured benchmark cases.
--
-- Output order:
--   banded5 512x16384:   D2, BGPC
--   banded5 1024x32768:  D2, BGPC
--   banded5 2048x65536:  D2, BGPC
--   stencil 64x64:       D2, BGPC
--   stencil 96x96:       D2, BGPC
--   stencil 128x128:     D2, BGPC

module D2 = import "../src/partial_d2_coloring"
module BGPC = import "../src/bgpc_vv_coloring"
module Cases = import "./bench_cases"

def num_colors_of [n] (colors: [n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def banded5_d2_colors (m:i64) (n:i64) : i64 =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_banded5 m n
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

def banded5_bgpc_colors (m:i64) (n:i64) : i64 =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_banded5 m n
  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

def stencil_d2_colors (h:i64) (w:i64) : i64 =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_stencil h w
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

def stencil_bgpc_colors (h:i64) (w:i64) : i64 =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_stencil h w
  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

entry color_counts_structured : []i64 =
  [ banded5_d2_colors 512i64 16384i64
  , banded5_bgpc_colors 512i64 16384i64
  , banded5_d2_colors 1024i64 32768i64
  , banded5_bgpc_colors 1024i64 32768i64
  , banded5_d2_colors 2048i64 65536i64
  , banded5_bgpc_colors 2048i64 65536i64

  , stencil_d2_colors 64i64 64i64
  , stencil_bgpc_colors 64i64 64i64
  , stencil_d2_colors 96i64 96i64
  , stencil_bgpc_colors 96i64 96i64
  , stencil_d2_colors 128i64 128i64
  , stencil_bgpc_colors 128i64 128i64
  ]
