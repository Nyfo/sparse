-- Bench the sparse preprocessing / coloring pipeline:
--  1) Build CSR from a dense boolean pattern
--  2) Greedy partial distance-2 coloring from CSR
--  3) BGPC V-V coloring from CSR
--
-- Includes both:
--  banded5 case
--  stencil case

module CSR   = import "../src/pattern_csr"
module Col   = import "../src/partial_d2_coloring"
module VV    = import "../src/bgpc_vv_coloring"
module Cases = import "./bench_cases"

-- Helper function:
def num_colors_of [n] (colors:[n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors


-- Datasets: banded5

-- ==
-- entry: mk_pat_test
entry mk_pat_test (m:i64) (n:i64) : (i64, i64, [m][n]bool) =
  let pat : [m][n]bool = Cases.mk_pat_banded5 m n
  in (m, n, pat)

-- ==
-- entry: mk_csr_test
entry mk_csr_test (m:i64) (n:i64)
  : (i64, i64, [m+1]i64, []i64, [n+1]i64, []i64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_banded5 m n
  in (m, n, row_offs, row_idx, col_offs, col_idx)


-- Datasets: stencil

-- ==
-- entry: mk_stencil_pat_test
entry mk_stencil_pat_test (h:i64) (w:i64)
  : (i64, i64, [h*w][h*w]bool) =
  let pat : [h*w][h*w]bool = Cases.pat_stencil2d
  in (h, w, pat)

-- ==
-- entry: mk_stencil_csr_test
entry mk_stencil_csr_test (h:i64) (w:i64)
  : (i64, i64, [h*w+1]i64, []i64, [h*w+1]i64, []i64) =
  let (row_offs, row_idx, col_offs, col_idx) = Cases.mk_csr_stencil h w
  in (h, w, row_offs, row_idx, col_offs, col_idx)

-- -- ------------------------------------------------------------
-- -- (1) Build CSR from dense pattern: banded5
-- -- ------------------------------------------------------------

-- -- ==
-- -- entry: bench_build_csr_from_dense_pattern
-- -- script input { mk_pat_test 256 2048 }
-- -- script input { mk_pat_test 256 4096 }
-- -- script input { mk_pat_test 256 8192 }
-- entry bench_build_csr_from_dense_pattern (m:i64) (n:i64) (pat:[m][n]bool) : i64 =
--   let ((row_offs, _row_idx), (_col_offs, _col_idx)) =
--     CSR.csr_bipartite_from_pattern pat
--   in row_offs[m]

-- ------------------------------------------------------------
-- (2) Old greedy coloring from CSR: banded5
-- ------------------------------------------------------------

-- ==
-- entry: bench_color_from_csr_greedy
-- script input { mk_csr_test 512 512 }
-- script input { mk_csr_test 1024 1024 }
-- script input { mk_csr_test 2048 2048 }
entry bench_color_from_csr_greedy (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- ------------------------------------------------------------
-- (3) V-V coloring from CSR: banded5
-- ------------------------------------------------------------

-- ==
-- entry: bench_color_from_csr_vv
-- script input { mk_csr_test 512 512 }
-- script input { mk_csr_test 1024 1024 }
-- script input { mk_csr_test 2048 2048 }
entry bench_color_from_csr_vv (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = VV.vv_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- -- ------------------------------------------------------------
-- -- (4) Build CSR from dense pattern: stencil
-- -- ------------------------------------------------------------

-- -- ==
-- -- entry: bench_build_csr_from_dense_pattern_stencil
-- -- script input { mk_stencil_pat_test 16 16 }
-- -- script input { mk_stencil_pat_test 32 32 }
-- -- script input { mk_stencil_pat_test 64 64 }
-- entry bench_build_csr_from_dense_pattern_stencil (h:i64) (w:i64)
--   (pat:[h*w][h*w]bool) : i64 =
--   let ((row_offs, _row_idx), (_col_offs, _col_idx)) =
--     CSR.csr_bipartite_from_pattern pat
--   in row_offs[h*w]

-- ------------------------------------------------------------
-- (5) Old greedy coloring from CSR: stencil
-- ------------------------------------------------------------

-- ==
-- entry: bench_color_from_csr_greedy_stencil
-- script input { mk_stencil_csr_test 16 16 }
-- script input { mk_stencil_csr_test 32 32 }
-- script input { mk_stencil_csr_test 64 64 }
entry bench_color_from_csr_greedy_stencil (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- ------------------------------------------------------------
-- (6) V-V coloring from CSR: stencil
-- ------------------------------------------------------------

-- ==
-- entry: bench_color_from_csr_vv_stencil
-- script input { mk_stencil_csr_test 16 16 }
-- script input { mk_stencil_csr_test 32 32 }
-- script input { mk_stencil_csr_test 64 64 }
entry bench_color_from_csr_vv_stencil (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = VV.vv_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- ------------------------------------------------------------
-- Datasets: spiky rows
-- ------------------------------------------------------------

-- ==
-- entry: mk_csr_spiky_test
entry mk_csr_spiky_test
  (m:i64) (n:i64) (small_deg:i64) (big_deg:i64) (num_big_rows:i64)
  : (i64, i64, i64, i64, i64, [m+1]i64, []i64, [n+1]i64, []i64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_spiky_rows m n small_deg big_deg num_big_rows
  in (m, n, small_deg, big_deg, num_big_rows,
      row_offs, row_idx, col_offs, col_idx)

-- -- ------------------------------------------------------------
-- -- Old greedy coloring from CSR: spiky rows
-- -- ------------------------------------------------------------

-- -- ==
-- -- entry: bench_color_from_csr_greedy_spiky
-- -- script input { mk_csr_spiky_test 2048 32768 5 256 16 }
-- -- script input { mk_csr_spiky_test 4096 65536 5 512 32 }
-- -- script input { mk_csr_spiky_test 8192 131072 5 1024 64 }
-- entry bench_color_from_csr_greedy_spiky
--   (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
--   (row_offs:[m+1]i64) (row_idx:[]i64)
--   (col_offs:[n+1]i64) (col_idx:[]i64)
--   : i64 =
--   let colors = Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
--   in num_colors_of colors

-- -- ------------------------------------------------------------
-- -- V-V coloring from CSR: spiky rows
-- -- ------------------------------------------------------------

-- -- ==
-- -- entry: bench_color_from_csr_vv_spiky
-- -- script input { mk_csr_spiky_test 2048 32768 5 256 16 }
-- -- script input { mk_csr_spiky_test 4096 65536 5 512 32 }
-- -- script input { mk_csr_spiky_test 8192 131072 5 1024 64 }
-- entry bench_color_from_csr_vv_spiky
--   (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
--   (row_offs:[m+1]i64) (row_idx:[]i64)
--   (col_offs:[n+1]i64) (col_idx:[]i64)
--   : i64 =
--   let colors = VV.vv_color_cols row_offs row_idx col_offs col_idx
--   in num_colors_of colors