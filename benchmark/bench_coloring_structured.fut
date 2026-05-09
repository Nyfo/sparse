-- Coloring-only benchmarks for the structured benchmark cases.
--
-- These use the same CSR sparsity patterns as bench_jvp_structured.fut,
-- but time only the coloring step and report the number of colors.

module D2 = import "../src/partial_d2_coloring"
module BGPC = import "../src/bgpc_vv_coloring"
module Cases = import "./bench_cases"

def num_colors_of [n] (colors: [n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

entry mk_banded_csr_test (m:i64) (n:i64)
  : (i64, i64, [m+1]i64, []i64, [n+1]i64, []i64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_banded5 m n
  in (m, n, row_offs, row_idx, col_offs, col_idx)

entry mk_stencil_csr_test (h:i64) (w:i64)
  : (i64, i64, [h*w+1]i64, []i64, [h*w+1]i64, []i64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_stencil h w
  in (h, w, row_offs, row_idx, col_offs, col_idx)

-- ==
-- entry: bench_color_banded5_d2
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_color_banded5_d2 (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [n]i64 =
  D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

-- ==
-- entry: bench_color_banded5_bgpc
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_color_banded5_bgpc (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [n]i64 =
  BGPC.vv_color_cols row_offs row_idx col_offs col_idx

-- ==
-- entry: bench_num_colors_banded5_d2
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_num_colors_banded5_d2 (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- ==
-- entry: bench_num_colors_banded5_bgpc
-- script input { mk_banded_csr_test 512 16384 }
-- script input { mk_banded_csr_test 1024 32768 }
-- script input { mk_banded_csr_test 2048 65536 }
entry bench_num_colors_banded5_bgpc (m:i64) (n:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- ==
-- entry: bench_color_stencil_d2
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_color_stencil_d2 (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  : [h*w]i64 =
  D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

-- ==
-- entry: bench_color_stencil_bgpc
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_color_stencil_bgpc (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  : [h*w]i64 =
  BGPC.vv_color_cols row_offs row_idx col_offs col_idx

-- ==
-- entry: bench_num_colors_stencil_d2
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_num_colors_stencil_d2 (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors

-- ==
-- entry: bench_num_colors_stencil_bgpc
-- script input { mk_stencil_csr_test 64 64 }
-- script input { mk_stencil_csr_test 96 96 }
-- script input { mk_stencil_csr_test 128 128 }
entry bench_num_colors_stencil_bgpc (h:i64) (w:i64)
  (row_offs:[h*w+1]i64) (row_idx:[]i64)
  (col_offs:[h*w+1]i64) (col_idx:[]i64)
  : i64 =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  in num_colors_of colors
