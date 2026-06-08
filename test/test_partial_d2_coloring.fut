module CSR = import "../src/pattern_csr"
module Col = import "../src/partial_d2_coloring"

def neigh (offs:[]i64) (idx:[]i64) (v:i64) : []i64 =
  let s = offs[v]
  let e = offs[v+1]
  in idx[s:e]

def eq_i64s (xs:[]i64) (ys:[]i64) : bool =
  length xs == length ys && and (map2 (==) xs ys)

def all_distinct (cs:[]i64) : bool =
  let k = length cs
  let (ok_final, _i) =
    loop (ok, i) = (true, 0i64)
    while ok && i < k do
      let ci = cs[i]

      let (ok2_final, _j) =
        loop (ok2, j) = (true, i + 1i64)
        while ok2 && j < k do
          let ok2' = ok2 && (ci != cs[j])
          in (ok2', j + 1i64)

      in (ok2_final, i + 1i64)
  in ok_final

def num_colors_of (colors:[]i64) : i64 =
  if length colors == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def valid_coloring [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (colors:[n]i64) : bool =
  let row_ok (i:i64) : bool =
    let cols = neigh row_offs row_idx i
    let cs = map (\c -> colors[c]) cols
    in all_distinct cs
  in reduce (&&) true (map row_ok (iota m))

def valid_row_coloring [m][n]
  (_row_offs:[m+1]i64) (_row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (colors:[m]i64) : bool =
  let col_ok (j:i64) : bool =
    let rows = neigh col_offs col_idx j
    let cs = map (\r -> colors[r]) rows
    in all_distinct cs
  in reduce (&&) true (map col_ok (iota n))

def pat_ex1 : [3][5]bool =
  [ [true,  false, false, true,  false]
  , [false, true,  false, false, true ]
  , [false, false, true,  false, false]
  ]

-- Normal mixed pattern with a known greedy column coloring.
-- ==
-- entry: test_partial_d2_cols_ex1_exact
-- input  { }
-- output { true }
entry test_partial_d2_cols_ex1_exact : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_ex1
  let colors =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 0i64, 0i64, 1i64, 1i64]
     && valid_coloring row_offs row_idx colors

def pat_empty : [3][4]bool =
  [ [false, false, false, false]
  , [false, false, false, false]
  , [false, false, false, false]
  ]

-- Empty pattern: no conflicts, so all columns can use the same color.
-- ==
-- entry: test_partial_d2_cols_empty_pattern
-- input  { }
-- output { true }
entry test_partial_d2_cols_empty_pattern : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_empty
  let colors =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 0i64, 0i64, 0i64]
     && valid_coloring row_offs row_idx colors

def pat_diag : [4][4]bool =
  [ [true,  false, false, false]
  , [false, true,  false, false]
  , [false, false, true,  false]
  , [false, false, false, true ]
  ]

-- Diagonal pattern: no column conflicts, so all columns can use the same color.
-- ==
-- entry: test_partial_d2_cols_diagonal_pattern
-- input  { }
-- output { true }
entry test_partial_d2_cols_diagonal_pattern : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_diag
  let colors =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 0i64, 0i64, 0i64]
     && valid_coloring row_offs row_idx colors

def pat_dense : [2][3]bool =
  [ [true, true, true]
  , [true, true, true]
  ]

-- Dense pattern: all columns conflict, so they must get distinct colors.
-- ==
-- entry: test_partial_d2_cols_dense_pattern
-- input  { }
-- output { true }
entry test_partial_d2_cols_dense_pattern : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_dense
  let colors =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 1i64, 2i64]
     && valid_coloring row_offs row_idx colors

def pat_star : [1][4]bool =
  [ [true, true, true, true] ]

-- Star pattern: one row contains all columns, forcing distinct column colors.
-- ==
-- entry: test_partial_d2_cols_star_pattern
-- input  { }
-- output { true }
entry test_partial_d2_cols_star_pattern : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_star
  let colors =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 1i64, 2i64, 3i64]
     && valid_coloring row_offs row_idx colors

-- Row coloring on ex1: no rows share a column, so one color is enough.
-- ==
-- entry: test_partial_d2_rows_ex1_valid
-- input  { }
-- output { true }
entry test_partial_d2_rows_ex1_valid : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_ex1
  let colors =
    Col.partial_d2_color_rows row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 0i64, 0i64]
     && valid_row_coloring row_offs row_idx col_offs col_idx colors
     && num_colors_of colors == 1i64

-- Row coloring on dense pattern: all rows conflict, so they need distinct colors.
-- ==
-- entry: test_partial_d2_rows_dense_pattern
-- input  { }
-- output { true }
entry test_partial_d2_rows_dense_pattern : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_dense
  let colors =
    Col.partial_d2_color_rows row_offs row_idx col_offs col_idx
  in eq_i64s colors [0i64, 1i64]
     && valid_row_coloring row_offs row_idx col_offs col_idx colors