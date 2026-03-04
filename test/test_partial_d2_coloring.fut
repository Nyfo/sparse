-- Tests for Partial D2 coloring

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
  let ok = true
  let i  = 0i64
  let (ok_final, _i) =
    loop (ok, i)
    while ok && i < k do
      let ci = cs[i]

      let ok2 = true
      let j   = i + 1i64
      let (ok2_final, _j) =
        loop (ok2, j)
        while ok2 && j < k do
          let ok2' = ok2 && (ci != cs[j])
          in (ok2', j + 1i64)

      in (ok2_final, i + 1i64)
  in ok_final

def valid_coloring [m][n] (row_offs:[m+1]i64) (row_idx:[]i64) (colors:[n]i64) : bool =
  let row_ok (i:i64) : bool =
    let cols = neigh row_offs row_idx i
    let cs   = map (\c -> colors[c]) cols
    in all_distinct cs
  in reduce (&&) true (map row_ok (iota m))

def pat_ex1 : [3][5]bool =
  [ [true,  false, false, true,  false]
  , [false, true,  false, false, true ]
  , [false, false, true,  false, false]
  ]

def colors_ex1_expected : [5]i64 = [0i64, 0i64, 0i64, 1i64, 1i64]

-- ==
-- entry: test_partial_d2_ex1_exact
-- input  { }
-- output { true }
entry test_partial_d2_ex1_exact : bool =
  let (row_offs, row_idx) = CSR.csr_rows_from_pattern pat_ex1
  let (col_offs, col_idx) = CSR.csr_cols_from_pattern pat_ex1
  let colors = Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors colors_ex1_expected
     && valid_coloring row_offs row_idx colors

def pat_star : [1][4]bool =
  [ [true, true, true, true] ]

def colors_star_expected : [4]i64 = [0i64, 1i64, 2i64, 3i64]

-- ==
-- entry: test_partial_d2_star_exact
-- input  { }
-- output { true }
entry test_partial_d2_star_exact : bool =
  let (row_offs, row_idx) = CSR.csr_rows_from_pattern pat_star
  let (col_offs, col_idx) = CSR.csr_cols_from_pattern pat_star
  let colors = Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in eq_i64s colors colors_star_expected
     && valid_coloring row_offs row_idx colors
