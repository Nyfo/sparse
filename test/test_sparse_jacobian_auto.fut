-- Tests for automatic sparse Jacobian mode selection (JVP vs VJP).

module Dense = import "../src/dense_jacobian"
module Auto  = import "../src/sparse_jacobian_auto"

def approx_eq_mat [m][n] (a:[m][n]f64) (b:[m][n]f64) (eps:f64) : bool =
  let row_ok (ra:[n]f64) (rb:[n]f64) : bool =
    reduce (&&) true (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in reduce (&&) true (map2 row_ok a b)

def mask_with_pattern [m][n] (pat:[m][n]bool) (j:[m][n]f64) : [m][n]f64 =
  map2 (\prow jrow ->
          map2 (\p x -> if p then x else 0.0f64) prow jrow)
       pat j

def csr_to_dense [m][n]
  (row_offs:[m+1]i64)
  (row_idx:[]i64)
  (vals:[]f64)
  : [m][n]f64 =
  map (\i ->
        let s = row_offs[i]
        let e = row_offs[i+1]
        let cols = row_idx[s:e]
        let vs   = vals[s:e]
        let row0 : [n]f64 = replicate n 0.0f64
        in scatter row0 cols vs)
      (iota m)

-- Example 1: JVP should be preferred
-- Columns need 2 colors, rows need 4 colors.
def f_ex1 (x:[4]f64) : [4]f64 =
  let y0 = x[0] + 2.0f64 * x[1]
  let y1 = x[0] * x[2]
  let y2 = x[0] - x[3] * x[3]
  let y3 = 5.0f64 * x[0]
  in [y0, y1, y2, y3]

def pat_ex1 : [4][4]bool =
  [ [true,  true,  false, false]
  , [true,  false, true,  false]
  , [true,  false, false, true ]
  , [true,  false, false, false]
  ]

-- ==
-- entry: test_sparse_auto_ex1_dense_with_info
-- input  { [2.0f64, -1.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_ex1_dense_with_info (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex1 (Dense.jac_dense_jvp f_ex1 x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_dense_with_info f_ex1 pat_ex1 x
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 2i64
     && num_row_colors == 4i64

-- ==
-- entry: test_sparse_auto_ex1_csr_with_info
-- input  { [2.0f64, -1.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_ex1_csr_with_info (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex1 (Dense.jac_dense_jvp f_ex1 x)
  let ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_csr_with_info f_ex1 pat_ex1 x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 2i64
     && num_row_colors == 4i64

-- Example 2: VJP should be preferred
-- Columns need 4 colors, rows need 2 colors.
def f_ex2 (x:[4]f64) : [4]f64 =
  let y0 = x[0] + x[1] + x[2] + x[3]
  let y1 = 3.0f64 * x[0]
  let y2 = x[1] * x[1]
  let y3 = 4.0f64 * x[2]
  in [y0, y1, y2, y3]

def pat_ex2 : [4][4]bool =
  [ [true,  true,  true,  true ]
  , [true,  false, false, false]
  , [false, true,  false, false]
  , [false, false, true,  false]
  ]

-- ==
-- entry: test_sparse_auto_ex2_dense_with_info
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_ex2_dense_with_info (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex2 (Dense.jac_dense_jvp f_ex2 x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_dense_with_info f_ex2 pat_ex2 x
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 4i64
     && num_row_colors == 2i64

-- ==
-- entry: test_sparse_auto_ex2_csr_with_info
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_ex2_csr_with_info (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex2 (Dense.jac_dense_jvp f_ex2 x)
  let ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_csr_with_info f_ex2 pat_ex2 x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 4i64
     && num_row_colors == 2i64

-- Example 3: tie, so JVP should be chosen
-- Both columns and rows need 1 color.
def f_ex3 (x:[5]f64) : [3]f64 =
  let y0 = 2.0f64 * x[0]
  let y1 = x[2] * x[2]
  let y2 = x[4] - 1.0f64
  in [y0, y1, y2]

def pat_ex3 : [3][5]bool =
  [ [true,  false, false, false, false]
  , [false, false, true,  false, false]
  , [false, false, false, false, true ]
  ]

-- ==
-- entry: test_sparse_auto_ex3_tie_prefers_jvp
-- input  { [1.0f64, 7.0f64, 3.0f64, 9.0f64, -2.0f64] }
-- output { true }
entry test_sparse_auto_ex3_tie_prefers_jvp (x:[5]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex3 (Dense.jac_dense_jvp f_ex3 x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_dense_with_info f_ex3 pat_ex3 x
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 1i64
     && num_row_colors == 1i64

-- ==
-- entry: test_sparse_auto_ex3_csr_tie_prefers_jvp
-- input  { [1.0f64, 7.0f64, 3.0f64, 9.0f64, -2.0f64] }
-- output { true }
entry test_sparse_auto_ex3_csr_tie_prefers_jvp (x:[5]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex3 (Dense.jac_dense_jvp f_ex3 x)
  let ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_csr_with_info f_ex3 pat_ex3 x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 1i64
     && num_row_colors == 1i64

-- Example 4: zero pattern / constant function
def f_ex4_zero (_x:[4]f64) : [2]f64 =
  [10.0f64, -3.0f64]

def pat_ex4_zero : [2][4]bool =
  [ [false, false, false, false]
  , [false, false, false, false]
  ]

-- ==
-- entry: test_sparse_auto_zero_pattern_dense
-- input  { [8.0f64, -2.0f64, 5.0f64, 11.0f64] }
-- output { true }
entry test_sparse_auto_zero_pattern_dense (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex4_zero (Dense.jac_dense_jvp f_ex4_zero x)
  let ja = Auto.jac_auto_dense f_ex4_zero pat_ex4_zero x
  in approx_eq_mat ja jd eps

-- ==
-- entry: test_sparse_auto_zero_pattern_csr
-- input  { [8.0f64, -2.0f64, 5.0f64, 11.0f64] }
-- output { true }
entry test_sparse_auto_zero_pattern_csr (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex4_zero (Dense.jac_dense_jvp f_ex4_zero x)
  let (row_offs, row_idx, vals) = Auto.jac_auto_csr f_ex4_zero pat_ex4_zero x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps

-- Example 5: mixed nonlinear case with empty row and unused column
-- VJP should be preferred here: columns need 3 colors, rows need 2.
def f_ex5 (x:[6]f64) : [5]f64 =
  let y0 = x[0] * x[1] + x[5]
  let y1 = 2.0f64 * x[2]
  let y2 = x[1] + x[3]
  let y3 = x[0] - x[5] * x[5]
  let y4 = 11.0f64
  in [y0, y1, y2, y3, y4]

def pat_ex5 : [5][6]bool =
  [ [true,  true,  false, false, false, true ]
  , [false, false, true,  false, false, false]
  , [false, true,  false, true,  false, false]
  , [true,  false, false, false, false, true ]
  , [false, false, false, false, false, false]
  ]

-- ==
-- entry: test_sparse_auto_ex5_dense_with_info
-- input  { [2.0f64, -3.0f64, 4.0f64, 1.5f64, 99.0f64, -2.0f64] }
-- output { true }
entry test_sparse_auto_ex5_dense_with_info (x:[6]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex5 (Dense.jac_dense_jvp f_ex5 x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_dense_with_info f_ex5 pat_ex5 x
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 3i64
     && num_row_colors == 2i64

-- ==
-- entry: test_sparse_auto_ex5_csr_with_info
-- input  { [2.0f64, -3.0f64, 4.0f64, 1.5f64, 99.0f64, -2.0f64] }
-- output { true }
entry test_sparse_auto_ex5_csr_with_info (x:[6]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex5 (Dense.jac_dense_jvp f_ex5 x)
  let ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    Auto.jac_auto_csr_with_info f_ex5 pat_ex5 x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 3i64
     && num_row_colors == 2i64