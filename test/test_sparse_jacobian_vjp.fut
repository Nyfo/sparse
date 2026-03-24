-- Tests for compressed VJP Jacobian using row coloring.

module Dense  = import "../src/dense_jacobian"
module Sparse = import "../src/sparse_jacobian_vjp"

def approx_eq_mat [m][n] (a:[m][n]f64) (b:[m][n]f64) (eps:f64) : bool =
  let row_ok (ra:[n]f64) (rb:[n]f64) : bool =
    reduce (&&) true (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in reduce (&&) true (map2 row_ok a b)

def mask_with_pattern [m][n] (pat:[m][n]bool) (j:[m][n]f64) : [m][n]f64 =
  map2 (\prow jrow ->
          map2 (\p x -> if p then x else 0.0f64) prow jrow)
       pat j

-- Example 1 (3x5)
def f_ex1 (x:[5]f64) : [3]f64 =
  let y0 = x[0] + 2.0f64 * x[3]
  let y1 = 5.0f64 * x[1]
  let y2 = x[2] * x[2]
  in [y0, y1, y2]

def pat_ex1 : [3][5]bool =
  [ [true,  false, false, true,  false]
  , [false, true,  false, false, false]
  , [false, false, true,  false, false]
  ]

-- ==
-- entry: test_sparse_vjp_ex1_matches_dense
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64] }
-- output { true }
entry test_sparse_vjp_ex1_matches_dense (x:[5]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex1 (Dense.jac_dense_vjp f_ex1 x)
  let js = Sparse.jac_vjp_dense f_ex1 pat_ex1 x
  in approx_eq_mat js jd eps

-- Example 2 (2x4)
def f_ex2 (x:[4]f64) : [2]f64 =
  let y0 = x[0] + x[1]
  let y1 = 7.0f64 * x[2]
  in [y0, y1]

def pat_ex2 : [2][4]bool =
  [ [true,  true,  false, false]
  , [false, false, true,  false]
  ]

-- ==
-- entry: test_sparse_vjp_ex2_matches_dense
-- input  { [2.0f64, 3.0f64, 5.0f64, 7.0f64] }
-- output { true }
entry test_sparse_vjp_ex2_matches_dense (x:[4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex2 (Dense.jac_dense_vjp f_ex2 x)
  let js = Sparse.jac_vjp_dense f_ex2 pat_ex2 x
  in approx_eq_mat js jd eps

-- Example 3: zero sparsity pattern / constant function
def f_ex3_zero (_x:[3]f64) : [2]f64 =
  [42.0f64, -7.0f64]

def pat_ex3_zero : [2][3]bool =
  [ [false, false, false]
  , [false, false, false]
  ]

-- ==
-- entry: test_sparse_vjp_zero_pattern_matches_dense
-- input  { [3.0f64, -1.0f64, 9.0f64] }
-- output { true }
entry test_sparse_vjp_zero_pattern_matches_dense (x:[3]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex3_zero (Dense.jac_dense_vjp f_ex3_zero x)
  let js = Sparse.jac_vjp_dense f_ex3_zero pat_ex3_zero x
  in approx_eq_mat js jd eps

-- Example 4: mixed nonlinear dependencies with an empty row
def f_ex4 (x:[6]f64) : [4]f64 =
  let y0 = x[0] * x[1] + 3.0f64 * x[4]
  let y1 = x[2] - x[5] * x[5]
  let y2 = x[1] + x[2] + x[3]
  let y3 = 11.0f64
  in [y0, y1, y2, y3]

def pat_ex4 : [4][6]bool =
  [ [true,  true,  false, false, true,  false]
  , [false, false, true,  false, false, true ]
  , [false, true,  true,  true,  false, false]
  , [false, false, false, false, false, false]
  ]

def row_colors_ex4 : [4]i64 = [0i64, 0i64, 1i64, 0i64]

-- ==
-- entry: test_sparse_vjp_ex4_matches_dense
-- input  { [1.5f64, -2.0f64, 0.5f64, 7.0f64, 3.0f64, -4.0f64] }
-- output { true }
entry test_sparse_vjp_ex4_matches_dense (x:[6]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex4 (Dense.jac_dense_vjp f_ex4 x)
  let js = Sparse.jac_vjp_dense f_ex4 pat_ex4 x
  in approx_eq_mat js jd eps

-- ==
-- entry: test_sparse_vjp_ex4_with_row_colors_matches_dense
-- input  { [1.5f64, -2.0f64, 0.5f64, 7.0f64, 3.0f64, -4.0f64] }
-- output { true }
entry test_sparse_vjp_ex4_with_row_colors_matches_dense (x:[6]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex4 (Dense.jac_dense_vjp f_ex4 x)
  let js = Sparse.jac_vjp_dense_with_row_colors f_ex4 pat_ex4 row_colors_ex4 x
  in approx_eq_mat js jd eps
-- Example 5: another mixed pattern with shared and unused structure
def f_ex5 (x:[5]f64) : [5]f64 =
  let y0 = x[0] * x[0] + x[4]
  let y1 = x[1] * x[3]
  let y2 = x[2] + 2.0f64 * x[4]
  let y3 = x[0] - x[2] * x[2]
  let y4 = -3.0f64
  in [y0, y1, y2, y3, y4]

def pat_ex5 : [5][5]bool =
  [ [true,  false, false, false, true ]
  , [false, true,  false, true,  false]
  , [false, false, true,  false, true ]
  , [true,  false, true,  false, false]
  , [false, false, false, false, false]
  ]

-- ==
-- entry: test_sparse_vjp_ex5_matches_dense
-- input  { [2.0f64, -1.5f64, 3.0f64, 4.0f64, -2.0f64] }
-- output { true }
entry test_sparse_vjp_ex5_matches_dense (x:[5]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_ex5 (Dense.jac_dense_vjp f_ex5 x)
  let js = Sparse.jac_vjp_dense f_ex5 pat_ex5 x
  in approx_eq_mat js jd eps
