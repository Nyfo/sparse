-- Tests for dense Jacobian baselines in src/dense_jacobian.fut.

module Dense = import "../src/dense_jacobian"

def approx_eq_mat [m][n] (a:[m][n]f64) (b:[m][n]f64) (eps:f64) : bool =
  let row_ok (ra:[n]f64) (rb:[n]f64) : bool =
    reduce (&&) true (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in reduce (&&) true (map2 row_ok a b)

-- Wide rectangular case: f : R^5 -> R^3.
def f_wide (x:[5]f64) : [3]f64 =
  let y0 = x[0] + 2.0f64 * x[3]
  let y1 = 5.0f64 * x[1]
  let y2 = x[2] * x[2]
  in [y0, y1, y2]

def jac_wide_expected (x:[5]f64) : [3][5]f64 =
  [ [1.0f64, 0.0f64,       0.0f64, 2.0f64, 0.0f64]
  , [0.0f64, 5.0f64,       0.0f64, 0.0f64, 0.0f64]
  , [0.0f64, 0.0f64, 2.0f64*x[2], 0.0f64, 0.0f64]
  ]

-- Square case with mixed linear and nonlinear dependencies.
def f_square (x:[4]f64) : [4]f64 =
  let y0 = 3.0f64 * x[0]
  let y1 = x[0] * x[1]
  let y2 = x[2] + x[3]
  let y3 = x[1] * x[1]
  in [y0, y1, y2, y3]

def jac_square_expected (x:[4]f64) : [4][4]f64 =
  [ [3.0f64,      0.0f64, 0.0f64, 0.0f64]
  , [x[1],        x[0],   0.0f64, 0.0f64]
  , [0.0f64,      0.0f64, 1.0f64, 1.0f64]
  , [0.0f64, 2.0f64*x[1], 0.0f64, 0.0f64]
  ]

-- Tall rectangular case: f : R^2 -> R^4.
def f_tall (x:[2]f64) : [4]f64 =
  let y0 = x[0]
  let y1 = x[1] * x[1]
  let y2 = x[0] + x[1]
  let y3 = 5.0f64
  in [y0, y1, y2, y3]

def jac_tall_expected (x:[2]f64) : [4][2]f64 =
  [ [1.0f64, 0.0f64]
  , [0.0f64, 2.0f64*x[1]]
  , [1.0f64, 1.0f64]
  , [0.0f64, 0.0f64]
  ]

-- Constant function: the Jacobian should be all zeros.
def f_constant (_x:[3]f64) : [2]f64 =
  [42.0f64, -7.0f64]

def jac_constant_expected : [2][3]f64 =
  [ [0.0f64, 0.0f64, 0.0f64]
  , [0.0f64, 0.0f64, 0.0f64]
  ]

-- ==
-- entry: test_dense_wide_jvp_matches_expected
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64] }
-- output { true }
entry test_dense_wide_jvp_matches_expected (x:[5]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_jvp f_wide x
  in approx_eq_mat j (jac_wide_expected x) eps

-- ==
-- entry: test_dense_wide_vjp_matches_expected
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64] }
-- output { true }
entry test_dense_wide_vjp_matches_expected (x:[5]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_vjp f_wide x
  in approx_eq_mat j (jac_wide_expected x) eps

-- ==
-- entry: test_dense_wide_jvp_equals_vjp
-- input  { [1.25f64, -2.0f64, 0.5f64, 10.0f64, 0.0f64] }
-- output { true }
entry test_dense_wide_jvp_equals_vjp (x:[5]f64) : bool =
  let eps = 1e-9f64
  let j1 = Dense.jac_dense_jvp f_wide x
  let j2 = Dense.jac_dense_vjp f_wide x
  in approx_eq_mat j1 j2 eps

-- ==
-- entry: test_dense_square_jvp_matches_expected
-- input  { [2.0f64, 3.0f64, 5.0f64, 7.0f64] }
-- output { true }
entry test_dense_square_jvp_matches_expected (x:[4]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_jvp f_square x
  in approx_eq_mat j (jac_square_expected x) eps

-- ==
-- entry: test_dense_square_vjp_matches_expected
-- input  { [2.0f64, 3.0f64, 5.0f64, 7.0f64] }
-- output { true }
entry test_dense_square_vjp_matches_expected (x:[4]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_vjp f_square x
  in approx_eq_mat j (jac_square_expected x) eps

-- ==
-- entry: test_dense_tall_jvp_matches_expected
-- input  { [3.0f64, -2.0f64] }
-- output { true }
entry test_dense_tall_jvp_matches_expected (x:[2]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_jvp f_tall x
  in approx_eq_mat j (jac_tall_expected x) eps

-- ==
-- entry: test_dense_tall_vjp_matches_expected
-- input  { [3.0f64, -2.0f64] }
-- output { true }
entry test_dense_tall_vjp_matches_expected (x:[2]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_vjp f_tall x
  in approx_eq_mat j (jac_tall_expected x) eps

-- ==
-- entry: test_dense_constant_jvp_matches_zero
-- input  { [1.0f64, 2.0f64, 3.0f64] }
-- output { true }
entry test_dense_constant_jvp_matches_zero (x:[3]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_jvp f_constant x
  in approx_eq_mat j jac_constant_expected eps

-- ==
-- entry: test_dense_constant_vjp_matches_zero
-- input  { [1.0f64, 2.0f64, 3.0f64] }
-- output { true }
entry test_dense_constant_vjp_matches_zero (x:[3]f64) : bool =
  let eps = 1e-9f64
  let j = Dense.jac_dense_vjp f_constant x
  in approx_eq_mat j jac_constant_expected eps