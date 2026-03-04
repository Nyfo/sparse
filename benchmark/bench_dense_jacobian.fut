-- Dense Jacobian benchmarks on the banded5 case.

module DJ    = import "../src/dense_jacobian"
module Cases = import "./bench_cases"

-- Dataset generator

entry mk_banded_test (m:i64) (n:i64) : (i64, i64, [m]i64, [n]f64) =
  let rows : [m]i64 = iota m
  let x    : [n]f64 = Cases.rand_vec 42i64
  in (m, n, rows, x)

-- Dense Jacobian via JVP

-- ==
-- entry: bench_dense_jvp_banded5
-- script input { mk_banded_test 256 2048 }
-- script input { mk_banded_test 256 4096 }
-- script input { mk_banded_test 256 8192 }
entry bench_dense_jvp_banded5 (m:i64) (n:i64) (rows:[m]i64) (x:[n]f64) : [m][n]f64 =
  DJ.jac_dense_jvp (\x0 -> Cases.f_banded5 rows x0) x

-- section: Dense Jacobian via VJP

-- ==
-- entry: bench_dense_vjp_banded5
-- script input { mk_banded_test 256 2048 }
-- script input { mk_banded_test 256 4096 }
-- script input { mk_banded_test 256 8192 }
entry bench_dense_vjp_banded5 (m:i64) (n:i64) (rows:[m]i64) (x:[n]f64) : [m][n]f64 =
  DJ.jac_dense_vjp (\x0 -> Cases.f_banded5 rows x0) x