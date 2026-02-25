-- Dense Jacobian baselines for f : [n]f64 -> [m]f64

-- One-hot vector
def onehot_f64 [n] (i:i64) : [n]f64 =
  replicate n 0.0f64 with [i] = 1.0f64

-- Dense via JVP (forward mode) with n calls
def jac_dense_jvp [m][n] (f:[n]f64 -> [m]f64) (x:[n]f64) : [m][n]f64 =
  let cols : [n][m]f64 = map (\j -> jvp f x (onehot_f64 j)) (iota n)
  in transpose cols

-- Dense via VJP (reverse mode) with m calls
def jac_dense_vjp [m][n] (f:[n]f64 -> [m]f64) (x:[n]f64) : [m][n]f64 =
  map (\i -> vjp f x (onehot_f64 i)) (iota m)
