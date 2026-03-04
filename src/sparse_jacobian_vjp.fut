-- Compressed Jacobian via VJP using ROW coloring from a user-provided sparsity pattern.

module CSR = import "./pattern_csr"
module Col = import "./partial_d2_coloring"

def num_colors_of [k] (colors:[k]i64) : i64 =
  if k == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def seed_for_row_color [m] (row_colors:[m]i64) (c:i64) : [m]f64 =
  map (\rc -> if rc == c then 1.0f64 else 0.0f64) row_colors

def decompress_rows [m][n]
  (pat:[m][n]bool)
  (row_colors:[m]i64)
  (ys:[][n]f64)
  : [m][n]f64 =
  map (\i ->
        map (\j ->
              if pat[i][j]
              then ys[row_colors[i]][j]
              else 0.0f64)
            (iota n))
      (iota m)

def jac_compressed_vjp [m][n]
  (f:[n]f64 -> [m]f64)
  (pat:[m][n]bool)
  (x:[n]f64)
  : [m][n]f64 =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let row_colors : [m]i64 =
    Col.partial_d2_color_rows row_offs row_idx col_offs col_idx

  let nc : i64 = num_colors_of row_colors

  let ys : [nc][n]f64 =
    map (\c ->
          let seed = seed_for_row_color row_colors c
          in vjp f x seed)
        (iota nc)

  in decompress_rows pat row_colors ys

-- Like jac_compressed_vjp, but assumes row_colors are already computed
def jac_compressed_vjp_with_row_colors [m][n]
  (f:[n]f64 -> [m]f64)
  (pat:[m][n]bool)
  (row_colors:[m]i64)
  (x:[n]f64)
  : [m][n]f64 =
  let nc : i64 = num_colors_of row_colors
  let ys : [nc][n]f64 =
    map (\c ->
          let seed = seed_for_row_color row_colors c
          in vjp f x seed)
        (iota nc)
  in decompress_rows pat row_colors ys