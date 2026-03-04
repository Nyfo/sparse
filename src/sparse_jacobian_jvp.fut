-- Compressed Jacobian via JVP using column coloring from a user-provided sparsity pattern.

module CSR = import "./pattern_csr"
module Col = import "./partial_d2_coloring"

def num_colors_of [n] (colors:[n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def seed_for_color [n] (colors:[n]i64) (c:i64) : [n]f64 =
  map (\colc -> if colc == c then 1.0f64 else 0.0f64) colors

def decompress_cols [m][n]
  (pat:[m][n]bool)
  (colors:[n]i64)
  (ys:[][m]f64)
  : [m][n]f64 =
  map (\i ->
        map (\j ->
              if pat[i][j]
              then ys[colors[j]][i]
              else 0.0f64)
            (iota n))
      (iota m)

def jac_compressed_jvp [m][n]
  (f:[n]f64 -> [m]f64)
  (pat:[m][n]bool)
  (x:[n]f64)
  : [m][n]f64 =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let colors : [n]i64 =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let nc : i64 = num_colors_of colors

  let ys : [nc][m]f64 =
    map (\c ->
          let seed = seed_for_color colors c
          in jvp f x seed)
        (iota nc)

  in decompress_cols pat colors ys

-- Like jac_compressed_jvp, but assumes colors are already computed
def jac_compressed_jvp_with_colors [m][n]
  (f:[n]f64 -> [m]f64)
  (pat:[m][n]bool)
  (colors:[n]i64)
  (x:[n]f64)
  : [m][n]f64 =
  let nc : i64 = num_colors_of colors
  let ys : [nc][m]f64 =
    map (\c ->
          let seed = seed_for_color colors c
          in jvp f x seed)
        (iota nc)
  in decompress_cols pat colors ys