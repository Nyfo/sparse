module CSR = import "./pattern_csr"
module Col = import "./bgpc_vv_coloring"
module JVP = import "./sparse_jacobian_jvp"
module VJP = import "./sparse_jacobian_vjp"

-- Assumes colors are zero-based and contiguous, as produced by greedy coloring.
def num_colors_of [k] (colors: [k]i64) : i64 =
  if k == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

-- Computes both column and row colorings from the sparsity pattern,
-- counts the colors, and decides whether JVP or VJP should be used.
--
-- use_jvp = true  => choose JVP
-- use_jvp = false => choose VJP
def prepare_jac_auto [m][n]
  (pat: [m][n]bool)
  : ([n]i64, [m]i64, i64, i64, bool) =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let col_colors =
    Col.vv_color_cols row_offs row_idx col_offs col_idx

  let row_colors =
    Col.vv_color_rows row_offs row_idx col_offs col_idx

  let num_col_colors = num_colors_of col_colors
  let num_row_colors = num_colors_of row_colors

  let use_jvp = num_col_colors <= num_row_colors

  in (col_colors, row_colors, num_col_colors, num_row_colors, use_jvp)

-- Returns only the chosen mode and coloring info.
def jac_auto_choice [m][n]
  (pat: [m][n]bool)
  : (bool, i64, i64) =
  let (_col_colors, _row_colors, num_col_colors, num_row_colors, use_jvp) =
    prepare_jac_auto pat
  in (use_jvp, num_col_colors, num_row_colors)

-- Sparse / CSR output
def jac_auto_csr [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64) =
  let (col_colors, row_colors, _num_col_colors, _num_row_colors, use_jvp) =
    prepare_jac_auto pat
  in if use_jvp
     then JVP.jac_jvp_csr_with_colors f pat col_colors x
     else VJP.jac_vjp_csr_with_row_colors f pat row_colors x

-- Sparse / CSR output with metadata
def jac_auto_csr_with_info [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64) =
  let (col_colors, row_colors, num_col_colors, num_row_colors, use_jvp) =
    prepare_jac_auto pat

  let (row_offs, row_idx, vals) =
    if use_jvp
    then JVP.jac_jvp_csr_with_colors f pat col_colors x
    else VJP.jac_vjp_csr_with_row_colors f pat row_colors x

  in ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors)

-- Dense output
def jac_auto_dense [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64)
  : [m][n]f64 =
  let (col_colors, row_colors, _num_col_colors, _num_row_colors, use_jvp) =
    prepare_jac_auto pat
  in if use_jvp
     then JVP.jac_jvp_dense_with_colors f pat col_colors x
     else VJP.jac_vjp_dense_with_row_colors f pat row_colors x

-- Dense output with metadata
def jac_auto_dense_with_info [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64)
  : ([m][n]f64, bool, i64, i64) =
  let (col_colors, row_colors, num_col_colors, num_row_colors, use_jvp) =
    prepare_jac_auto pat

  let jac =
    if use_jvp
    then JVP.jac_jvp_dense_with_colors f pat col_colors x
    else VJP.jac_vjp_dense_with_row_colors f pat row_colors x

  in (jac, use_jvp, num_col_colors, num_row_colors)