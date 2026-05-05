module CSR = import "./pattern_csr"
module Col = import "./bgpc_vv_coloring"
module JVP = import "./sparse_jacobian_jvp"
module VJP = import "./sparse_jacobian_vjp"

-- Assumes colors are zero-based and contiguous, as produced by greedy coloring.
def num_colors_of [k] (colors: [k]i64) : i64 =
  if k == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

-- Precompute the structure-dependent part of the automatic sparse Jacobian pipeline.
-- This computes CSR structure, both colorings, the number of colors, and whether
-- JVP or VJP should be used.
--
-- use_jvp = true  => choose JVP
-- use_jvp = false => choose VJP
def prepare_jac_auto [m][n]
  (pat: [m][n]bool)
  : ([m+1]i64, []i64, [n+1]i64, []i64,
     [n]i64, [m]i64, i64, i64, bool) =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let col_colors =
    Col.vv_color_cols row_offs row_idx col_offs col_idx

  let row_colors =
    Col.vv_color_rows row_offs row_idx col_offs col_idx

  let num_col_colors = num_colors_of col_colors
  let num_row_colors = num_colors_of row_colors

  let use_jvp = num_col_colors <= num_row_colors

  in (row_offs, row_idx, col_offs, col_idx,
      col_colors, row_colors,
      num_col_colors, num_row_colors, use_jvp)


-- --------------------- Prepared auto pipeline ---------------------

-- Return the sparse Jacobian in CSR format using prepared structure/coloring.
def eval_prepared_auto_csr [m][n]
  (f: [n]f64 -> [m]f64)
  (prepared: ([m+1]i64, []i64, [n+1]i64, []i64,
              [n]i64, [m]i64, i64, i64, bool))
  (x: [n]f64) =
  let (row_offs, row_idx, col_offs, col_idx,
       col_colors, row_colors,
       _num_col_colors, _num_row_colors, use_jvp) = prepared

  let jvp_prepared = (row_offs, row_idx, col_offs, col_idx, col_colors)
  let vjp_prepared = (row_offs, row_idx, col_offs, col_idx, row_colors)

  in if use_jvp
     then JVP.eval_prepared_jvp_csr f jvp_prepared x
     else VJP.eval_prepared_vjp_csr f vjp_prepared x

-- Return the sparse Jacobian in CSR format with metadata.
def eval_prepared_auto_csr_with_info [m][n]
  (f: [n]f64 -> [m]f64)
  (prepared: ([m+1]i64, []i64, [n+1]i64, []i64,
              [n]i64, [m]i64, i64, i64, bool))
  (x: [n]f64) =
  let (_row_offs, _row_idx, _col_offs, _col_idx,
       _col_colors, _row_colors,
       num_col_colors, num_row_colors, use_jvp) = prepared

  let (row_offs, row_idx, vals) =
    eval_prepared_auto_csr f prepared x

  in ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors)

-- Return a dense Jacobian using prepared structure/coloring.
def eval_prepared_auto_dense [m][n]
  (f: [n]f64 -> [m]f64)
  (prepared: ([m+1]i64, []i64, [n+1]i64, []i64,
              [n]i64, [m]i64, i64, i64, bool))
  (x: [n]f64)
  : [m][n]f64 =
  let (row_offs, row_idx, col_offs, col_idx,
       col_colors, row_colors,
       _num_col_colors, _num_row_colors, use_jvp) = prepared

  let jvp_prepared = (row_offs, row_idx, col_offs, col_idx, col_colors)
  let vjp_prepared = (row_offs, row_idx, col_offs, col_idx, row_colors)

  in if use_jvp
     then JVP.eval_prepared_jvp_dense f jvp_prepared x
     else VJP.eval_prepared_vjp_dense f vjp_prepared x

-- Return a dense Jacobian with metadata.
def eval_prepared_auto_dense_with_info [m][n]
  (f: [n]f64 -> [m]f64)
  (prepared: ([m+1]i64, []i64, [n+1]i64, []i64,
              [n]i64, [m]i64, i64, i64, bool))
  (x: [n]f64)
  : ([m][n]f64, bool, i64, i64) =
  let (_row_offs, _row_idx, _col_offs, _col_idx,
       _col_colors, _row_colors,
       num_col_colors, num_row_colors, use_jvp) = prepared

  let jac =
    eval_prepared_auto_dense f prepared x

  in (jac, use_jvp, num_col_colors, num_row_colors)


-- -------------- Full auto pipeline from a sparsity pattern --------------

-- Returns only the chosen mode and coloring info.
def jac_auto_choice [m][n]
  (pat: [m][n]bool)
  : (bool, i64, i64) =
  let (_row_offs, _row_idx, _col_offs, _col_idx,
       _col_colors, _row_colors,
       num_col_colors, num_row_colors, use_jvp) =
    prepare_jac_auto pat
  in (use_jvp, num_col_colors, num_row_colors)

-- Sparse / CSR output.
def jac_auto_csr [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64) =
  let prepared = prepare_jac_auto pat
  in eval_prepared_auto_csr f prepared x

-- Sparse / CSR output with metadata.
def jac_auto_csr_with_info [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64) =
  let prepared = prepare_jac_auto pat
  in eval_prepared_auto_csr_with_info f prepared x

-- Dense output.
def jac_auto_dense [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64)
  : [m][n]f64 =
  let prepared = prepare_jac_auto pat
  in eval_prepared_auto_dense f prepared x

-- Dense output with metadata.
def jac_auto_dense_with_info [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64)
  : ([m][n]f64, bool, i64, i64) =
  let prepared = prepare_jac_auto pat
  in eval_prepared_auto_dense_with_info f prepared x