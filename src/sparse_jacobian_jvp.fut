-- Sparse Jacobian via JVP using column coloring from a user-provided
-- sparsity pattern.
--
-- This module exposes three output forms:
--   1) compressed representation
--   2) sparse Jacobian in CSR format
--   3) dense Jacobian

module CSR = import "./pattern_csr"
module Col = import "./partial_d2_coloring"
-- module Col = import "./bgpc_vv_coloring"

def num_colors_of [n] (colors: [n]i64) : i64 =
  if n == 0 then 0i64
  else 1i64 + reduce i64.max 0i64 colors

def seed_for_color [n] (colors: [n]i64) (c: i64) : [n]f64 =
  map (\colc -> if colc == c then 1.0f64 else 0.0f64) colors

-- Compressed evaluation
def compressed_ys_jvp [m][n]
  (f: [n]f64 -> [m]f64)
  (colors: [n]i64)
  (x: [n]f64)
  : [][m]f64 =
  let nc = num_colors_of colors
  in map (\c ->
            let seed = seed_for_color colors c
            in jvp f x seed)
         (iota nc)

-- Reconstruction helpers:
-- Reconstruct only the nonzero Jacobian values in CSR order.
-- If row_idx[p] = j belongs to row i, then
--   vals[p] = J[i,j] = ys[colors[j]][i]
def compressed_to_csr_vals [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (colors: [n]i64)
  (ys: [][m]f64)
  : []f64 =
  let nnz = length row_idx
  let vals0 = replicate nnz 0.0f64

  let (vals_final, _i) =
    loop (vals, i) = (vals0, 0i64)
    while i < m do
      let s = row_offs[i]
      let e = row_offs[i+1]
      let cols = row_idx[s:e]
      let seg  = map (\j -> ys[colors[j]][i]) cols
      let vals' = vals with [s:e] = seg
      in (vals', i + 1i64)

  in vals_final

-- Convert a CSR matrix to dense.
def csr_to_dense [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (vals: []f64)
  : [m][n]f64 =
  map (\i ->
        let s = row_offs[i]
        let e = row_offs[i+1]
        let cols = row_idx[s:e]
        let vs   = vals[s:e]
        let row0 : [n]f64 = replicate n 0.0f64
        in scatter row0 cols vs)
      (iota m)

-- Compressed output:
-- Returns:
--   ((row_offs,row_idx), (col_offs,col_idx), colors, ys)
-- This is the compressed Jacobian representation:
def jac_jvp_compressed [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64) =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let colors =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let ys = compressed_ys_jvp f colors x

  in ((row_offs, row_idx), (col_offs, col_idx), colors, ys)

-- Like jac_jvp_compressed, but assumes colors are already computed.
def jac_jvp_compressed_with_colors [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (colors: [n]i64)
  (x: [n]f64) =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let ys = compressed_ys_jvp f colors x

  in ((row_offs, row_idx), (col_offs, col_idx), colors, ys)

-- Sparse / CSR output:
-- Returns the Jacobian in CSR format:
--   (row_offs, row_idx, vals)
def jac_jvp_csr [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64) =
  let ((row_offs, row_idx), (_col_offs, _col_idx), colors, ys) =
    jac_jvp_compressed f pat x

  let vals = compressed_to_csr_vals row_offs row_idx colors ys
  in (row_offs, row_idx, vals)

-- Like jac_jvp_csr, but assumes colors are already computed.
def jac_jvp_csr_with_colors [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (colors: [n]i64)
  (x: [n]f64) =
  let ((row_offs, row_idx), (_col_offs, _col_idx), _colors, ys) =
    jac_jvp_compressed_with_colors f pat colors x

  let vals = compressed_to_csr_vals row_offs row_idx colors ys
  in (row_offs, row_idx, vals)

-- Dense output:
def jac_jvp_dense [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (x: [n]f64)
  : [m][n]f64 =
  let (row_offs, row_idx, vals) =
    jac_jvp_csr f pat x
  in csr_to_dense row_offs row_idx vals

-- Like jac_jvp_dense, but assumes colors are already computed
def jac_jvp_dense_with_colors [m][n]
  (f: [n]f64 -> [m]f64)
  (pat: [m][n]bool)
  (colors: [n]i64)
  (x: [n]f64)
  : [m][n]f64 =
  let (row_offs, row_idx, vals) =
    jac_jvp_csr_with_colors f pat colors x
  in csr_to_dense row_offs row_idx vals