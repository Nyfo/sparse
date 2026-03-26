-- Fair JVP benchmark on an irregular spiky-row sparsity pattern.
--
-- Compare sparse pipelines that all produce CSR Jacobian values.
-- This family is designed to give the GPU-oriented BGPC coloring a
-- better chance than the very regular banded/stencil cases.

module Sparse = import "../src/sparse_jacobian_jvp"
module BGPC = import "../src/bgpc_vv_coloring"
module D2 = import "../src/partial_d2_coloring"
module Cases = import "./bench_cases"

def f_spiky [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (x:[n]f64)
  : [m]f64 =
  map (\i ->
        let js = row_idx[row_offs[i]:row_offs[i+1]]
        let s1 =
          reduce (+) 0.0f64
            (map (\j ->
                    let v = x[j]
                    let v2 = v * v
                    in f64.sin v + 0.1f64 * v2 + 0.01f64 * v2 * v)
                 js)
        let s2 =
          reduce (+) 0.0f64
            (map (\j ->
                    let v = x[j]
                    in f64.cos (0.5f64 * v) + 1.0f64 / (1.0f64 + v * v))
                 js)
        let s3 =
          reduce (+) 0.0f64
            (map (\j ->
                    let v = x[j]
                    in 1.0f64 / (1.0f64 + f64.abs v))
                 js)
        in s1 + 0.01f64 * s1 * s2 + 0.001f64 * s2 * s3)
      (iota m)

entry mk_spiky_csr_test
  (m:i64) (n:i64) (small_deg:i64) (big_deg:i64) (num_big_rows:i64)
  : (i64, i64, i64, i64, i64, [m+1]i64, []i64, [n+1]i64, []i64, [n]f64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_spiky_rows m n small_deg big_deg num_big_rows
  let x : [n]f64 = Cases.rand_vec 42i64
  in (m, n, small_deg, big_deg, num_big_rows,
      row_offs, row_idx, col_offs, col_idx, x)

-- ==
-- entry: bench_sparse_jvp_to_csr_spiky_bgpc
-- script input { mk_spiky_csr_test 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test 32768 524288 5 1024 256 }
entry bench_sparse_jvp_to_csr_spiky_bgpc
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (x:[n]f64)
  : []f64 =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (f_spiky row_offs row_idx) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_to_csr_spiky_d2
-- script input { mk_spiky_csr_test 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test 32768 524288 5 1024 256 }
entry bench_sparse_jvp_to_csr_spiky_d2
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (x:[n]f64)
  : []f64 =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (f_spiky row_offs row_idx) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
