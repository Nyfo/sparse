-- Fair JVP benchmark on an irregular spiky-row sparsity pattern.
--
-- Compare dense, total sparse, coloring-only, and precolored sparse
-- pipelines that all use the same underlying spiky CSR family.

module Dense = import "../src/dense_jacobian"
module Sparse = import "../src/sparse_jacobian_jvp"
module BGPC = import "../src/bgpc_vv_coloring"
module D2 = import "../src/partial_d2_coloring"
module Cases = import "./bench_cases"

def dense_to_csr_vals [m][n]
  (row_offs:[m+1]i64)
  (row_idx:[]i64)
  (j:[m][n]f64)
  : []f64 =
  let nnz = length row_idx
  let vals0 = replicate nnz 0.0f64

  let (vals_final, _i) =
    loop (vals, i) = (vals0, 0i64)
    while i < m do
      let s = row_offs[i]
      let e = row_offs[i+1]
      let cols = row_idx[s:e]
      let jrow = j[i]
      let seg = map (\col -> jrow[col]) cols
      let vals' = vals with [s:e] = seg
      in (vals', i + 1i64)

  in vals_final

def colors_summary [n] (colors:[n]i64) : (i64, i64) =
  let checksum = reduce (+) 0i64 colors
  let nc = if n == 0 then 0i64 else 1i64 + reduce i64.max 0i64 colors
  in (checksum, nc)

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

entry mk_spiky_csr_test_with_bgpc_colors
  (m:i64) (n:i64) (small_deg:i64) (big_deg:i64) (num_big_rows:i64)
  : (i64, i64, i64, i64, i64, [m+1]i64, []i64, [n+1]i64, []i64, [n]f64, [n]i64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_spiky_rows m n small_deg big_deg num_big_rows
  let x : [n]f64 = Cases.rand_vec 42i64
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  in (m, n, small_deg, big_deg, num_big_rows,
      row_offs, row_idx, col_offs, col_idx, x, colors)

entry mk_spiky_csr_test_with_d2_colors
  (m:i64) (n:i64) (small_deg:i64) (big_deg:i64) (num_big_rows:i64)
  : (i64, i64, i64, i64, i64, [m+1]i64, []i64, [n+1]i64, []i64, [n]f64, [n]i64) =
  let (row_offs, row_idx, col_offs, col_idx) =
    Cases.mk_csr_spiky_rows m n small_deg big_deg num_big_rows
  let x : [n]f64 = Cases.rand_vec 42i64
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in (m, n, small_deg, big_deg, num_big_rows,
      row_offs, row_idx, col_offs, col_idx, x, colors)

-- Dense sizes stay moderate to avoid dense Jacobian memory blowups.
-- ==
-- entry: bench_dense_jvp_to_csr_spiky
-- script input { mk_spiky_csr_test 2048 32768 5 256 64 }
-- script input { mk_spiky_csr_test 3072 49152 5 512 128 }
entry bench_dense_jvp_to_csr_spiky
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (_col_offs:[n+1]i64) (_col_idx:[]i64)
  (x:[n]f64)
  : []f64 =
  let j = Dense.jac_dense_jvp (f_spiky row_offs row_idx) x
  in dense_to_csr_vals row_offs row_idx j

-- ==
-- entry: bench_sparse_jvp_to_csr_spiky_bgpc
-- script input { mk_spiky_csr_test 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test 32768 524288 5 1024 256 }
-- script input { mk_spiky_csr_test 40960 655360 5 1280 320 }
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
-- script input { mk_spiky_csr_test 40960 655360 5 1280 320 }
entry bench_sparse_jvp_to_csr_spiky_d2
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (x:[n]f64)
  : []f64 =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  let ys = Sparse.compressed_ys_jvp (f_spiky row_offs row_idx) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_color_spiky_bgpc
-- script input { mk_spiky_csr_test 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test 32768 524288 5 1024 256 }
-- script input { mk_spiky_csr_test 40960 655360 5 1280 320 }
entry bench_color_spiky_bgpc
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (_x:[n]f64)
  : (i64, i64) =
  let colors = BGPC.vv_color_cols row_offs row_idx col_offs col_idx
  in colors_summary colors

-- ==
-- entry: bench_color_spiky_d2
-- script input { mk_spiky_csr_test 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test 32768 524288 5 1024 256 }
-- script input { mk_spiky_csr_test 40960 655360 5 1280 320 }
entry bench_color_spiky_d2
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (_x:[n]f64)
  : (i64, i64) =
  let colors = D2.partial_d2_color_cols row_offs row_idx col_offs col_idx
  in colors_summary colors

-- ==
-- entry: bench_sparse_jvp_to_csr_spiky_precolored_bgpc
-- script input { mk_spiky_csr_test_with_bgpc_colors 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test_with_bgpc_colors 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test_with_bgpc_colors 32768 524288 5 1024 256 }
-- script input { mk_spiky_csr_test_with_bgpc_colors 40960 655360 5 1280 320 }
entry bench_sparse_jvp_to_csr_spiky_precolored_bgpc
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (_col_offs:[n+1]i64) (_col_idx:[]i64)
  (x:[n]f64) (colors:[n]i64)
  : []f64 =
  let ys = Sparse.compressed_ys_jvp (f_spiky row_offs row_idx) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_to_csr_spiky_precolored_d2
-- script input { mk_spiky_csr_test_with_d2_colors 8192 131072 5 256 64 }
-- script input { mk_spiky_csr_test_with_d2_colors 16384 262144 5 512 128 }
-- script input { mk_spiky_csr_test_with_d2_colors 32768 524288 5 1024 256 }
-- script input { mk_spiky_csr_test_with_d2_colors 40960 655360 5 1280 320 }
entry bench_sparse_jvp_to_csr_spiky_precolored_d2
  (m:i64) (n:i64) (_small_deg:i64) (_big_deg:i64) (_num_big_rows:i64)
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (_col_offs:[n+1]i64) (_col_idx:[]i64)
  (x:[n]f64) (colors:[n]i64)
  : []f64 =
  let ys = Sparse.compressed_ys_jvp (f_spiky row_offs row_idx) colors x
  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
