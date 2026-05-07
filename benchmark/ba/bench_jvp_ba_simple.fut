module Dense = import "../../src/dense_jacobian"
module Sparse = import "../../src/sparse_jacobian_jvp"
module CSR = import "../../src/pattern_csr"
module D2 = import "../../src/partial_d2_coloring"
module BGPC = import "../../src/bgpc_vv_coloring"
module BA = import "./ba_gradbench_original"

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

def unpack_ba_x
  (num_cams:i64) (num_points:i64) (num_obs:i64)
  (x: [11*num_cams + 3*num_points + num_obs]f64)
  : ([num_cams][11]f64, [num_points][3]f64, [num_obs]f64) =
  let cams : [num_cams][11]f64 =
    tabulate num_cams (\i ->
      tabulate 11i64 (\j ->
        x[11i64*i + j]))

  let points : [num_points][3]f64 =
    tabulate num_points (\i ->
      tabulate 3i64 (\j ->
        x[11i64*num_cams + 3i64*i + j]))

  let weights : [num_obs]f64 =
    tabulate num_obs (\i ->
      x[11i64*num_cams + 3i64*num_points + i])

  in (cams, points, weights)

def ba_residual_flat [num_obs]
  (num_cams:i64) (num_points:i64)
  (obs: [num_obs][2]i32)
  (feat: [num_obs][2]f64)
  (x: [11*num_cams + 3*num_points + num_obs]f64)
  : [3*num_obs]f64 =
  let (cams_arr, points_arr, weights) =
    unpack_ba_x num_cams num_points num_obs x

  let cams =
    map BA.unpack_cam cams_arr

  let points =
    map BA.point_3d points_arr

  let feats =
    map BA.point_2d feat

  let (reproj_err, weight_err) =
    BA.ba_objective cams points weights obs feats

  in tabulate (3*num_obs) (\r ->
       if r < 2*num_obs then
         let ob = r / 2i64
         let e = reproj_err[ob]
         in if r % 2i64 == 0i64 then e.x else e.y
       else
         weight_err[r - 2*num_obs])

def mk_ba_obs (num_cams:i64) (num_points:i64) (num_obs:i64)
  : [num_obs][2]i32 =
  tabulate num_obs (\i ->
    [i32.i64 (i % num_cams),
     i32.i64 ((7i64*i + 3i64) % num_points)])

def mk_ba_features (num_obs:i64) : [num_obs][2]f64 =
  tabulate num_obs (\_ -> [0.0f64, 0.0f64])

def mk_ba_x (num_cams:i64) (num_points:i64) (num_obs:i64)
  : [11*num_cams + 3*num_points + num_obs]f64 =
  tabulate (11*num_cams + 3*num_points + num_obs) (\idx ->
    if idx < 11*num_cams then
      let k = idx % 11i64
      in if k == 6i64 then 1.0f64 else 0.0f64
    else if idx < 11*num_cams + 3*num_points then
      let rel = idx - 11*num_cams
      let point_id = rel / 3i64
      let k = rel % 3i64
      in if k == 0i64 then
           0.01f64 * f64.i64 (point_id % 17i64)
         else if k == 1i64 then
           0.01f64 * f64.i64 ((3i64 * point_id) % 17i64)
         else
           4.0f64 + 0.01f64 * f64.i64 (point_id % 13i64)
    else
      1.0f64)

def pat_ba [num_obs]
  (num_cams:i64) (num_points:i64)
  (obs: [num_obs][2]i32)
  : [3*num_obs][11*num_cams + 3*num_points + num_obs]bool =
  tabulate (3*num_obs) (\r ->
    tabulate (11*num_cams + 3*num_points + num_obs) (\j ->
      if r < 2*num_obs then
        let ob = r / 2i64
        let cam_id = i64.i32 obs[ob, 0]
        let point_id = i64.i32 obs[ob, 1]

        let cam_start = 11i64 * cam_id
        let point_start = 11i64*num_cams + 3i64*point_id
        let weight_col = 11i64*num_cams + 3i64*num_points + ob

        in (j >= cam_start && j < cam_start + 11i64) ||
           (j >= point_start && j < point_start + 3i64) ||
           j == weight_col
      else
        let ob = r - 2*num_obs
        let weight_col = 11i64*num_cams + 3i64*num_points + ob
        in j == weight_col))

entry mk_ba_csr_test (num_cams:i64) (num_points:i64) (num_obs:i64)
  : (i64, i64, i64,
     [3*num_obs+1]i64, []i64,
     [11*num_cams + 3*num_points + num_obs + 1]i64, []i64,
     [num_obs][2]i32, [num_obs][2]f64,
     [11*num_cams + 3*num_points + num_obs]f64) =
  let obs = mk_ba_obs num_cams num_points num_obs
  let feat = mk_ba_features num_obs
  let x = mk_ba_x num_cams num_points num_obs

  let pat : [3*num_obs][11*num_cams + 3*num_points + num_obs]bool =
    pat_ba num_cams num_points obs

  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  in (num_cams, num_points, num_obs,
      row_offs, row_idx, col_offs, col_idx,
      obs, feat, x)

-- ==
-- entry: bench_dense_jvp_to_csr_ba
-- script input { mk_ba_csr_test 2 8 16 }
-- script input { mk_ba_csr_test 4 16 64 }
-- script input { mk_ba_csr_test 8 32 128 }
entry bench_dense_jvp_to_csr_ba (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (_col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (_col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : []f64 =
  let j =
    Dense.jac_dense_jvp (\x0 -> ba_residual_flat num_cams num_points obs feat x0) x
  in dense_to_csr_vals row_offs row_idx j

-- ==
-- entry: bench_sparse_jvp_to_csr_ba_d2
-- script input { mk_ba_csr_test 2 8 16 }
-- script input { mk_ba_csr_test 4 16 64 }
-- script input { mk_ba_csr_test 8 32 128 }
entry bench_sparse_jvp_to_csr_ba_d2 (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : []f64 =
  let colors =
    D2.partial_d2_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys

-- ==
-- entry: bench_sparse_jvp_to_csr_ba_bgpc
-- script input { mk_ba_csr_test 2 8 16 }
-- script input { mk_ba_csr_test 4 16 64 }
-- script input { mk_ba_csr_test 8 32 128 }
entry bench_sparse_jvp_to_csr_ba_bgpc (num_cams:i64) (num_points:i64) (num_obs:i64)
  (row_offs:[3*num_obs+1]i64) (row_idx:[]i64)
  (col_offs:[11*num_cams + 3*num_points + num_obs + 1]i64) (col_idx:[]i64)
  (obs:[num_obs][2]i32) (feat:[num_obs][2]f64)
  (x:[11*num_cams + 3*num_points + num_obs]f64)
  : []f64 =
  let colors =
    BGPC.vv_color_cols row_offs row_idx col_offs col_idx

  let ys =
    Sparse.compressed_ys_jvp
      (\x0 -> ba_residual_flat num_cams num_points obs feat x0)
      colors
      x

  in Sparse.compressed_to_csr_vals row_offs row_idx colors ys
