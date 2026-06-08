-- This adapts the GradBench HT objective to the flat-vector input used by
-- the sparse Jacobian benchmarks.

module HT = import "./ht_gradbench_original"

def ht_nnz_per_row : i64 = HT.theta_count + 2i64

def identity4 : [4][4]f64 =
  tabulate_2d 4i64 4i64 (\i j ->
    if i == j then 1.0f64 else 0.0f64)

def unpack_ht_x [num_obs]
  (x: [HT.theta_count + 2*num_obs]f64)
  : ([HT.theta_count]f64, [2*num_obs]f64) =
  let theta : [HT.theta_count]f64 =
    tabulate HT.theta_count (\i -> x[i])

  let us : [2*num_obs]f64 =
    tabulate (2i64*num_obs) (\i -> x[HT.theta_count + i])

  in (theta, us)

def ht_residual_flat [num_bones][num_obs][num_vertices]
  (parents: [num_bones]i32)
  (base_relatives: [num_bones][4][4]f64)
  (inverse_base_absolutes: [num_bones][4][4]f64)
  (weights: [num_bones][num_vertices]f64)
  (base_positions: [4][num_vertices]f64)
  (triangles: [num_vertices][3]i32)
  (is_mirrored: bool)
  (correspondences: [num_obs]i32)
  (points: [3][num_obs]f64)
  (x: [HT.theta_count + 2*num_obs]f64)
  : [3*num_obs]f64 =
  let (theta, us) =
    unpack_ht_x x

  let residuals =
    HT.calculate_objective
      parents
      base_relatives
      inverse_base_absolutes
      weights
      base_positions
      triangles
      is_mirrored
      correspondences
      points
      theta
      us

  in tabulate (3i64*num_obs) (\r ->
       let q = r / 3i64
       let d = r % 3i64
       in residuals[q, d])

def mk_ht_parents (num_bones: i64) : [num_bones]i32 =
  tabulate num_bones (\i ->
    if i == 0i64 then -1i32 else i32.i64 (i - 1i64))

def mk_ht_base_relatives (num_bones: i64)
  : [num_bones][4][4]f64 =
  replicate num_bones identity4

def mk_ht_inverse_base_absolutes (num_bones: i64)
  : [num_bones][4][4]f64 =
  replicate num_bones identity4

def mk_ht_weights (num_bones: i64) (num_vertices: i64)
  : [num_bones][num_vertices]f64 =
  let w = 1.0f64 / f64.i64 num_bones
  in tabulate_2d num_bones num_vertices (\_ _ -> w)

def mk_ht_base_positions (num_vertices: i64)
  : [4][num_vertices]f64 =
  tabulate_2d 4i64 num_vertices (\d v ->
    if d == 0i64 then
      0.01f64 * f64.i64 (v % 97i64)
    else if d == 1i64 then
      0.02f64 * f64.i64 (v % 89i64)
    else if d == 2i64 then
      1.0f64 + 0.01f64 * f64.i64 (v % 83i64)
    else
      1.0f64)

def mk_ht_triangles (num_vertices: i64)
  : [num_vertices][3]i32 =
  tabulate num_vertices (\t ->
    [ i32.i64 (t % num_vertices)
    , i32.i64 ((t + 1i64) % num_vertices)
    , i32.i64 ((t + 2i64) % num_vertices)
    ])

def mk_ht_correspondences (num_obs: i64) (num_vertices: i64)
  : [num_obs]i32 =
  tabulate num_obs (\i ->
    i32.i64 (i % num_vertices))

def mk_ht_points (num_obs: i64)
  : [3][num_obs]f64 =
  tabulate_2d 3i64 num_obs (\d i ->
    if d == 0i64 then
      0.1f64 + 0.001f64 * f64.i64 (i % 101i64)
    else if d == 1i64 then
      0.2f64 + 0.001f64 * f64.i64 (i % 103i64)
    else
      1.0f64 + 0.001f64 * f64.i64 (i % 107i64))

def mk_ht_theta : [HT.theta_count]f64 =
  tabulate HT.theta_count (\i ->
    0.01f64 * f64.i64 (i + 1i64))

def mk_ht_us (num_obs: i64)
  : [2*num_obs]f64 =
  tabulate (2i64*num_obs) (\i ->
    if i % 2i64 == 0i64 then 0.2f64 else 0.3f64)

def mk_ht_x (num_obs: i64)
  : [HT.theta_count + 2*num_obs]f64 =
  let theta = mk_ht_theta
  let us = mk_ht_us num_obs
  in tabulate (HT.theta_count + 2i64*num_obs) (\i ->
       if i < HT.theta_count then theta[i] else us[i - HT.theta_count])

-- CSR row structure for the HT Jacobian.
def mk_ht_row_offs (num_obs: i64)
  : [3*num_obs + 1]i64 =
  tabulate (3i64*num_obs + 1i64) (\i ->
    (HT.theta_count + 2i64) * i)

def mk_ht_row_idx (num_obs: i64)
  : [3*num_obs * (HT.theta_count + 2)]i64 =
  tabulate (3i64*num_obs * (HT.theta_count + 2i64)) (\p ->
    let row = p / (HT.theta_count + 2i64)
    let q = row / 3i64
    let k = p % (HT.theta_count + 2i64)
    in if k < HT.theta_count
       then k
       else HT.theta_count + 2i64*q + (k - HT.theta_count))

-- Column-wise adjacency for the same pattern.
def mk_ht_col_offs (num_obs: i64)
  : [HT.theta_count + 2*num_obs + 1]i64 =
  tabulate (HT.theta_count + 2i64*num_obs + 1i64) (\c ->
    if c < HT.theta_count then
      c * 3i64 * num_obs
    else
      HT.theta_count * 3i64 * num_obs + (c - HT.theta_count) * 3i64)

def mk_ht_col_idx (num_obs: i64)
  : [3*num_obs * (HT.theta_count + 2)]i64 =
  let theta_nnz = HT.theta_count * 3i64 * num_obs
  in tabulate (3i64*num_obs * (HT.theta_count + 2i64)) (\p ->
       if p < theta_nnz then
         p % (3i64*num_obs)
       else
         let t = p - theta_nnz
         let u_col = t / 3i64
         let d = t % 3i64
         let q = u_col / 2i64
         in 3i64*q + d)

def mk_ht_csr_data (num_bones: i64) (num_obs: i64) (num_vertices: i64)
  : (i64, i64, i64,
     [3*num_obs + 1]i64, []i64,
     [HT.theta_count + 2*num_obs + 1]i64, []i64,
     [num_bones]i32,
     [num_bones][4][4]f64,
     [num_bones][4][4]f64,
     [num_bones][num_vertices]f64,
     [4][num_vertices]f64,
     [num_vertices][3]i32,
     bool,
     [num_obs]i32,
     [3][num_obs]f64,
     [HT.theta_count + 2*num_obs]f64) =
  let parents =
    mk_ht_parents num_bones

  let base_relatives =
    mk_ht_base_relatives num_bones

  let inverse_base_absolutes =
    mk_ht_inverse_base_absolutes num_bones

  let weights =
    mk_ht_weights num_bones num_vertices

  let base_positions =
    mk_ht_base_positions num_vertices

  let triangles =
    mk_ht_triangles num_vertices

  let is_mirrored =
    false

  let correspondences =
    mk_ht_correspondences num_obs num_vertices

  let points =
    mk_ht_points num_obs

  let x =
    mk_ht_x num_obs

  let row_offs =
    mk_ht_row_offs num_obs

  let row_idx =
    mk_ht_row_idx num_obs

  let col_offs =
    mk_ht_col_offs num_obs

  let col_idx =
    mk_ht_col_idx num_obs

  in (num_bones, num_obs, num_vertices,
      row_offs, row_idx,
      col_offs, col_idx,
      parents,
      base_relatives,
      inverse_base_absolutes,
      weights,
      base_positions,
      triangles,
      is_mirrored,
      correspondences,
      points,
      x)
