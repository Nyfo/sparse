-- BGPC V-V coloring for Jacobian sparsity patterns
--
-- Paper mapping:
--  Algorithm 1 -> vv_color_side_order
--  Algorithm 4 -> color_workqueue_vertex
--  Algorithm 5 -> remove_conflicts_vertex

import "../lib/github.com/diku-dk/segmented/segmented"

-- Helper function:
def bool_to_i64 (b:bool) : i64 =
  if b then 1i64 else 0i64

-- Helper function:
-- Compact xs according to flags using scan + scatter.
-- Keeps xs[i] exactly when flags[i] is true.
def compact_true_values (xs:[]i64) (flags:[]bool) : []i64 =
  let counts : []i64 = map bool_to_i64 flags
  let pos    : []i64 = scan (+) 0i64 counts
  let out_sz : i64 =
    if length pos == 0
    then 0i64
    else pos[length pos - 1]

  let tagged : [](i64, (i64, bool)) = zip xs (zip pos flags)
  let kept   : [](i64, (i64, bool)) =
    filter (\(_, (_, keep)) -> keep) tagged

  let dst : []i64 = map (\(_, (p, _)) -> p - 1i64) kept
  let vals: []i64 = map (\(x, _) -> x) kept

  let out0 : []i64 = replicate out_sz 0i64
  in scatter out0 dst vals

-- Helper function used by Algorithm 4:
-- Mark forbidden colors for one vertex w by traversing its
-- BGPC neighborhood exactly once
def mark_forbidden_colors_vertex [nets][verts]
  (net_offs:[nets+1]i64) (net_idx:[]i64)
  (vert_offs:[verts+1]i64) (vert_idx:[]i64)
  (colors:[verts]i64)
  (seen:*[]bool)
  (w:i64)
  : *[]bool =
  let seen_final =
    loop seen_acc = seen for v in vert_idx[vert_offs[w] : vert_offs[w+1]] do
      let seen_acc' =
        loop seen_acc_inner = seen_acc
        for u in net_idx[net_offs[v] : net_offs[v+1]] do
          let cu = colors[u]
          in if u != w && cu >= 0i64 && cu < length seen_acc_inner
             then seen_acc_inner with [cu] = true
             else seen_acc_inner
      in seen_acc'
  in seen_final

-- Helper function used by Algorithm 4:
-- Find the first color not marked forbidden.
def first_free_color (seen:[]bool) : i64 =
  let c0 = 0i64
  let c_final =
    loop c = c0
    while c < length seen && seen[c] do
      c + 1i64
  in c_final

-- Helper function used by Algorithm 4:
-- First-fit color for one vertex w, using a forbidden-color
-- array built in one neighborhood traversal.
def first_fit_color [nets][verts]
  (net_offs:[nets+1]i64) (net_idx:[]i64)
  (vert_offs:[verts+1]i64) (vert_idx:[]i64)
  (colors:[verts]i64)
  (color_bound:i64)
  (w:i64)
  : i64 =
  let seen0 : []bool = replicate color_bound false
  let seen1 =
    mark_forbidden_colors_vertex
      net_offs net_idx vert_offs vert_idx colors seen0 w
  in first_free_color seen1

-- Algorithm 4: BGPC-COLORWORKQUEUE-VERTEX
--
-- Input:
--  W = current work queue
--  color_bound = current bound on first free color
--  colors = incomplete coloring with no conflicts among kept vertices
--
-- Output:
--  optimistic coloring after coloring all vertices in W
--  next_color_bound = updated bound based on colors
def color_workqueue_vertex [nets][verts]
  (net_offs:[nets+1]i64) (net_idx:[]i64)
  (vert_offs:[verts+1]i64) (vert_idx:[]i64)
  (W:[]i64)
  (color_bound:i64)
  (colors:*[verts]i64)
  : (*[verts]i64, i64) =
  let tentative : []i64 =
    map (\w ->
          first_fit_color
            net_offs net_idx vert_offs vert_idx colors color_bound w)
        W

  let max_tent = reduce i64.max (-1i64) tentative
  let next_color_bound =
    if max_tent < 0i64
    then color_bound
    else i64.max color_bound (max_tent + 2i64)

  -- Safe because vertices in W are unique.
  let colors' = scatter colors W tentative

  in (colors', next_color_bound)

-- Helper function used by Algorithm 5:
-- Conflict rule from Algorithm 3:
--   if colors[u] == colors[w] and w > u then w loses
def loses_conflict_vertex [nets][verts]
  (net_offs:[nets+1]i64) (net_idx:[]i64)
  (vert_offs:[verts+1]i64) (vert_idx:[]i64)
  (colors:[verts]i64)
  (w:i64)
  : bool =
  let k0    = vert_offs[w]
  let k_end = vert_offs[w+1]

  let (lost_final, _k) =
    loop (lost, k) = (false, k0)
    while k < k_end && not lost do
      let v = vert_idx[k]

      let t0    = net_offs[v]
      let t_end = net_offs[v+1]

      let (lost_in_net_final, _t) =
        loop (lost_in_net, t) = (false, t0)
        while t < t_end && not lost_in_net do
          let u = net_idx[t]
          let lost_in_net' =
            lost_in_net || (u != w && colors[u] == colors[w] && w > u)
          in (lost_in_net', t + 1i64)

      let lost' = lost || lost_in_net_final
      in (lost', k + 1i64)

  in lost_final

-- Algorithm 5: BGPC-REMOVECONFLICTS-VERTEX
--
-- Input:
--  W = current work queue
--  colors = optimistic coloring
--
-- Output:
--  Wnext = vertices to recolor next iteration
--  colors = same coloring, but losers reset to -1
def remove_conflicts_vertex [nets][verts]
  (net_offs:[nets+1]i64) (net_idx:[]i64)
  (vert_offs:[verts+1]i64) (vert_idx:[]i64)
  (W:[]i64)
  (colors:*[verts]i64)
  : ([]i64, *[verts]i64) =
  let lose_flags : []bool =
    map (\w ->
          loses_conflict_vertex
            net_offs net_idx vert_offs vert_idx colors w)
        W

  let Wnext : []i64 = compact_true_values W lose_flags

  -- Reset losing vertices to -1.
  let reset_vals : []i64 = map (\_ -> -1i64) Wnext
  let colors' = scatter colors Wnext reset_vals

  in (Wnext, colors')

-- Algorithm 1: GREEDYGRAPHCOLORING
--
-- Repeatedly as shown in the algorithm 1 pseudocode:
--  1) color current work queue optimistically
--  2) remove conflicts
-- until W is empty
def vv_color_side_order [nets][verts]
  (net_offs:[nets+1]i64) (net_idx:[]i64)
  (vert_offs:[verts+1]i64) (vert_idx:[]i64)
  (order:[verts]i64)
  : [verts]i64 =
  let colors0      : [verts]i64 = replicate verts (-1i64)
  let color_bound0 : i64        = 1i64

  -- Turn fixed-size order into existential work queue.
  let W0 : []i64 = filter (\_ -> true) order

  let (colors_final, _W_final, _color_bound_final) =
    loop (colors, W, color_bound) = (colors0, W0, color_bound0)
    while length W > 0 do
      let (colors1, color_bound1) =
        color_workqueue_vertex
          net_offs net_idx vert_offs vert_idx W color_bound colors

      let (W1, colors2) =
        remove_conflicts_vertex
          net_offs net_idx vert_offs vert_idx W colors1

      in (colors2, W1, color_bound1)

  in colors_final

-- Color columns: rows are nets, columns are vertices.
def vv_color_cols [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [n]i64 =
  vv_color_side_order row_offs row_idx col_offs col_idx (iota n)

-- Color rows: swap the two sides.
def vv_color_rows [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [m]i64 =
  vv_color_side_order col_offs col_idx row_offs row_idx (iota m)
