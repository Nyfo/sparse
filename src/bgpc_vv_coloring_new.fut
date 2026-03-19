import "../lib/github.com/diku-dk/segmented/segmented"

def bool_to_i64 (b: bool) : i64 =
  if b then 1i64 else 0i64

def compact_true_values [n] (xs: [n]i64) (flags: [n]bool) : []i64 =
  let counts : [n]i64 = map bool_to_i64 flags
  let pos    : [n]i64 = scan (+) 0i64 counts
  let out_sz : i64 =
    if n == 0
    then 0i64
    else pos[n - 1]

  let tagged : [n](i64, (i64, bool)) = zip xs (zip pos flags)
  let kept   : [](i64, (i64, bool)) =
    filter (\(_, (_, keep)) -> keep) tagged

  let dst  : []i64 = map (\(_, (p, _)) -> p - 1i64) kept
  let vals : []i64 = map (\(x, _) -> x) kept

  let out0 : []i64 = replicate out_sz 0i64
  in scatter out0 dst vals

def scatter_true_at_indices [n][k]
  (dest: *[n]bool)
  (idxs: [k]i64)
  : *[n]bool =
  let vals : [k]bool = replicate k true
  in scatter dest idxs vals

def mark_forbidden_colors_vertex [nets][verts]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (colors: [verts]i64)
  (seen: *[]bool)
  (w: i64)
  : *[]bool =
  let k0    = vert_offs[w]
  let k_end = vert_offs[w + 1]

  let (seen_final, _k) =
    loop (seen_acc, k) = (seen, k0)
    while k < k_end do
      let v = vert_idx[k]

      let t0    = net_offs[v]
      let t_end = net_offs[v + 1]

      let (seen_acc2, _t) =
        loop (seen_acc_inner, t) = (seen_acc, t0)
        while t < t_end do
          let u  = net_idx[t]
          let cu = colors[u]

          let seen_acc_inner' =
            if u != w && cu >= 0i64 && cu < length seen_acc_inner
            then seen_acc_inner with [cu] = true
            else seen_acc_inner

          in (seen_acc_inner', t + 1i64)

      in (seen_acc2, k + 1i64)

  in seen_final

def first_free_color (seen: []bool) : i64 =
  let c0 = 0i64
  let c_final =
    loop c = c0
    while c < length seen && seen[c] do
      c + 1i64
  in c_final

def first_fit_color [nets][verts]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (colors: [verts]i64)
  (color_bound: i64)
  (w: i64)
  : i64 =
  let seen0 : []bool = replicate color_bound false
  let seen1 =
    mark_forbidden_colors_vertex
      net_offs net_idx vert_offs vert_idx colors seen0 w
  in first_free_color seen1

def color_workqueue_vertex [nets][verts][k]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (W: [k]i64)
  (color_bound: i64)
  (colors: *[verts]i64)
  : (*[verts]i64, i64) =
  let tentative : [k]i64 =
    map (\w ->
          first_fit_color
            net_offs net_idx vert_offs vert_idx colors color_bound w)
        W

  let max_tent = reduce i64.max (-1i64) tentative
  let next_color_bound =
    if max_tent < 0i64
    then color_bound
    else i64.max color_bound (max_tent + 2i64)

  let colors' = scatter colors W tentative
  in (colors', next_color_bound)

def loses_conflict_vertex [nets][verts]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (colors: [verts]i64)
  (w: i64)
  : bool =
  let k0    = vert_offs[w]
  let k_end = vert_offs[w + 1]

  let (lost_final, _k) =
    loop (lost, k) = (false, k0)
    while k < k_end && not lost do
      let v = vert_idx[k]

      let t0    = net_offs[v]
      let t_end = net_offs[v + 1]

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

def expand_owner_nets [verts][k]
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (W: [k]i64)
  : [](i64, i64) =
  let owners : [k]i64 = iota k

  let size (owner: i64) : i64 =
    let w = W[owner]
    in vert_offs[w + 1] - vert_offs[w]

  let get (owner: i64) (j: i64) : (i64, i64) =
    let w   = W[owner]
    let pos = vert_offs[w] + j
    let net = vert_idx[pos]
    in (owner, net)

  in expand size get owners

def expand_owner_neigh2 [nets]
  (net_offs: [nets + 1]i64) (net_idx: []i64)
  (owner_nets: [](i64, i64))
  : [](i64, i64) =
  let size ((_, net): (i64, i64)) : i64 =
    net_offs[net + 1] - net_offs[net]

  let get ((owner, net): (i64, i64)) (j: i64) : (i64, i64) =
    let pos = net_offs[net] + j
    let u   = net_idx[pos]
    in (owner, u)

  in expand size get owner_nets

def lose_flags_flat [nets][verts][k]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (W: [k]i64)
  (colors: [verts]i64)
  : [k]bool =
  let owner_nets : [](i64, i64) =
    expand_owner_nets vert_offs vert_idx W

  let owner_neigh2 : [](i64, i64) =
    expand_owner_neigh2 net_offs net_idx owner_nets

  let losing_owners : []i64 =
    map (\(owner, _) -> owner)
      (filter (\(owner, u) ->
                 let w  = W[owner]
                 let cw = colors[w]
                 in cw >= 0i64 && u != w && colors[u] == cw && w > u)
              owner_neigh2)

  let lose0 : [k]bool = replicate k false
  in scatter_true_at_indices lose0 losing_owners

def remove_conflicts_vertex_orig [nets][verts][k]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (W: [k]i64)
  (colors: *[verts]i64)
  : ([]i64, *[verts]i64) =
  let lose_flags : [k]bool =
    map (\w ->
          loses_conflict_vertex
            net_offs net_idx vert_offs vert_idx colors w)
        W

  let Wnext0 : []i64 =
    compact_true_values W lose_flags

  let Wnext : []i64 =
    copy Wnext0

  let reset_vals : []i64 =
    map (\_ -> -1i64) Wnext

  let colors' =
    scatter colors Wnext reset_vals

  in (Wnext, colors')

def remove_conflicts_vertex_flat [nets][verts][k]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (W: [k]i64)
  (colors: *[verts]i64)
  : ([]i64, *[verts]i64) =
  let lose_flags : [k]bool =
    lose_flags_flat
      net_offs net_idx vert_offs vert_idx W colors

  let Wnext0 : []i64 =
    compact_true_values W lose_flags

  let Wnext : []i64 =
    copy Wnext0

  let reset_vals : []i64 =
    map (\_ -> -1i64) Wnext

  let colors' =
    scatter colors Wnext reset_vals

  in (Wnext, colors')

def remove_conflicts_vertex [nets][verts][k]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (W: [k]i64)
  (colors: *[verts]i64)
  : ([]i64, *[verts]i64) =
  let flat_limit : i64 = 4000i64
  in if length W <= flat_limit
     then remove_conflicts_vertex_flat
            net_offs net_idx vert_offs vert_idx W colors
     else remove_conflicts_vertex_orig
            net_offs net_idx vert_offs vert_idx W colors

def vv_color_side_order [nets][verts]
  (net_offs:  [nets + 1]i64) (net_idx:  []i64)
  (vert_offs: [verts + 1]i64) (vert_idx: []i64)
  (order: [verts]i64)
  : [verts]i64 =
  let colors0      : [verts]i64 = replicate verts (-1i64)
  let color_bound0 : i64        = 1i64
  let W0 : [verts]i64 = order

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


def vv_color_cols [m][n]
  (row_offs: [m + 1]i64) (row_idx: []i64)
  (col_offs: [n + 1]i64) (col_idx: []i64)
  : [n]i64 =
  vv_color_side_order row_offs row_idx col_offs col_idx (iota n)

def vv_color_rows [m][n]
  (row_offs: [m + 1]i64) (row_idx: []i64)
  (col_offs: [n + 1]i64) (col_idx: []i64)
  : [m]i64 =
  vv_color_side_order col_offs col_idx row_offs row_idx (iota m)