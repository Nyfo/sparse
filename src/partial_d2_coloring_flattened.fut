import "../lib/github.com/diku-dk/segmented/segmented"

def mark_forbidden_colors [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (col_offs: [n+1]i64)
  (col_idx: []i64)
  (colors: [n]i64)
  (seen: *[n]i64)
  (stamp: i64)
  (v: i64)
  : *[n]i64 =
  let col_neighbor k = col_idx[k]

  let size w =
    row_offs[w+1] - row_offs[w]

  let get w i =
    let t = row_offs[w] + i
    let x = row_idx[t]
    let c = colors[x]
    in if c >= 0i64 then c else -1i64

  let forbidden =
    expand size get (map col_neighbor (col_offs[v]..<col_offs[v+1]))

  in scatter seen forbidden (map (const stamp) forbidden)

def find_index 'a [n] (p: a -> bool) (as: [n]a) : i64 =
  let op (x, i) (y, j) =
    if x && y then
      if i < j then (x, i) else (y, j)
    else if y then
      (y, j)
    else
      (x, i)
  in (reduce_comm op (false, -1i64) (zip (map p as) (iota n))).1

def first_free_color [n] (seen: [n]i64) (stamp: i64) : i64 =
  find_index (!= stamp) seen

def partial_d2_color_cols_order [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (col_offs: [n+1]i64)
  (col_idx: []i64)
  (order: [n]i64)
  : [n]i64 =
  let colors0: [n]i64 = replicate n (-1i64)
  let seen0: [n]i64 = replicate n (-1i64)

  let (colors_final, _seen_final, _stamp_final, _k_final) =
    loop (colors, seen, stamp, k) = (colors0, seen0, 0i64, 0i64)
    while k < n do
      let v = order[k]
      let seen1 =
        mark_forbidden_colors row_offs row_idx col_offs col_idx colors seen stamp v
      let c = first_free_color seen1 stamp
      let colors1 = colors with [v] = c
      in (colors1, seen1, stamp + 1i64, k + 1i64)

  in colors_final

def partial_d2_color_cols [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (col_offs: [n+1]i64)
  (col_idx: []i64)
  : [n]i64 =
  partial_d2_color_cols_order row_offs row_idx col_offs col_idx (iota n)

def partial_d2_color_rows [m][n]
  (row_offs: [m+1]i64)
  (row_idx: []i64)
  (col_offs: [n+1]i64)
  (col_idx: []i64)
  : [m]i64 =
  partial_d2_color_cols_order col_offs col_idx row_offs row_idx (iota m)
