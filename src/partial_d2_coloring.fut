def neigh (offs:[]i64) (idx:[]i64) (v:i64) : []i64 =
  let s = offs[v]
  let e = offs[v+1]
  in idx[s:e]

-- seen is UNIQUE (*)
def mark_forbidden_colors [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (colors:[n]i64)
  (seen:*[n]i64) -- Den her skal åbenbart have *
  (stamp:i64)
  (v:i64)
  : *[n]i64 =
  let k = col_offs[v]
  let k_end = col_offs[v+1]
  let (seen_final, _k) =
    loop (seen, k)
    while k < k_end do
      let w = col_idx[k] -- a row neighbor of v

      let t = row_offs[w]
      let t_end = row_offs[w+1]
      let (seen2_final, _t) =
        loop (seen, t)
        while t < t_end do
          let x = row_idx[t] -- a column neighbor of row w
          let c = colors[x]
          let seen' =
            if c >= 0i64 then seen with [c] = stamp else seen
          in (seen', t + 1i64)

      in (seen2_final, k + 1i64)
  in seen_final

def first_free_color [n] (seen:[n]i64) (stamp:i64) : i64 =
  let c0 = 0i64
  let c_final =
    loop c0
    while c0 < n && seen[c0] == stamp do
      let c1 = c0 + 1i64
      in c1
  in c_final

def partial_d2_color_cols_order [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  (order:[n]i64)
  : [n]i64 =
  let colors0 : [n]i64 = replicate n (-1i64)
  let seen0   : [n]i64 = replicate n (-1i64)
  let stamp0  = 0i64
  let k0      = 0i64

  let (colors_final, _seen_final, _stamp_final, _k) =
    loop (colors0, seen0, stamp0, k0)
    while k0 < n do
      let v = order[k0]
      -- This below leads to so many nested loops, RETHINK THIS!!!
      let seen1 = mark_forbidden_colors row_offs row_idx col_offs col_idx colors0 seen0 stamp0 v
      let c = first_free_color seen1 stamp0
      let colors1 = colors0 with [v] = c
      let stamp1 = stamp0 + 1i64
      let k1 = k0 + 1i64
      in (colors1, seen1, stamp1, k1)

  in colors_final

def partial_d2_color_cols [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [n]i64 =
  partial_d2_color_cols_order row_offs row_idx col_offs col_idx (iota n)

-- Color ROWS instead of columns:
def partial_d2_color_rows [m][n]
  (row_offs:[m+1]i64) (row_idx:[]i64)
  (col_offs:[n+1]i64) (col_idx:[]i64)
  : [m]i64 =
  partial_d2_color_cols_order col_offs col_idx row_offs row_idx (iota m)


