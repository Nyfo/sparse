-- benchmark/bench_cases.fut
-- Shared benchmark cases (helpers + test functions + patterns).

module CSR = import "../src/pattern_csr"
module Col = import "../src/partial_d2_coloring"

-- Reused the randomizer from my DPP project about vtrees
def rand_vec [n] (seed:i64) : [n]f64 =
  -- hash function (based on MurmurHash3), reused from earlier DPP project.
  let hash (x: u64) : u64 =
    let x = x ^ (x >> 30)
    let x = x * 0xbf58476d1ce4e5b9u64
    let x = x ^ (x >> 27)
    let x = x * 0x94d049bb133111ebu64
    in x ^ (x >> 31)

  let rand_f64 (k:i64) : f64 =
    let h   : u64 = hash (u64.i64 (seed + k))
    let r01 : f64 = f64.u64 (h % 1000000u64) / 1000000.0f64
    in r01

  in tabulate n (\i -> rand_f64 i)

def wrap (i:i64) (n:i64) : i64 =
  let r = i % n
  in if r < 0i64 then r + n else r

-- Case: banded5 matrix (each output depends on 5 inputs)
-- Assumes n is a multiple of m (stride = n/m).

def f_banded5 [m][n] (rows:[m]i64) (x:[n]f64) : [m]f64 =
  let stride : i64 = n / m
  in map (\i ->
            let b  = (i * stride) % n
            let j0 = b
            let j1 = wrap (b + 1i64) n
            let j2 = wrap (b - 1i64) n
            let j3 = wrap (b + 2i64) n
            let j4 = wrap (b - 2i64) n
            in (x[j0] + x[j1] + x[j2] + x[j3] + x[j4]) / 5.0f64)
         rows

def pat_banded5 [m][n] (rows:[m]i64) : [m][n]bool =
  let stride : i64 = n / m
  in map (\i ->
            let b  = (i * stride) % n
            let j0 = b
            let j1 = wrap (b + 1i64) n
            let j2 = wrap (b - 1i64) n
            let j3 = wrap (b + 2i64) n
            let j4 = wrap (b - 2i64) n
            in tabulate n (\j ->
                 j == j0 || j == j1 || j == j2 || j == j3 || j == j4))
         rows

-- functions to make matrices (used by bench files' entry points)

def mk_pat_banded5 (m:i64) (n:i64) : [m][n]bool =
  let rows : [m]i64 = iota m
  in pat_banded5 rows

def mk_csr_banded5 (m:i64) (n:i64) : ([m+1]i64, []i64, [n+1]i64, []i64) =
  let rows : [m]i64 = iota m
  let pat  : [m][n]bool = pat_banded5 rows
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat
  in (row_offs, row_idx, col_offs, col_idx)

def mk_banded5_inputs (m:i64) (n:i64) : ([m][n]bool, [m]i64, [n]f64) =
  let rows : [m]i64 = iota m
  let x    : [n]f64 = rand_vec 42i64
  let pat  : [m][n]bool = pat_banded5 rows
  in (pat, rows, x)

def mk_banded5_inputs_with_colors (m:i64) (n:i64)
  : ([m][n]bool, [n]i64, [m]i64, [n]f64) =
  let rows : [m]i64 = iota m
  let x    : [n]f64 = rand_vec 42i64
  let pat  : [m][n]bool = pat_banded5 rows

  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat
  let colors : [n]i64 =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx

  in (pat, colors, rows, x)

def mk_banded5_inputs_with_row_colors (m:i64) (n:i64)
  : ([m][n]bool, [m]i64, [m]i64, [n]f64) =
  let rows : [m]i64 = iota m
  let x    : [n]f64 = rand_vec 42i64
  let pat  : [m][n]bool = pat_banded5 rows

  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let row_colors : [m]i64 =
    Col.partial_d2_color_rows row_offs row_idx col_offs col_idx

  in (pat, row_colors, rows, x)

-- section: Case = 2D 5-point stencil with wrap-around
-- Input/output length is h*w.

def stencil2d [h][w] (xs:[h*w]f64) : [h*w]f64 =
  let xs2d : [h][w]f64 = unflatten xs
  let f c e wv s nv = (c + e + wv + s + nv) / 5.0f64
  let ys2d : [h][w]f64 =
    map5 (map5 f)
         xs2d
         (map (rotate 1) xs2d)
         (map (rotate (-1)) xs2d)
         (rotate 1 xs2d)
         (rotate (-1) xs2d)
  in flatten ys2d

def pat_stencil2d [h][w] : [h*w][h*w]bool =
  let N : i64 = h * w
  in tabulate N (\p ->
       let r = p / w
       let c = p % w
       let east  = r*w + ((c + 1i64) % w)
       let west  = r*w + ((c + w - 1i64) % w)
       let south = (((r + 1i64) % h) * w) + c
       let north = (((r + h - 1i64) % h) * w) + c
       in tabulate N (\q ->
            q == p || q == east || q == west || q == south || q == north))

def mk_stencil_inputs (h:i64) (w:i64) : ([h*w][h*w]bool, [h*w]f64) =
  let x   : [h*w]f64 = rand_vec 42i64
  let pat : [h*w][h*w]bool = pat_stencil2d
  in (pat, x)

def mk_stencil_inputs_with_colors (h:i64) (w:i64)
  : ([h*w][h*w]bool, [h*w]i64, [h*w]f64) =
  let x   : [h*w]f64 = rand_vec 42i64
  let pat : [h*w][h*w]bool = pat_stencil2d

  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat

  let colors : [h*w]i64 =
    Col.partial_d2_color_cols row_offs row_idx col_offs col_idx

  in (pat, colors, x)

def mk_csr_stencil (h:i64) (w:i64) : ([h*w+1]i64, []i64, [h*w+1]i64, []i64) =
  let pat : [h*w][h*w]bool = pat_stencil2d
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat
  in (row_offs, row_idx, col_offs, col_idx)

-- ------------------------------------------------------------
-- Spiky-row synthetic BGPC case
--
-- Most rows have degree small_deg.
-- A small number of rows have degree big_deg.
-- The big rows are forced to overlap inside a smaller "hot" set
-- of columns to make coloring harder/more realistic.
--
-- Returns full bipartite CSR:
--   row_offs,row_idx : rows -> columns
--   col_offs,col_idx : columns -> rows
-- ------------------------------------------------------------

def row_is_big (i:i64) (m:i64) (num_big_rows:i64) : bool =
  let b =
    if num_big_rows < m then num_big_rows else m
  let spacing =
    if b <= 0i64 then 1i64
    else
      let s = m / b
      in if s < 1i64 then 1i64 else s
  in b > 0i64 && i % spacing == 0i64 && i / spacing < b

def mk_csr_spiky_rows (m:i64) (n:i64)
  (small_deg:i64) (big_deg:i64) (num_big_rows:i64)
  : ([m+1]i64, []i64, [n+1]i64, []i64) =
  let small_deg' =
    if small_deg < n then small_deg else n
  let big_deg' =
    if big_deg < n then big_deg else n

  -- Big rows overlap within a smaller hot subset of columns.
  let hot_cols =
    let h = n / 8i64
    let h1 = if h > big_deg' then h else big_deg'
    in if h1 > 0i64 then h1 else 1i64

  -- Row degrees.
  let row_degs : [m]i64 =
    map (\i ->
          if row_is_big i m num_big_rows
          then big_deg'
          else small_deg')
        (iota m)

  let row_offs : [m+1]i64 = CSR.counts_to_offs row_degs
  let nnz : i64 = row_offs[m]

  -- Build row-wise adjacency directly.
  let row_idx0 : []i64 = replicate nnz 0i64
  let row_ids0 : []i64 = replicate nnz 0i64

  let (row_idx, row_ids, _i) =
    loop (row_idx_acc, row_ids_acc, i) = (row_idx0, row_ids0, 0i64)
    while i < m do
      let d = row_degs[i]
      let start = row_offs[i]

      let big = row_is_big i m num_big_rows

      -- Small rows spread across all columns.
      let base_small = wrap (131i64 * i + 17i64 * (i / 7i64)) n

      -- Big rows overlap strongly inside the hot subset.
      let base_big = wrap (37i64 * i + 19i64 * (i / 5i64)) hot_cols

      let cols : []i64 =
        if big
        then map (\k -> wrap (base_big + k) hot_cols) (iota d)
        else map (\k -> wrap (base_small + k) n) (iota d)

      let rows : []i64 = replicate d i

      let row_idx_acc' =
        row_idx_acc with [start : start + d] = cols
      let row_ids_acc' =
        row_ids_acc with [start : start + d] = rows

      in (row_idx_acc', row_ids_acc', i + 1i64)

  -- Count column degrees.
  let col_counts0 : [n]i64 = replicate n 0i64

  let (col_counts, _p1) =
    loop (counts, p) = (col_counts0, 0i64)
    while p < nnz do
      let j = row_idx[p]
      let counts' = counts with [j] = counts[j] + 1i64
      in (counts', p + 1i64)

  let col_offs : [n+1]i64 = CSR.counts_to_offs col_counts

  -- Fill column-wise adjacency.
  let next0 : [n]i64 = map (\j -> col_offs[j]) (iota n)
  let col_idx0 : []i64 = replicate nnz 0i64

  let (col_idx, _next, _p2) =
    loop (col_idx_acc, next, p) = (col_idx0, next0, 0i64)
    while p < nnz do
      let j = row_idx[p]
      let pos = next[j]

      let col_idx_acc' =
        col_idx_acc with [pos] = row_ids[p]
      let next' =
        next with [j] = pos + 1i64

      in (col_idx_acc', next', p + 1i64)

  in (row_offs, row_idx, col_offs, col_idx)