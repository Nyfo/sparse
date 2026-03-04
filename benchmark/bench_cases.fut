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