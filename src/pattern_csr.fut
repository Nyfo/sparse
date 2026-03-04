-- CSR (Compressed Sparse Row) from a dense boolean sparsity pattern.

def count_true [n] (row:[n]bool) : i64 =
  reduce (+) 0 (map (\b -> if b then 1i64 else 0i64) row)

def counts_to_offs [m] (counts:[m]i64) : [m+1]i64 =
  let pref : [m]i64 = scan (+) 0 counts
  in replicate (m+1) 0i64 with [1:m+1] = pref

def csr_rows_from_pattern [m][n] (pat:[m][n]bool) : ([m+1]i64, []i64) =
  let counts : [m]i64 = map count_true pat
  let offs   : [m+1]i64 = counts_to_offs counts

  let mask : [m*n]bool = flatten pat

  let cols_flat : [m*n]i64 = map (\k -> k % n) (iota (m*n))

  let pairs : [m*n](bool, i64) = map2 (\b c -> (b, c)) mask cols_flat
  let kept  : [](bool, i64)    = filter (\(b, _) -> b) pairs
  let idx   : []i64            = map (\(_, c) -> c) kept

  in (offs, idx)

def csr_cols_from_pattern [m][n] (pat:[m][n]bool) : ([n+1]i64, []i64) =
  csr_rows_from_pattern (transpose pat)

def csr_bipartite_from_pattern [m][n] (pat:[m][n]bool)
  : (([m+1]i64, []i64), ([n+1]i64, []i64)) =
  let rows = csr_rows_from_pattern pat
  let cols = csr_cols_from_pattern pat
  in (rows, cols)

-- Reference version: loop-based CSR:
def csr_rows_from_pattern_loop [m][n] (pat:[m][n]bool) : ([m+1]i64, []i64) =
  let counts : [m]i64 = map count_true pat
  let offs   : [m+1]i64 = counts_to_offs counts
  let nnz : i64 = offs[m]
  let idx = replicate nnz 0i64
  let i   = 0i64

  let (idx, _i) =
    loop (idx, i)
    while i < m do
      let cols : []i64 = filter (\j -> pat[i][j]) (iota n)
      let start : i64 = offs[i]
      let len   : i64 = length cols
      let idx' = idx with [start : start + len] = cols
      in (idx', i + 1i64)

  in (offs, idx)
