-- Tests for row-wise CSR construction and bipartite CSR construction.

module CSR = import "../src/pattern_csr"

def eq_bool_mat [m][n] (a:[m][n]bool) (b:[m][n]bool) : bool =
  reduce (&&) true
    (map2 (\ra rb -> reduce (&&) true (map2 (==) ra rb)) a b)

def has (neigh:[]i64) (j:i64) : bool =
  reduce (||) false (map (== j) neigh)

def pat_from_csr [m][n] (offs:[m+1]i64) (idx:[]i64) : [m][n]bool =
  map (\i ->
         let s = offs[i]
         let e = offs[i+1]
         let neigh = idx[s:e]
         in map (\j -> has neigh j) (iota n))
      (iota m)

def pat_from_csr_n [m] (n:i64) (offs:[m+1]i64) (idx:[]i64) : [m][n]bool =
  map (\i ->
         let s = offs[i]
         let e = offs[i+1]
         let neigh = idx[s:e]
         in map (\j -> has neigh j) (iota n))
      (iota m)

def pat_ex1 : [3][5]bool =
  [ [true,  false, false, true,  false]
  , [false, true,  false, false, true ]
  , [false, false, true,  false, false]
  ]

-- ==
-- entry: test_csr_roundtrip_rows_ex1
-- input  { }
-- output { true }
entry test_csr_roundtrip_rows_ex1 : bool =
  let (offs, idx) = CSR.csr_rows_from_pattern pat_ex1
  let pat2 = pat_from_csr offs idx
  in eq_bool_mat pat_ex1 pat2

def pat_empty : [3][4]bool =
  [ [false, false, false, false]
  , [false, false, false, false]
  , [false, false, false, false]
  ]

-- ==
-- entry: test_csr_roundtrip_rows_empty_pattern
-- input  { }
-- output { true }
entry test_csr_roundtrip_rows_empty_pattern : bool =
  let (offs, idx) = CSR.csr_rows_from_pattern pat_empty
  let pat2 = pat_from_csr offs idx
  in eq_bool_mat pat_empty pat2

def pat_dense : [2][3]bool =
  [ [true, true, true]
  , [true, true, true]
  ]

-- ==
-- entry: test_csr_roundtrip_rows_dense_pattern
-- input  { }
-- output { true }
entry test_csr_roundtrip_rows_dense_pattern : bool =
  let (offs, idx) = CSR.csr_rows_from_pattern pat_dense
  let pat2 = pat_from_csr offs idx
  in eq_bool_mat pat_dense pat2

def pat_empty_rows : [4][5]bool =
  [ [false, false, false, false, false]
  , [false, false, true,  false, false]
  , [false, false, false, false, false]
  , [true,  false, false, false, true ]
  ]

-- ==
-- entry: test_csr_roundtrip_rows_with_empty_rows
-- input  { }
-- output { true }
entry test_csr_roundtrip_rows_with_empty_rows : bool =
  let (offs, idx) = CSR.csr_rows_from_pattern pat_empty_rows
  let pat2 = pat_from_csr offs idx
  in eq_bool_mat pat_empty_rows pat2

def pat_diag : [4][4]bool =
  [ [true,  false, false, false]
  , [false, true,  false, false]
  , [false, false, true,  false]
  , [false, false, false, true ]
  ]

-- ==
-- entry: test_csr_roundtrip_rows_diagonal_pattern
-- input  { }
-- output { true }
entry test_csr_roundtrip_rows_diagonal_pattern : bool =
  let (offs, idx) = CSR.csr_rows_from_pattern pat_diag
  let pat2 = pat_from_csr offs idx
  in eq_bool_mat pat_diag pat2

-- ==
-- entry: test_csr_bipartite_matches_rows_and_transpose_ex1
-- input  { }
-- output { true }
entry test_csr_bipartite_matches_rows_and_transpose_ex1 : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_ex1

  let (exp_row_offs, exp_row_idx) =
    CSR.csr_rows_from_pattern pat_ex1

  let (exp_col_offs, exp_col_idx) =
    CSR.csr_rows_from_pattern (transpose pat_ex1)

  in eq_bool_mat
       (pat_from_csr_n 5 row_offs row_idx)
       (pat_from_csr_n 5 exp_row_offs exp_row_idx)
     &&
     eq_bool_mat
       (pat_from_csr_n 3 col_offs col_idx)
       (pat_from_csr_n 3 exp_col_offs exp_col_idx)

-- ==
-- entry: test_csr_bipartite_matches_rows_and_transpose_empty_rows
-- input  { }
-- output { true }
entry test_csr_bipartite_matches_rows_and_transpose_empty_rows : bool =
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_empty_rows

  let (exp_row_offs, exp_row_idx) =
    CSR.csr_rows_from_pattern pat_empty_rows

  let (exp_col_offs, exp_col_idx) =
    CSR.csr_rows_from_pattern (transpose pat_empty_rows)

  in eq_bool_mat
       (pat_from_csr_n 5 row_offs row_idx)
       (pat_from_csr_n 5 exp_row_offs exp_row_idx)
     &&
     eq_bool_mat
       (pat_from_csr_n 4 col_offs col_idx)
       (pat_from_csr_n 4 exp_col_offs exp_col_idx)