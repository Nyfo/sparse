-- Tests for CSR construction

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

-- ==
-- entry: test_csr_roundtrip_cols_ex1
-- input  { }
-- output { true }
entry test_csr_roundtrip_cols_ex1 : bool =
  let patT = transpose pat_ex1
  let (offs, idx) = CSR.csr_rows_from_pattern patT
  let pat2 = pat_from_csr offs idx
  in eq_bool_mat patT pat2
