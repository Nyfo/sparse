.PHONY: test bench bench-gpu clean

# section: tests (C backend)
test:
	futhark test test/test_dense_jacobian.fut
	futhark test test/test_pattern_csr.fut
	futhark test test/test_partial_d2_coloring.fut
	futhark test test/test_sparse_jacobian_jvp.fut
	futhark test test/test_sparse_jacobian_vjp.fut

# section: tests (CUDA backend)
test-gpu:
	futhark test --backend=cuda test/test_dense_jacobian.fut
	futhark test --backend=cuda test/test_pattern_csr.fut
	futhark test --backend=cuda test/test_partial_d2_coloring.fut
	futhark test --backend=cuda test/test_sparse_jacobian_jvp.fut
	futhark test --backend=cuda test/test_sparse_jacobian_vjp.fut

# section: benches (C backend)
bench:
	futhark bench benchmark/bench_dense_jacobian.fut
	futhark bench benchmark/bench_jvp_structured.fut
	futhark bench benchmark/bench_vjp_structured.fut
	futhark bench benchmark/bench_jvp_spiky.fut

# section: benches (CUDA backend)
bench-gpu:
	futhark bench --backend=cuda benchmark/bench_dense_jacobian.fut
	futhark bench --backend=cuda benchmark/bench_jvp_structured.fut
	futhark bench --backend=cuda benchmark/bench_vjp_structured.fut
	futhark bench --backend=cuda benchmark/bench_jvp_spiky.fut

# section: cleanup
clean:
	rm -f test/*.c benchmark/*.c benchmark/ba/*.c benchmark/ht/*.c
	rm -f test/test_dense_jacobian \
	      test/test_pattern_csr \
	      test/test_partial_d2_coloring \
	      test/test_sparse_jacobian_jvp \
	      test/test_sparse_jacobian_vjp \
	      test/test_sparse_jacobian_auto \
	      test/test_bgpc_vv_coloring
	rm -f benchmark/bench_dense_jacobian \
	      benchmark/bench_jvp_structured \
	      benchmark/bench_vjp_structured \
	      benchmark/bench_coloring_structured \
	      benchmark/color_counts_structured
	rm -f benchmark/ba/bench_jvp_ba_simple \
	      benchmark/ba/bench_coloring_ba \
	      benchmark/ba/color_counts_ba \
	      benchmark/ba/test_jvp_ba_correctness
	rm -f benchmark/ht/bench_jvp_ht_simple \
	      benchmark/ht/test_jvp_ht_correctness
