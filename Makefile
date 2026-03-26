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
	futhark bench benchmark/bench_sparse_pipeline.fut
	futhark bench benchmark/bench_sparse_jvp.fut
	futhark bench benchmark/bench_sparse_vjp.fut

# section: benches (CUDA backend)
bench-gpu:
	futhark bench --backend=cuda benchmark/bench_dense_jacobian.fut
	futhark bench --backend=cuda benchmark/bench_sparse_pipeline.fut
	futhark bench --backend=cuda benchmark/bench_sparse_jvp.fut
	futhark bench --backend=cuda benchmark/bench_sparse_vjp.fut

# section: cleanup
clean:
	rm -f test/*.c benchmark/*.c
	rm -f test/test_dense_jacobian \
	      test/test_pattern_csr \
	      test/test_partial_d2_coloring \
	      test/test_sparse_jacobian_jvp \
	      test/test_sparse_jacobian_vjp \
		  test/test_bgpc_vv_coloring
	rm -f benchmark/bench_dense_jacobian \
	      benchmark/bench_sparse_pipeline \
	      benchmark/bench_sparse_jvp \
	      benchmark/bench_sparse_vjp \
		  benchmark/bench_jvp_csr_fair \
		  benchmark/bench_jvp_csr_spiky