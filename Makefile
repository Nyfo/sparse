.PHONY: test test-gpu \
        bench-structured-cpu bench-structured-gpu \
        bench-ba-cpu bench-ba-gpu \
        bench-ht-cpu bench-ht-gpu \
        bench-coloring bench-precolored \
        clean

# section: tests
test:
	futhark test test/test_dense_jacobian.fut
	futhark test test/test_pattern_csr.fut
	futhark test test/test_partial_d2_coloring.fut
	futhark test test/test_bgpc_vv_coloring.fut
	futhark test test/test_sparse_jacobian_jvp.fut
	futhark test test/test_sparse_jacobian_vjp.fut
	futhark test test/test_sparse_jacobian_auto.fut
	futhark test benchmark/ba/test_jvp_ba_correctness.fut
	futhark test benchmark/ht/test_jvp_ht_correctness.fut

test-gpu:
	futhark test --backend=cuda test/test_dense_jacobian.fut
	futhark test --backend=cuda test/test_pattern_csr.fut
	futhark test --backend=cuda test/test_partial_d2_coloring.fut
	futhark test --backend=cuda test/test_bgpc_vv_coloring.fut
	futhark test --backend=cuda test/test_sparse_jacobian_jvp.fut
	futhark test --backend=cuda test/test_sparse_jacobian_vjp.fut
	futhark test --backend=cuda test/test_sparse_jacobian_auto.fut

# section: end-to-end benchmarks
bench-structured-cpu:
	futhark bench --backend=multicore --runs=10 benchmark/bench_jvp_structured.fut | tee results/structured_jvp_cpu.txt

bench-structured-gpu:
	futhark bench --backend=cuda --runs=10 --entry-point=bench_dense_jvp_banded5 benchmark/bench_jvp_structured.fut | tee results/structured_banded5_dense_gpu.txt
	futhark bench --backend=cuda --runs=10 --entry-point=bench_dense_jvp_stencil benchmark/bench_jvp_structured.fut | tee results/structured_dense_gpu.txt
	futhark bench --backend=cuda --runs=10 --entry-point=bench_sparse_jvp_banded5_bgpc_compressed benchmark/bench_jvp_structured.fut | tee results/structured_banded5_bgpc_gpu.txt
	futhark bench --backend=cuda --runs=10 --entry-point=bench_sparse_jvp_stencil_bgpc_compressed benchmark/bench_jvp_structured.fut | tee results/structured_stencil_bgpc_gpu.txt

bench-ba-cpu:
	futhark bench --backend=multicore --runs=10 benchmark/ba/bench_jvp_ba_simple.fut | tee results/ba_jvp_cpu.txt

bench-ba-gpu:
	futhark bench --backend=cuda --runs=10 --entry-point=bench_dense_jvp_ba benchmark/ba/bench_jvp_ba_simple.fut | tee results/ba_dense_gpu.txt
	futhark bench --backend=cuda --runs=10 --entry-point=bench_sparse_jvp_ba_bgpc_compressed benchmark/ba/bench_jvp_ba_simple.fut | tee results/ba_bgpc_gpu.txt

bench-ht-cpu:
	futhark bench --backend=multicore --runs=10 benchmark/ht/bench_jvp_ht_simple.fut | tee results/ht_jvp_cpu.txt

bench-ht-gpu:
	futhark bench --backend=cuda --runs=10 --entry-point=bench_dense_jvp_ht benchmark/ht/bench_jvp_ht_simple.fut | tee results/ht_dense_gpu.txt
	futhark bench --backend=cuda --runs=10 --entry-point=bench_sparse_jvp_ht_bgpc_compressed benchmark/ht/bench_jvp_ht_simple.fut | tee results/ht_bgpc_gpu.txt

# section: breakdown benchmarks
bench-coloring:
	futhark bench --backend=multicore --runs=10 benchmark/bench_coloring_structured.fut | tee results/structured_coloring_cpu.txt
	futhark bench --backend=multicore --runs=10 benchmark/ba/bench_coloring_ba.fut | tee results/ba_coloring_cpu.txt
	futhark bench --backend=multicore --runs=10 benchmark/ht/bench_coloring_ht.fut | tee results/ht_coloring_cpu.txt

bench-precolored:
	futhark bench --backend=multicore --runs=10 benchmark/bench_jvp_precolored_structured.fut | tee results/structured_precolored_cpu.txt
	futhark bench --backend=multicore --runs=10 benchmark/ba/bench_jvp_ba_precolored.fut | tee results/ba_precolored_cpu.txt
	futhark bench --backend=multicore --runs=10 benchmark/ht/bench_jvp_ht_precolored.fut | tee results/ht_precolored_cpu.txt

# section: cleanup
clean:
	rm -f test/*.c benchmark/*.c benchmark/ba/*.c benchmark/ht/*.c results/*.c
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
	      benchmark/bench_jvp_precolored_structured \
	      benchmark/color_counts_structured
	rm -f benchmark/ba/bench_jvp_ba_simple \
	      benchmark/ba/bench_jvp_ba_precolored \
	      benchmark/ba/bench_coloring_ba \
	      benchmark/ba/color_counts_ba \
	      benchmark/ba/bench_adbench_ba \
	      benchmark/ba/test_jvp_ba_correctness
	rm -f benchmark/ht/bench_jvp_ht_simple \
	      benchmark/ht/bench_jvp_ht_precolored \
	      benchmark/ht/bench_coloring_ht \
	      benchmark/ht/color_counts_ht \
	      benchmark/ht/bench_adbench_ht \
	      benchmark/ht/test_jvp_ht_correctness
	rm -f results/color_counts_ba \
	      results/color_counts_ht \
	      results/color_counts_structured