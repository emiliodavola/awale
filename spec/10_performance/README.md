10_performance: Performance constraints, data structures, and optimization plan

Performance principles
- Prioritize correctness first; optimize after tests and reference implementations are stable
- Minimize allocations in critical loops (sowing, MCTS rollout/backup)
- Use StaticArrays and preallocated buffers where appropriate

Data-structure choices
- GameState.board: SVector{12,UInt8}
- MCTS node arrays: use preallocated arrays or custom pool allocator to avoid garbage in hot loops
- Hashing: xxhash64 or SipHash with fixed seed; avoid platform-dependent behavior

Optimization roadmap
1. Profile hot paths (MCTS selection, expansion, backup)
2. Implement preallocated node pools and reuse nodes across searches where safe
3. Batch network inference across multiple expansions using batched predict_batch
4. Introduce transposition table with size limits and LRU eviction
5. GPU inference: isolate code paths and ensure deterministic CPU fallbacks for tests

Benchmarks
- Microbenchmarks for: single transition, legal action generation, single MCTS simulation, batched inference
- End-to-end benchmarks: self-play game time with fixed simulation count, training throughput (examples/sec)

Testable performance assertions
- Memory allocations per MCTS simulation under a threshold (measured by BenchmarkTools)
- Batched inference throughput scales with batch size up to target GPU memory limits