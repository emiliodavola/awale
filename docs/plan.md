# Code Review & Next Steps - Awale RL System

This document summarizes the findings of the comprehensive code review performed on 2026-06-15 and outlines the necessary steps to align the implementation with the research-grade specifications.

## 🚩 Critical Issues Found

### 1. Missing Game-Ending Conditions

* **Location:** `src/Awale/Env.jl`
* **Severity:** High
* **Description:** The current `is_terminal` logic only checks for seed capture counts (25) or lack of legal moves. It fails to account for:
  * **Grand Slam:** Immediate win when a player captures all opponent seeds or performs a full-capture.
  * **Repetitions (Draws):** No mechanism is implemented to detect repeated game states using the `history_hash`, which should trigger a draw.

### 2. Missing Transposition Table in MCTS

* **Location:** `src/Awale/MCTS.jl`
* **Severity:** High
* **Description:** The MCTS implementation currently uses a standard tree structure (`Dict{Int, MCTSNode}`). This violates the specification in `spec/05_mcts/README.md`. Without a transposition table, identical board positions reached via different move sequences are explored redundantly, severely limiting search efficiency and depth.

### 3. Performance Bottlenecks (Excessive Allocations)

* **Location:** `src/Awale/Env.jl`, `src/Awale/Model.jl`
* **Severity:** Medium
* **Description:** The "hot path" of the search loop contains several allocation hotspots that will drastically slow down MCTS simulations:
  * `simulate_move`: Frequent calls to `collect(s.board)` creating new vectors.
  * `encode_state`: Multiple allocations during tensor preparation (array creation and `vcat`).
  * `serialize_state`: Allocating a new `Vector{UInt8}` for every hash calculation.

---

## 🚀 Required Next Steps

### Phase 1: Rule Alignment & Correctness

[ ] **Implement Grand Slam logic** in `is_terminal` (`src/Awale/Env.jl`).
[ ] **Implement Repetition Detection**: Integrate a state tracking mechanism (using `history_hash`) to detect draws.

### Phase 2: MCTS Architectural Upgrade

[ ] **Design and implement the Transposition Table**: Move from a simple tree to a global lookup table based on canonicalized state hashes.
[ ] **Refactor `select_and_expand`** in `MCTS.jl` to utilize the transposition table for node retrieval.

### Phase 3: Hot-Path Optimization

[ ] **Eliminate allocations in `simulate_move`**: Avoid `collect()` on StaticArrays; work with them directly or use pre-allocated buffers.
[ ] **Optimize `encode_state`**: Refactor the state-to-tensor pipeline to minimize intermediate array creations.
[ ] **Optimize `serialize_state`**: Implement a non-allocating serialization method for hashing.

### Phase 4: Validation & Benchmarking

[ ] **Add Property Tests** for the new terminal conditions (Grand Slam, Draws).
[ ] **Benchmark MCTS search depth/speed** before and after Transposition Table and allocation optimizations to quantify improvement.
