# Code Review & Next Steps - Awale RL System (v2.0)
<!-- Updated on 2026-06-15 -->

This document outlines the necessary, architecturally sound steps to align the implementation with research-grade specifications, ensuring strict compliance with codified contracts, particularly regarding state efficiency (Transposition Tables) and game completeness (Terminal States).

## 🚩 Critical Issues & Architectural Gaps Found

**The following gaps must be addressed in order to achieve a production-ready system:**

### 1. Game Termination Logic Completeness (`src/Awale/Env.jl`)

- **Grand Slam Win Condition:** The `is_terminal` function needs explicit logic for an immediate win (e.g., capturing all opponent seeds, or achieving full board control). This condition must be checked *before* checking for move availability.
- **Draw Detection (State History):** Implementing draw detection is critical. This must utilize the existing `history_hash` set within the `GameState` object to detect when a canonicalized state is encountered twice. A rule needs to be defined (e.g., 3 identical states) before declaring a draw and invoking a special termination signal that avoids counting it as a win/loss for score tracking.

### 2. MCTS Efficiency: Transposition Table Integration (`src/Awale/MCTS.jl`)

The architecture is structurally sound, but the TT must be actively used in all three major node lifecycle functions:

- **Key Generation:** A dedicated utility function, `hash_to_u64(canonicalized_state)`, must be created or confirmed to reliably map any canonical state representation to a unique `UInt64` key for dictionary lookup.
- **Selection (`select_puct`):** When selecting the best child action, check if the resulting state's hash exists in the global TT. If found, use this cached value to adjust the UCT calculation ($Q_{cached}$ and $N_{cached}$) to bias selection toward historically valuable branches immediately.
- **Expansion/Backup (`expand`, `backup`):** Every time a node is successfully expanded or backed up (i.e., after a move transition), the state's key must be written to, or updated in, the global TT with the latest aggregated $\langle Q_{total}, N_{visits} \rangle$ tuple.

### 3. Performance: Hot-Path Allocations & Memory Usage

The search loop (`Env.jl`, `MCTS.jl`) has several allocation hotspots that will cause excessive garbage collection overhead and performance degradation during deep simulations:

- **State Simulation:** Avoid `collect(s.board)` in `simulate_move`. Instead, pass or work directly with the underlying `StaticArray` structures to generate the next state without creating redundant temporary vectors.
- **Tensor Preparation:** Refactor `encode_state` and related tensor functions (`vcat`) to use optimized memory pooling or pre-allocated buffers (e.g., using `Base.fixed` types where possible) instead of generating intermediate arrays on every step.

---

## ⏳ Scheduled Start Date

**Work is paused until 2026-06-16.** The next steps detailed below will be executed on that date or later. Please review this plan and notify me when you are ready to resume the work.

---

## 🚀 Actionable Next Steps Roadmap

The effort should be structured into four distinct, sequential phases:

### Phase 1: Core Logic Fixes & State Completeness (High Priority)

[ ] **Implement Grand Slam Win Condition:** Update `is_terminal` in `src/Awale/Env.jl` to check for explicit win conditions.
[ ] **Implement Draw Detection:** Integrate draw logic into `is_terminal` using the existing state history (`Set{UInt64}`) mechanism in `GameState`.

### Phase 2: MCTS Architecture Upgrade (High Priority)

[ ] **Utility Layer:** Implement/confirm the robust `hash_to_u64` function.
[ ] **Integrate TT into Selection:** Refactor `select_puct` in `src/Awale/MCTS.jl` to query and bias UCT calculations using the transposition table cache (See Section 2 above).
[ ] **Write to TT:** Modify both `expand` and `backup` functions in `src/Awale/MCTS.jl` to ensure the global TT is always updated after a successful state transition.

### Phase 3: Hot-Path Optimization & Refactoring (Medium Priority)

[ ] **Refactor State Simulation:** Optimize move simulation (`simulate_move`) by minimizing allocations, avoiding general `collect()` calls on fixed/static arrays.
[ ] **Optimize Encoding:** Refactor state encoding functions to minimize memory churn during tensor construction in both `Env.jl` and `MCTS.jl`.

### Phase 4: Validation & Benchmarking (Verification)

[ ] **Unit Tests:** Add dedicated property tests for the new terminal conditions (Grand Slam, Draws).
[ ] **Benchmark:** Run systematic benchmarks to quantify speed-up *after* implementing both TT caching and memory optimizations. This comparison must validate performance improvements against a baseline of pure theoretical MCTS search time vs. optimized implementation.
