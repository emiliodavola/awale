# Proposal: Intra-Iteration MCTS Diagnostics

## Intent

Add telemetry to measure MCTS vs network policy alignment inside each training iteration -- no algorithm, hyperparameter, or behavior changes. Current diagnostics show loss and entropy but no visibility into whether network predictions converge toward the MCTS target or drift over training.

## Scope

### In Scope
- `src/Awale/Training.jl` -- stats accumulator struct, compute metrics 1–5 per batch, print MCTS Diagnostics block
- `train.jl` -- reference state set creation at iteration 1, compute Network Drift (metric 7) post-iteration, print Network Drift line

### Out of Scope
- Any change to `MCTS.jl`, `Model.jl`, `ReplayBuffers.jl` -- zero modifications
- Any algorithm, hyperparameter, search behavior, or training loop change
- Metrics not derivable from already-computed values (zero new forward passes except drift)

## Capabilities

### New Capabilities
- `mcts-diagnostics`: new intra-iteration telemetry for MCTS policy alignment -- 7 metrics printed in a dedicated diagnostics block

### Modified Capabilities
None -- pure additive metrics

## Approach

Two additive changes, zero training-performance impact:

**Phase 1: Intra-iteration metrics (Training.jl)**
- Add `MCTSDiagnostics` struct holding accumulators for: KL divergence (derived from policy_loss − mean_target_entropy), Top-1/2/3 agreement, root confidence (max visit counts per game), per-sample target entropy stats, and L1 policy distance.
- All values are already computed inside `train_step()` (after_probs, Y_pi, after_log_probs) or available from `collect_selfplay_data` (visit-count policy). Zero new forward passes for metrics 1–5.
- Accumulate per batch in `run_training_iteration()`, print a `MCTS Diagnostics` block after the existing Diagnostics block with avg/median/P25/P75/P95/P99/min/max as appropriate.

**Phase 2: Network drift (train.jl)**
- At iteration 1, sample and persist a reference set of ~128–256 canonicalized states from self-play.
- After each iteration, run `predict_batch_inference()` on the reference set and compute KL divergence between current network output and iteration-1 output.
- This is the ONLY metric needing new forward passes: one `predict_batch_inference` call per iteration (plus one at iteration 1 to store reference).

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/Awale/Training.jl` | Modified | Add `MCTSDiagnostics` accumulator, compute metrics 1–5 per batch, reformat diagnostics block |
| `train.jl` | Modified | Add reference set creation at iter 1, compute Network Drift post-iteration, print MCTS Diagnostics |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Accumulator overhead | Low | Struct + array push -- trivial cost vs forward pass |
| Reference set staleness | Low | Fixed set from iter 1 -- drift is intentional |
| GPU memory for drift | Low | 128 states x 48 features -- negligible |

## Rollback Plan

Revert the two file changes. The accumulator struct and reference set have zero coupling to other modules.

## Dependencies

- `predict_batch_inference` from `Model.jl` -- already exists, no changes needed
- `MCTS.jl` `search_with_stats` -- already returns visit-count policy, no changes needed

## Success Criteria

- [ ] All 7 metrics print in a dedicated `MCTS Diagnostics` block each iteration
- [ ] Training loop produces identical losses to baseline (zero numerical drift)
- [ ] All existing tests pass unchanged
