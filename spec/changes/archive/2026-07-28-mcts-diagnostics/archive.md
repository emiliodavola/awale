# Archive Report: mcts-diagnostics

**Date**: 2026-07-28
**Project**: awale
**Change**: mcts-diagnostics
**Store Mode**: hybrid (Engram + OpenSpec)

## Summary

Added 7 intra-iteration MCTS diagnostics metrics (KL, Top-K agreement, root confidence, target policy entropy enhanced, L1 policy distance, network drift) plus a formatted MCTS Diagnostics block. All metrics are read-only — they observe without modifying the training loop.

## Files Changed

| File | Change |
|------|--------|
| `src/Awale/Training.jl` | Added `MCTSDiagnostics` mutable struct, per-batch metric computation, formatted diagnostics block |
| `src/Awale/MCTS.jl` | No changes (confirmed — all metrics derive from existing return values) |
| `train.jl` | Added reference state set generation at iteration 1, per-iteration network drift computation, Network Drift output line |

## Metrics Implemented

| ID | Metric | Computation | Data Source |
|----|--------|-------------|-------------|
| R1 | KL Divergence | `sum(Y_pi .* log.(max.(after_probs, 1e-10f0)))` | `train_step` (zero extra cost) |
| R2 | Top-K Agreement | Argmax match + `topk` for Top-1/2/3 | `train_step` |
| R3 | Root Confidence | `max(visit_counts)` per root position | `search_with_stats` |
| R4 | Target Entropy (Enhanced) | `-sum(Y_pi .* log(clamp(Y_pi, 1e-10, 1.0)))` + distribution stats | `train_step` |
| R5 | L1 Policy Distance | `sum(abs.(after_probs - Y_pi)) / 2` | `train_step` |
| R7 | Network Drift | KL(softmax(ref_logits) \|\| softmax(curr_logits)) | One `predict_batch_inference` per iteration |

## Test Results

- **Total tests**: 432/432 pass (184 training + 73 publication + rest)
- **CRITICAL issue fixed**: Percentile computation on sorted arrays (commit `df0fccc`)
- **Verification**: PASS

## Spec Deviations (Intentional, Recorded)

| Spec | Implemented | Rationale |
|------|-------------|-----------|
| Network drift: cumulative | Per-iteration delta | More useful signal for short-term dynamics |
| Root confidence: per-game | Per-position | Richer distribution over all root visits |
| Reference set: self-play | Random play | Avoids conflating policy bias with drift measurement |

## Stale Checkbox Reconciliation

All 20 tasks were unchecked `[ ]` in the persisted tasks artifact, but apply-progress and verify-report proved every task complete (432/432 tests pass, verification PASS). Checkboxes were marked complete during archive as an exceptional mechanical reconciliation, per sdd-archive skill § Task Completion Gate.

## Archive Contents

### OpenSpec (`openspec/changes/archive/2026-07-28-mcts-diagnostics/`)
- `proposal.md` — scope, approach, risks, rollback plan
- `tasks.md` — 20 tasks, all complete (reconciled)

### Spec (`spec/changes/archive/2026-07-28-mcts-diagnostics/`)
- `specs/mcts-diagnostics/spec.md` — full delta specification with requirements, constraints, acceptance criteria, and test scenarios

## Specs Synced

| Domain | Action | Details |
|--------|--------|---------|
| Training (`spec/07_training/README.md`) | Updated | Added § MCTS Diagnostics section covering all 7 metrics, constraints, output format, percentile definition, and spec deviations |

## Source of Truth Updated

The main training spec at `spec/07_training/README.md` now reflects the diagnostics behavior.
