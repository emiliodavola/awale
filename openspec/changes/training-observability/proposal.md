# Proposal: Training Observability

## Intent

Today, MCTS diagnostics and network drift are **printed then discarded** — lost on crash, invisible to analysis. No versioning, history, or convergence detection. Fix all 19 gaps with zero behavior change.

## Scope

### In Scope

| Phase | Deliverables |
|-------|-------------|
| **1 — Docs** | Rename "Replay Coverage" → "Replay Fill %". Document all metric formulas, code locations, and interpretation with observed values |
| **2 — Persistence** | JSONL with ALL fields + versioning. Enriched `PromotionRecord` (gap, total, elo, timestamp). Historical promo summary |
| **3 — New Metrics** | Larger Elo K + delta. Value calibration (MSE/MAE/Pearson/Spearman). MCTS root Q vs network value. Δ-metrics. Moving averages (5/10/20). Cost docs |
| **4 — Dashboard** | Passive convergence detection. Health indicators. Warnings. Reuse audit |

### Out of Scope

No changes to training algorithm, MCTS behavior, self-play, promotion, hyperparams, optimizer, scheduler, replay buffer. No web dashboard.

## Capabilities

### New
- `training-observability`: Enhanced instrumentation, JSONL persistence, promotion enrichment, versioning, convergence detection, health dashboard

### Modified
- `training-metrics`: Delta for value calibration, Elo, Δ-metrics, moving averages, JSONL, enriched `PromotionRecord`, renamed "Replay Coverage" → "Replay Fill %"

## Approach

| Step | File | Change |
|------|------|--------|
| 1 | `Training.jl` | Extend `TrainingResult` with MCTS aggregates. Backward-compatible |
| 2 | `MCTS.jl` | Add read-only root Q to `search_with_stats` return |
| 3 | `Metrics.jl` | Enrich `PromotionRecord`. Larger Elo K. Convergence/health. Versioning |
| 4 | `train.jl` | JSONL injection. Dual-write alongside CSV. Convergence detection |
| 5 | `docs/metrics.md` | (NEW) Formula + interpretation + cost per metric |

JSONL: `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl`. `JSON3.jl`, append-mode.

## Affected Areas

`Metrics.jl`, `Training.jl`, `MCTS.jl` (root Q return), `train.jl`, `docs/metrics.md` (new), `openspec/specs/training-metrics/spec.md` (delta)

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| JSONL I/O overhead | Low | ~1KB/iter vs training compute |
| MCTS.jl change vs no-MCTS constraint | Low | Read-only return — no algorithm path changed |
| TrainingResult bloat | Low | Same pattern as existing aggregate fields |

## Rollback Plan

Delete `.jsonl`, revert train.jl/`TrainingResult`/`PromotionRecord` changes. Pure additive.

## Success Criteria

- [ ] All 19 areas addressed
- [ ] JSONL contains every metric + versioning per run
- [ ] `PromotionRecord` has all 5 enrichment fields
- [ ] "Replay Coverage" → "Replay Fill %" everywhere
- [ ] Convergence/health prints without behavioral feedback
- [ ] All tests pass unchanged
