# Proposal: Inter-iteration training progress tracking

## Intent

Currently, the training loop prints per-iteration loss, win rate, and buffer size but **persists nothing historical**. After a run ends, there is no way to answer: did the parameter updates converge? How many promotions happened per run? What was the Elo trajectory? Researchers must re-run or hack ad-hoc logging. This change adds structured, persistent per-iteration metrics without altering training behavior.

## Scope

### In Scope
- New `src/Awale/Metrics.jl` module: `EloTracker`, `PromotionRecord`, `LearningCurveWriter`, `print_progress_diagnostics`
- Register module in `src/Awale/Awale.jl` (include + using + export)
- ~30 lines of injection in `train.jl` main loop (after loss/eval/promotion)
- Learning curve CSV at `checkpoints/<arch>/log/learning_curve.csv` (plain Julia, no CSV.jl dep)
- Promotion history CSV at `checkpoints/<arch>/log/promotions.csv`

### Out of Scope
- Optimizer/replay/RNG state persistence (resume contract stays "weights-only")
- Changes to `Training.jl`, `MCTS.jl`, `Model.jl`, `State.jl`, `Env.jl`
- Algorithm behavior changes
- Training dashboard or real-time visualization
- GPU metrics collection

## Capabilities

### New Capabilities
- `training-metrics`: Per-iteration tracking of parameter norms, Elo rating, promotion events, and learning curve persistence. Spec covers data structures, CSV schema, and integration points.

### Modified Capabilities
- None

## Approach

Add a pure-tracking module (`Metrics.jl`) with four components:

1. **EloTracker** — Simple 1D Elo (1 update per eval round), candidate vs best anchor, persisted in CSV.
2. **PromotionRecord** — Struct per promotion event (iteration, W/L/D, anchor results). Appended atomically to promotions CSV.
3. **LearningCurveWriter** — Appends one CSV row per iteration (loss, win_rate, param_norm, elo, replay_fill_pct).
4. **print_progress_diagnostics** — Formatted block after each iteration: param Δ norm, best streak, gaps, fill %.

Injection in `train.jl`: ~10 lines for setup (before loop), ~20 lines inside loop (after promotion block, before `write_training_state`). Metrics reset on restart (consistent with weights-only contract).

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/Awale/Metrics.jl` | **New** | 4 tracking components |
| `src/Awale/Awale.jl` | **Modified** | Add include + using + export for Metrics |
| `train.jl` | **Modified** | ~30 lines of injection in main loop |
| `config.toml` | **Optional** | `[metrics]` section for CSV flush interval |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CSV I/O slows training loop | Low | Append-only writes, buffers flush every N iterations |
| Metrics.jl becomes coupled to training details | Low | Pure data collection — no back-references to train.jl types |

## Rollback Plan

Revert `train.jl` injection lines, delete `Metrics.jl`, remove the `using`/`export` from `Awale.jl`. Zero operational impact — purely additive tracking.

## Dependencies

- Julia stdlib `Printf` for CSV formatting
- `Flux.destructure` for parameter norm (already a dependency)

## Success Criteria

- [ ] Parameter update norm printed after each iteration
- [ ] `learning_curve.csv` exists after 1+ iterations with correct schema
- [ ] `promotions.csv` records each promotion event with W/L/D
- [ ] EloTracker computes monotonic Elo updates per eval round
- [ ] All metrics survive a Ctrl+C restart (pick up from resumed iteration)
