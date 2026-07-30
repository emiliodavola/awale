# Tasks: training-observability

## Review Workload Forecast

| Metric | Value |
|--------|-------|
| Estimated new/changed lines | ~580 |
| Review budget | 1200 |
| **Decision needed before apply** | No |
| **Chained PRs recommended** | No |
| **Chain strategy** | size-exception |
| **400-line budget risk** | Low |

380 lines of source changes + ~200 lines of docs + ~160 lines of tests ≈ 740 lines peak, well inside the 1200-line budget. No splitting needed.

---

## Phase 1 — TrainingResult + MCTS root Q (Foundation)

- [x] **T1** (`src/Awale/Training.jl`): Add 7 fields (`kl_mean`, `kl_median`, `top1_pct`, `top2_pct`, `top3_pct`, `root_conf_mean`, `l1_mean`) to `TrainingResult` struct.
- [x] **T2** (`src/Awale/Training.jl`): Compute the 7 aggregate values from existing vectors (`all_kl_per_position`, `all_top1/2/3`, `all_root_confidences`, `all_l1_per_position`) after the diagnostics print block and pass them into both `TrainingResult` constructor calls.
- [x] **T3** (`src/Awale/MCTS.jl`): Add `root_q = root.value_sum[] / max(1, root.visits[])` after the sim loop, return 4-tuple `(best_action, counts, maximum(counts), root_q)` instead of 3.
- [x] **T4** (`src/Awale/MCTS.jl`): Update `search` wrapper destructuring to ignore 4th value. Update `num_sims ≤ 0` early return to include `0.0f0` as 4th value.
- [x] **T5** (`src/Awale/Training.jl`): Update `collect_selfplay_data` destructuring to destructure 4 values and push `root_q` into new `all_root_q` accumulator.
- [x] **T6**: Write unit tests verifying `TrainingResult` new fields behave correctly (zero on no-data path, populated on normal path).

## Phase 2 — Metrics.jl enrichment

- [x] **T7** (`src/Awale/Metrics.jl`): Add 5 fields to `PromotionRecord`: `gap_since_last::Int`, `total_promotions_at_event::Int`, `elo_candidate::Float64`, `elo_best::Float64`, `timestamp::String`.
- [x] **T8** (`src/Awale/Metrics.jl`): Change `EloTracker` default `k` from 32 to 64.
- [x] **T9** (`src/Awale/Metrics.jl`): Add `compute_value_calibration(v_hat, returns)` returning `(mae, pearson_r, spearman_rho)`. Reuse incoming predictions — no extra forward pass.
- [x] **T10** (`src/Awale/Metrics.jl`): Add `print_historical_summary(pt::ProgressTracker)` printing total promos, avg/median/best/closest WR, largest gap, longest streak, avg gap.
- [x] **T11** (`src/Awale/Metrics.jl`): Rename all diagnostic print labels from "Replay Coverage" → "Replay Fill %".
- [x] **T12** (`train.jl`): Update `PromotionRecord` construction call sites (lines 795–803) and `save_promotion_history`/`load_promotion_history!` (lines 541–592) to include all 5 new enrichment fields.
- [x] **T13**: Write tests for PromotionRecord TOML roundtrip, Elo K=64 vs K=32 delta, value calibration on perfect predictions, and historical summary formatting.

## Phase 3 — JSONL persistence + train.jl injection

- [x] **T14** (`train.jl`, setup): Open JSONL file handle at `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl` (append mode, no header). Capture `metric_version`, `git_commit`, `architecture`, `config_hash` (SHA256 of active config). Store in local vars.
- [x] **T15** (`train.jl`, per-iteration): Compute iteration wall-clock duration via `time_ns()` around the iteration body.
- [x] **T16** (`train.jl`, after MCTS aggregates): Compute value calibration by reusing the drift forward pass data (or a shared batch) — call `compute_value_calibration` with network value predictions vs actual returns from the training step.
- [x] **T17** (`train.jl`, after drift): Compute Elo expected score, actual score, delta, and upset from `elo_tracker` and promotion evaluation results.
- [x] **T18** (`train.jl`): Compute Δ-metrics: `ΔKL`, ΔPolicy distance, ΔTop1, ΔDrift, ΔGrad norm, ΔParam update. Store previous iteration values in local vars; first iteration yields `NaN`.
- [x] **T19** (`train.jl`): Compute moving averages (5/10/20 windows) for avg_loss, policy_loss, value_loss, KL, Top1, drift, rating, entropy. Maintain ring buffers; `NaN` until window full.
- [x] **T20** (`train.jl`, convergence): Implement sliding window (20 iter) stability checks for KL, drift, Top1, param update magnitude. Print `ACTIVE`/`STALLED` per metric. Passive observation only — no control flow change.
- [x] **T21** (`train.jl`, health dashboard): Build single-line string: `Net:ACTIVE Srch:HIGH Drift:LOW Promo:1/50 Rply:82% ValCal:OK`. Print each iteration.
- [x] **T22** (`train.jl`, warnings): Print `⚠` warnings when Top-1 > 95%, drift ≈ 0, or param update > 5× rolling mean.
- [x] **T23** (`train.jl`): Assemble JSONL dict with ALL fields (CSV fields + MCTS aggregates + drift KL + root Q + value calibration + Elo + Δ-metrics + moving averages + versioning). Write via `JSON.json`, flush, one line per iteration.
- [x] **T24** (`train.jl`): Ensure `jsonl_io` closes in the `finally` block alongside `csv_io`.
- [x] **T25**: Write integration tests: JSONL file created and appended correctly, dual-write produces matching CSV + JSONL values, "Replay Fill %" grep returns zero old-label hits in print paths.

## Phase 4 — Documentation

- [x] **T26** (`docs/metrics.md`, NEW): Document every metric with formula, Julia code location, interpretation guidance, observed values/ranges, and the O9-required cost table: time (O expression + μs), memory (bytes/iter), frequency, and reuse opportunities.
