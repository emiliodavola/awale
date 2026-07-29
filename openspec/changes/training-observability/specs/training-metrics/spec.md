# Delta for training-metrics

## ADDED Requirements

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| R10 | **Value calibration**: Compute MAE, Pearson r, Spearman ρ between network value prediction (`v_hat`) and actual return. MUST reuse existing forward pass — no extra inference | MUST | Batch of 1000 → `{mae:0.28, pearson:0.62, spearman:0.58}` from already-computed value head outputs |
| R11 | **Enhanced Elo**: Compute expected score `E = 1/(1 + 10^((best − candidate)/400))`, actual score `S = WR/100`, rating delta `Δ = K·(S − E)`, upset size `= |S − E|`. K configurable, default 64 | MUST | Candidate 1520, best 1550, WR 55% → E≈0.46, S=0.55, Δ=+5.76, upset=0.09. Written to JSONL |
| R12 | **Δ-metrics**: Compute `ΔKL = |KL_i − KL_{i−1}|`, ΔPolicy distance, ΔTop1, ΔDrift, ΔGrad norm, ΔParam update. From consecutive iterations. JSONL-only | SHOULD | KL_i=0.45, KL_{i−1}=0.42 → ΔKL=0.03. First iteration yields NaN for all Δ-metrics |
| R13 | **Moving averages**: Rolling mean over windows of 5, 10, 20 iterations for: avg_loss, policy_loss, value_loss, KL, Top1, drift, rating, entropy. NaN until window fully populated. JSONL-only | SHOULD | Iter 25, window-5 → mean of iters 21–25. Iter 3, window-5 → NaN |
| R14 | **JSONL writer**: Alongside existing CSV, write one JSONL entry per iteration to `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl`. Append-mode, flush per iteration, no header row | MUST | File with 10 lines → line 11 appended. No file → created. Entry includes ALL existing CSV fields + all new ADDED fields |

## MODIFIED Requirements

### R1 — Parameter update norm (computation reuse)

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| R1 | Compute `||θ_new − θ_old||_2` via `Flux.destructure` before/after `run_training_iteration`. Reuse the destructured vector for drift computation — no separate capture call needed | MUST | Each iteration captures params once via `destructure`, computes update norm and drift KL from same vector |

(Previously: explicit capture call required. Reuse note added — destructured vector already available for drift)

### R4 — Enriched PromotionRecord

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| R4 | Persist each promotion as `PromotionRecord` to `checkpoints/<arch>/promotion_history.toml`. Reload on resume. `PromotionRecord` fields: `iteration`, `win_rate_vs_best`, `wins`, `losses`, `draws`, `random_anchor_wr`, `promotion_score`, `gap_since_last`, `total_promotions_at_event`, `elo_candidate`, `elo_best`, `timestamp` | MUST | Promo at iter 100, gap 35 from last, total=3, elo_candidate=1540, elo_best=1520 → TOML has all 12 fields. Resume loads prior records with enriched fields |

(Previously: 7 fields. Added gap_since_last, total_promotions_at_event, elo_candidate, elo_best, timestamp)

### R5 — Elo with larger K

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| R5 | `EloTracker`: `E=1/(1+10^((best-candidate)/400))`, `Δ=K·(score-E)`. K configurable, default 64. Candidate updates each iter, best-only on promotion | MUST | Candidate 1500, best 1500, K=64, 55% WR → Δ=+5.76, candidate rises faster. No promotion → best frozen |

(Previously: K=32 fixed. Now K configurable, default 64)

### R6 — JSONL dual-write

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| R6 | CSV at `checkpoints/<arch>/log/learning_curve_<arch>_<release_id>.csv`, append, one row/iter. Header: `iteration,avg_loss,policy_loss,value_loss,grad_norm,pred_entropy,target_entropy,param_update_norm,replay_fill_pct,avg_game_len,baseline_wr,candidate_vs_best_wr,promoted,elo_candidate,elo_best`. MUST also write JSONL at `metrics_<arch>_<release_id>.jsonl` with ALL columns + new metrics (R10–R13) | MUST | No CSV → header + row 1. CSV with 10 rows → row 11, no duplicate header. JSONL with 10 lines → line 11, no header |

(Previously: CSV only. Now CSV + JSONL dual-write)

## RENAMED Requirements

### "Replay coverage" → "Replay Fill %"

(Reason: Current label is misleading — metric is `length(buffer)/capacity × 100`, i.e. buffer fill percentage, not state-space coverage)
(Migration: Update all diagnostic print labels and spec references. CSV column name `replay_fill_pct` unchanged — rename applies to human-readable labels only)
