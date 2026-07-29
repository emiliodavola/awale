# Training Metrics Specification

## Purpose

Add structured per-iteration tracking (parameter norms, Elo, promotions, learning curve CSV) to the training loop. All metrics are read-only observers — no algorithm behavior changes.

## Requirements

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| R1 | Compute `\|\|θ_new - θ_old\|\|_2` via `Flux.destructure` before/after `run_training_iteration` | MUST | Each iteration captures old params, runs training, captures new, computes norm, writes to CSV + diagnostics |
| R2 | Compute replay fill pct as `length(buffer)/capacity*100` | SHOULD | Diagnostics print `N/C*100`, not a cached value |
| R3 | `ProgressTracker` tracks `last_best_iter`, `total_promotions`, gaps, `longest_streak`, `current_streak`. On promotion: `gap = iter - last_best_iter` | MUST | First promo → `gaps=[50]`, `total_promotions=1`. Streak 10 broken at iter 60 → `current_streak=0`, `longest_streak≥10`, gap `10` appended |
| R4 | Persist each promotion as `PromotionRecord` to `checkpoints/<arch>/promotion_history.toml`. Reload on resume | MUST | Active run promos at iter N → `[[promotions]]` entry in TOML. Resume with 3 prior records → history loads those 3 |
| R5 | `EloTracker`: `E=1/(1+10^((best-candidate)/400))`, `Δ=K*(score-E)`. Candidate updates each iter, best only on promotion | MUST | Candidate 1500, best 1500, K=32, 55% WR → candidate rises. No promotion → best freezes at 1500 |
| R6 | CSV at `checkpoints/<arch>/log/learning_curve_<arch>_<release_id>.csv`, append mode, one row/iter. Header: `iteration,avg_loss,policy_loss,value_loss,grad_norm,pred_entropy,target_entropy,param_update_norm,replay_fill_pct,avg_game_len,baseline_wr,candidate_vs_best_wr,promoted,elo_candidate,elo_best` | MUST | No file → header + data. Existing file with 10 rows → row 11 appended, no duplicate header |
| R7 | Print diagnostics block each iter: param norm, best iter, iters since promo, total promos, avg gap, longest streak, Elo candidate/best | MUST | Iter 25, 2 promos, Elo 1520/1550 → all 7 fields printed |
| R8 | On promotion, print promotion #, iters since previous promo, win rate vs best | MUST | Promo #3 at iter 100, last at 65, 58.5% → prints `Promotion #3`, `Iterations since previous promotion: 35`, `Win rate vs previous best: 58.5%` |
| R9 | Resume: ProgressTracker resets, CSV appends, promo history reloads, EloTracker resets to initial | MUST | CSV with 42 rows → row 43 appended. Elo resets to 1500/1500 |

## Constraints

- No changes to `MCTS.jl`, `Training.jl`, `Model.jl`, `Evaluation.jl`, `ReplayBuffers.jl`
- No new external deps (stdlib `Printf` + `Flux.destructure` only)
- CSV writes MUST use temp-then-rename atomic pattern to prevent data loss on crash
- All existing tests MUST pass unchanged
- `promotion_history.toml` uses TOML array-of-tables (`[[promotions]]`)
