# Tasks: Inter-iteration Training Progress Tracking

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~300–350 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: pending
400-line budget risk: Low

## Phase 1: Foundation — Core Structs & Constructors

- [ ] 1.1 Create `src/Awale/Metrics.jl` with `ProgressTracker` struct (last_best_iter, total_promotions, gaps, longest_streak, current_streak) and `record!()` method
- [ ] 1.2 Add `EloTracker` struct (candidate, best, k) with `update!()` and `promote!()` using Elo formula: E=1/(1+10^((best-candidate)/400)), Δ=K*(score-E)
- [ ] 1.3 Add `PromotionRecord` immutable struct (iteration, candidate_vs_best_wr, total_promotions_at_event, gap_since_last, elo_candidate, elo_best)
- [ ] 1.4 Add `LearningCurveWriter` struct (path, io) with constructor that creates dir via `mkpath`, writes CSV header if file is new, opens in append mode
- [ ] 1.5 Implement `LearningCurveWriter.append!()` — format CSV row via Printf.@sprintf, atomic write via `mktemp`+`mv` (reuse pattern from `Model.atomic_write`)
- [ ] 1.6 Implement `PromotionRecord.append_to_toml!()` — serialize `[[promotions]]` entry, atomically write to `promotion_history.toml`
- [ ] 1.7 Add `load_promotion_history(path)` — `isfile` guard, `try/catch` on `TOML.parsefile`, return `Vector{PromotionRecord}` or empty on missing/corrupt
- [ ] 1.8 Implement `print_progress_diagnostics(iter, progress, elo, param_norm, replay_fill)` — formatted stdout block with all 7 fields from R7

## Phase 2: Module Registration

- [ ] 2.1 Add `include("Awale/Metrics.jl")` to `src/Awale.jl`
- [ ] 2.2 Add `using .Metrics` and export new symbols from `src/Awale.jl` (or add to existing `export *`)

## Phase 3: train.jl Injection

- [ ] 3.1 Add `using .Awale.Metrics: ...` import to train.jl header
- [ ] 3.2 **ZONE A** (before main loop, ~10 lines): instantiate `ProgressTracker()`, `EloTracker()`, `LearningCurveWriter(csv_path)`, compute CSV path via `training_log_dir()`, compute promo_history path, load existing promotions on resume (R9)
- [ ] 3.3 **ZONE B** (inside loop, after loss/eval block, ~12 lines): capture old params via `Flux.destructure(model[])`, wait for training iteration, capture new params, compute `param_norm`, update elo each iter, on promotion call `promote!(elo)` + `record!(progress, iter, true)`, `append!(csv_writer, ...)` (R1, R5, R6)
- [ ] 3.4 **ZONE C** (inside loop, after promotion block, ~8 lines): call `print_progress_diagnostics(...)`, on promotion print formatted promotion event (R7, R8)

## Phase 4: Testing

- [ ] 4.1 Unit test `EloTracker`: verify Elo formula, Δ=K*(score-E), candidate-only updates, best freezes on no promotion, `promote!` copies candidate→best (R5)
- [ ] 4.2 Unit test `ProgressTracker`: verify gap computation, streak tracking, multiple promotions (R3)
- [ ] 4.3 Unit test `LearningCurveWriter`: verify header write on new file, append no-header on existing file, row format matches speculation, atomic write resilience (R6)
- [ ] 4.4 Unit test `load_promotion_history`: missing file → empty, valid TOML → parsed records, corrupt TOML → empty with warning (R4, R9)
- [ ] 4.5 Integration test: verify `train.jl` with 2 iterations produces CSV with correct rows, promotion history TOML, and diagnostics print without crash (R1–R9)
- [ ] 4.6 Run full test suite — all existing tests MUST pass unchanged (constraint)

## Phase 5: Verification

- [ ] 5.1 Smoke test: run `julia --project=. train.jl` with low iteration count, confirm CSV + TOML files created and readable
- [ ] 5.2 Resume test: kill training mid-run, restart, confirm CSV appends (no duplicate header), promos reloaded, Elo resets to 1500/1500 (R9)
