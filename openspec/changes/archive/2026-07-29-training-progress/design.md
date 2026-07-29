# Design: Inter-iteration Training Progress Tracking

## Technical Approach

Add a new `src/Awale/Metrics.jl` submodule (four components) that observes the training loop and persists structured per-iteration metrics. Components are stateless value objects — train.jl owns lifecycle. Zero changes to Training.jl, MCTS.jl, Model.jl, Evaluation.jl, ReplayBuffers.jl.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| Elo rating scope | Latest-candidate vs best (not full rating pool) | Full TrueSkill / multi-model Elo | Matches eval structure: one candidate per iteration, best updates only on promotion |
| Param norm computation | `Flux.destructure` before/after `run_training_iteration` | Loss gradient norm | Measures weight-space convergence, not optimization — available without changing training internals |
| Promotion history format | TOML array-of-tables | CSV, JSON | Consistent with existing `training_state.toml` and config files; `[[promotions]]` is idiomatic TOML |
| CSV Library | Plain Julia (Printf @sprintf) | CSV.jl | No new external deps per constraint; CSV is simple enough for fixed-column append |
| Atomic write pattern | `mktemp` + `mv` (matches `Model.atomic_write`) | `write` directly | Prevents data loss on crash; reuses existing project pattern |
| Resume: Elo/Progress | Reset to initial values | Persist full state | Weights-only contract: optimizer, replay, RNG are not persisted — Elo derivable from CSV, Progress cheap to recompute |

## Data Flow

```
train.jl loop (iter N)
    │
    ├─ before run_training_iteration → Flux.destructure(model) → θ_old
    │
    ├─ run_training_iteration(...)    ← (no changes to this function)
    │
    ├─ after run_training_iteration
    │   ├─ Flux.destructure(model) → θ_new
    │   ├─ param_update_norm = norm(θ_new - θ_old)
    │   │
    │   ├─ EloTracker: candidate_elo = update(candidate_elo, W/L/D vs best)
    │   │                    └─ on promotion: best_elo = candidate_elo
    │   │
    │   ├─ ProgressTracker: record(iter, promoted, candidate_rate, elo)
    │   │                    └─ tracks last_best_iter, streaks, gaps, total_promotions
    │   │
    │   ├─ LearningCurveWriter: append_row(iter, loss, wr, param_norm, elo, ...)
    │   │   └─ learning_curve_<arch>_<release>.csv
    │   │
    │   ├─ PromotionRecord: on promotion → append to promotion_history.toml
    │   │
    │   ├─ print_progress_diagnostics(iter, tracker, elo, param_norm, buffer)
    │   │   └─ stdout: param norm, best iter, promos, gaps, Elo
    │   │
    │   └─ promotion event print → formatted "Promotion #N" message
    │
    └─ write_training_state(...) (existing, unchanged)
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/Awale/Metrics.jl` | **Create** | New submodule: ProgressTracker, EloTracker, LearningCurveWriter, PromotionRecord |
| `src/Awale.jl` | Modify | Add `include("Awale/Metrics.jl")`, `using .Metrics`, add to export |
| `train.jl` | Modify | ~30 lines: injection for metrics setup + per-iteration tracking + diagnostics |

## Interfaces / Contracts

```julia
# --- ProgressTracker ---
mutable struct ProgressTracker
    last_best_iter::Int           # iteration of most recent promotion
    total_promotions::Int         # count of all promotions
    gaps::Vector{Int}             # iter gap for each promotion
    longest_streak::Int           # max consecutive iters without underperformance
    current_streak::Int           # current consecutive iters above threshold
end

ProgressTracker()  # all fields zero
record!(t, iter, promoted)  # update streaks, gaps on promotion

# --- EloTracker ---
mutable struct EloTracker
    candidate::Float64   # current candidate Elo (updates each iter)
    best::Float64        # best-model Elo (updates only on promotion)
    k::Float64           # K-factor (default 32.0)
end

EloTracker(; k=32.0)  # candidate=1500, best=1500
update!(e, candidate_wr)  # Δ = K * (score - expected), where score = candidate_wr/100
promote!(e)                # best = candidate (called on promotion)

# --- LearningCurveWriter ---
struct LearningCurveWriter
    path::String
    io::Union{IO, Nothing}  # nothing = closed/error state
end

LearningCurveWriter(path)  # creates dir, writes header if not exists
append!(w, row)            # writes CSV row with Printf, atomic write per iter
close!(w)                  # flushes and closes

# --- PromotionRecord (value struct) ---
struct PromotionRecord
    iteration::Int
    candidate_vs_best_wr::Float64
    total_promotions_at_event::Int
    gap_since_last::Int
    elo_candidate::Float64
    elo_best::Float64
end
```

## Resume Logic

| Data | Resume Behavior | Rationale |
|------|----------------|-----------|
| `learning_curve_*.csv` | **Append**: file exists with header + N rows → write row N+1, no duplicate header | CSV is append-only log |
| `promotion_history.toml` | **Reload**: existing `[[promotions]]` entries loaded in-memory for diagnostics; new events appended | TOML array-of-tables |
| `EloTracker` | **Reset** to (1500, 1500) | Elo derivable from CSV history; weights-only contract |
| `ProgressTracker` | **Reset** to zeros | Computed from iteration events; cheap to rebuild |

**Resume in train.jl**: after the checkpoint-detection block, if `start_iter[] > 1`, metrics are initialized fresh (Elo=1500/1500, Progress=zeros). CSV writer opens existing file in append mode. Promotion history loaded into a `Vector{PromotionRecord}` from `promotion_history.toml` if it exists.

## File Locations

| File | Path (relative to checkpoint namespace dir) |
|------|--------------------------------------------|
| Learning curve CSV | `log/learning_curve_<arch>_<release_id>.csv` |
| Promotion history TOML | `promotion_history.toml` (same dir as `training_state.toml`) |

Both paths resolve via `checkpoint_namespace_dir()`. The CSV filename includes architecture slug + release ID for traceability across runs.

## Error Handling

| Scenario | Handling |
|----------|----------|
| CSV write failure (disk full) | `try/catch` around atomic write — logs warning to stderr, sets writer IO to `nothing`. Diagnostics still print. Training continues. |
| `promotion_history.toml` missing on resume | `isfile()` guard — return empty `Vector{PromotionRecord}`, no crash |
| Corrupt `promotion_history.toml` | `try/catch` on `TOML.parsefile` — log warning, return empty vector, training continues |
| Metrics init before log dir exists | `mkpath` in `LearningCurveWriter(path)` constructor — mirrors `training_log_dir()` pattern in train.jl |

## Injection Points in train.jl

The ~30 lines of injection occupy three zones (no other changes to train.jl):

```
ZONE A — Setup (before `for iter in ...`, ~10 lines)
├── using .Awale.Metrics: ProgressTracker, EloTracker, LearningCurveWriter, ...
├── θ_old_ref = Ref(...)
├── progress = ProgressTracker()
├── elo = EloTracker()
├── csv_path = joinpath(training_log_dir(), "learning_curve_<arch>_<release>.csv")
├── csv_writer = LearningCurveWriter(csv_path)
├── promo_history_path = joinpath(checkpoint_namespace_dir(), "promotion_history.toml")
└── (resume check: if start_iter > 1, load existing promo history)

ZONE B — Inside loop, before promotion block (~12 lines)
├── old_params, _ = Flux.destructure(model[])
├── (wait for run_training_iteration to complete)
├── new_params, _ = Flux.destructure(model[])
├── param_norm = norm(new_params - old_params)
├── update!(elo, win_rate / 100.0)                    # Elo update every iteration
├── if selection.promoted
│       promote!(elo)                                  # freeze best Elo
│       record!(progress, iter, true)                  # update streaks/gaps
│   end
├── append!(csv_writer, (iter, loss, win_rate, param_norm, elo.candidate, elo.best, ...))

ZONE C — Inside loop, after promotion block (~8 lines)
├── print_progress_diagnostics(iter, progress, elo, param_norm, length(replay_buffer), ...)
├── # Promotion event print (if selection.promoted: formatted "Promotion #N" message)
└── (existing write_training_state call unchanged)
```

**Exact anchoring lines**:
- ZONE A anchors after `agent_random = RandomAgent()` (line 599) and before the `if start_iter[] <= NUM_ITERATIONS` (line 601).
- ZONE B anchors after the loss/eval block (after line 628 `last_baseline_win_rate[] = ...`) and before `save_model(model[], ...)` (line 630).
- ZONE C anchors after `maybe_promote_best!` block (after line 651) and before snapshot check (line 653).

The design respects the existing code structure — no `Training.jl` modification needed.
