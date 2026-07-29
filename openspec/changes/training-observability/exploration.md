## Exploration: training-observability

### Current State

The training loop has **three persistence mechanisms** and **two console-only diagnostic blocks**, all producing metrics at the per-iteration granularity.

**Persisted:**
1. **Learning curve CSV** (`checkpoints/<arch>/log/learning_curve_<arch>_<release_id>.csv`) — one row per iteration, append-mode, 15 columns. Written at line 826 of `train.jl`.
2. **Promotion history TOML** (`checkpoints/<arch>/promotion_history.toml`) — `[[promotions]]` array-of-tables, appended atomically on each promotion event (writes via `save_promotion_history` at line 814, full file rewrite each time).
3. **Training state TOML** (`checkpoints/<arch>/training_state.toml`) — `last_iter`, `best_selection_score`. Written at line 848.

**Console-only (printed but never persisted):**
1. **MCTS Diagnostics block** — KL(target || network) distribution, Top-1/2/3 agreement %, Root confidence distribution, Policy L1 distance distribution, Target policy entropy distribution. All accumulated in `all_kl_per_position`, `all_top1/2/3`, `all_root_confidences`, `all_l1_per_position`, `all_pos_entropy` vectors inside `run_training_iteration` — printed at lines 350–425 of `Training.jl`, discarded when the function returns.
2. **Network Drift block** — `avg_drift_kl` computed from reference set before/after each iteration (lines 718–752 of `train.jl`). Printed, then discarded.
3. **Progress diagnostics** — param_update_norm, best iteration, iters since promotion, total promotions, avg gap, longest streak, Elo candidate/best (line 846 of `train.jl`).
4. **Promotion event** — formatted "Promotion #N" message with gap and win rate (line 809).

All console-only diagnostics are **lost on crash, lost between runs, and invisible to any external tool**.

### Current CSV Schema

15 columns, written at `train.jl:826-843`:

```
iteration,avg_loss,policy_loss,value_loss,grad_norm,pred_entropy,target_entropy,
param_update_norm,replay_fill_pct,avg_game_len,baseline_wr,candidate_vs_best_wr,
promoted,elo_candidate,elo_best
```

Key observations:
- `candidate_vs_best_wr` is `nothing` on bootstrap (no best to compare against) — CSV gets an empty cell.
- `promoted` is a raw `yes`/`no` string.
- No MCTS diagnostics (KL, Top-K, L1, Root confidence), no network drift, no timestamps, no wall-clock duration.

### Console-Only Metrics

| Metric | Where computed | Where printed | Can be derived from CSV? |
|--------|----------------|---------------|--------------------------|
| KL(target\|\|network) mean/med/P25/P75/P95/max | `Training.jl:312-367` | `Training.jl:370-375` | No |
| Top-1/2/3 agreement % | `Training.jl:318-325` | `Training.jl:380-384` | No |
| Root confidence mean/med/min/max/P25/P50/P75/P95 | `Training.jl:386-401` | `Training.jl:393-401` | No |
| Policy L1 distance mean/med/P25/P75/P95 | `Training.jl:328-329` | `Training.jl:407-410` | No |
| Target policy entropy mean/med/min/max/P25/P75/P95 | `Training.jl:332-333` | `Training.jl:417-423` | No |
| Network drift KL | `train.jl:718-748` | `train.jl:750-752` | No |
| Positions generated, Samples consumed | `Training.jl:351-353` | `Training.jl:351-353` | No |
| Avg gap between promotions | `Metrics.jl:124-125` | `Metrics.jl:124-125` | Partially (from promo history) |

### Current Replay Coverage Formula

From `Training.jl:339-342`:

```julia
replay_capacity = replay_buffer.capacity
replay_fill = length(replay_buffer)
replay_pct = round(replay_fill / replay_capacity * 100, digits=1)
```

This is **buffer fill percentage** (`length(buffer) / capacity * 100`), not state-space coverage. It tells you how full the circular buffer is. On iteration 1 with 200 positions generated and a 50,000-capacity buffer, it's `0.4%`. It grows monotonically until reaching 100% at capacity.

The name "Replay coverage" is misleading — the user has already decided to rename to **"Replay Fill %"**.

### Missing Promotion Record Fields

**Current `PromotionRecord`** (`Metrics.jl:49-57`):
```julia
struct PromotionRecord
    iteration::Int
    win_rate_vs_best::Float64
    wins::Int
    losses::Int
    draws::Int
    random_anchor_wr::Union{Nothing, Float64}
    promotion_score::Float64
end
```

**Not present but would be useful for observability:**
- `total_promotions_at_event::Int` — how many promotions had happened at this point (was in the original design spec)
- `gap_since_last::Int` — iterations since previous promotion (was in the original design spec)
- `elo_candidate::Float64` — candidate Elo at promotion time (was in original design)
- `elo_best::Float64` — best Elo at promotion time (was in original design)
- `timestamp::String` — wall-clock timestamp of the event
- `cumulative_positions::Int` — total positions generated so far
- `anchor_detailed_results::Dict` — W/L/D for each anchor agent

The original design proposal had a different shape with event context (gap, total count, Elos). The implementation replaced those with raw anchor data (random_anchor_wr, raw W/L/D, promotion_score). Both have value — the raw data allows recomputation, but the event context fields are missing and would need to be recomputed from history.

### Forward Pass Reuse Opportunities

**Already-computed values that can be reused without new forward passes:**

| Value | Already computed in | Reuse status |
|-------|-------------------|--------------|
| KL(target || network) | `Training.jl:312-314` (accumulated per position in loop) | Already computed but not persisted |
| Top-1/2/3 agreement | `Training.jl:318-325` | Already computed but not persisted |
| Policy L1 distance | `Training.jl:328-329` | Already computed but not persisted |
| Target entropy per position | `Training.jl:332-333` | Already computed but not persisted |
| Root confidence per position | `Training.jl:128` (from `search_with_stats` 3rd return) | Already computed but not persisted |
| Network drift KL | `train.jl:748` | Already computed but not persisted |
| Param update norm | `train.jl:737` | Already in CSV |
| Grad norm | `Training.jl:196` → `step_result.grad_norm` | Already in CSV |

**Values requiring new or modified computation:**

| Value | Where it would come from | New work required |
|-------|--------------------------|-------------------|
| MCTS searched root value (Q) | `root.value_sum[] / root.visits[]` in MCTS tree | Small: modify `search_with_stats` return to include root Q |
| Iteration wall-clock duration | `@elapsed` or `time_ns()` around iteration body | Trivial: already available in train.jl |
| Model size / parameter count | `Flux.destructure(model[])` -> length | Trivial: one-time |
| Learning rate at iteration | From optimizer state | Small: depends on optimizer struct shape |
| Policy entropy of searched MCTS policy | From `counts` returned by `search_with_stats` | Already in `pi_target` — computed but not summarized |
| Value prediction stats (mean, std) | From `predict_raw` outputs | One forward pass dimension — depends on batch size |

### Injection Points

The new JSONL persistence and enhanced metrics would hook into `train.jl` at these locations:

**Setup (before main loop, around current line 695-706):**
- Open JSONL metrics file (separate from CSV)
- Write JSONL header/metadata entry (config snapshot, architecture, commit SHA)

**Per-iteration (inside the `for iter` loop, around lines 714-849):**
1. **Before `run_training_iteration`** (before line 721): Record iteration start timestamp, capture `θ_old`
2. **After `run_training_iteration`** (after line 736): Capture `TrainingResult` fields + MCTS diagnostics accumulators (but MCTS diagnostics are currently NOT returned from `run_training_iteration` — they're only printed)
3. **After drift computation** (after line 752): Capture drift KL
4. **After baseline eval** (after line 757): Capture baseline WR
5. **After promotion eval** (after line 775): Capture promotion evaluation results
6. **After promotion decision** (after line 817): Capture promotion record
7. **Write JSONL entry** (alongside CSV write at line 843): Flush structured JSONL entry with ALL accumulated fields

**Critical observation**: MCTS diagnostics (KL, Top-K, L1, Entropy) are computed inside `run_training_iteration` but **never returned** in the `TrainingResult` struct. To persist them, we need to either:
- Option A: Extend `TrainingResult` to include these fields (changes `Training.jl`)
- Option B: Add a separate accumulation/callback mechanism
- Option C: Return a richer struct from `run_training_iteration`

### Approaches

#### 1. JSONL Persistence Format

**Approach: Dedicated JSONL file, one JSON object per iteration**

```jsonl
{"iter": 1, "timestamp": "2026-07-29T13:00:00Z", "duration_s": 45.2, "avg_loss": 4.21, "policy_loss": 2.10, "value_loss": 2.11, "grad_norm": 3.45, "pred_entropy": 0.89, "target_entropy": 0.72, "param_update_norm": 0.52, "replay_fill_pct": 0.4, "avg_game_len": 34.2, "baseline_wr": 95.0, "candidate_vs_best_wr": null, "promoted": false, "elo_candidate": 1500.0, "elo_best": 1500.0, "mcts_kl_mean": 0.45, "mcts_kl_median": 0.38, "mcts_top1_pct": 42.5, "mcts_top2_pct": 68.1, "mcts_top3_pct": 82.3, "mcts_root_conf_mean": 0.32, "mcts_l1_mean": 0.41, "drift_kl": 0.12}
```

Pros:
- Append-only, crash-safe without atomic rename (one line per iteration is already atomic at the OS page level when flushed)
- Schema-less — fields can be added/removed without migration
- Parseable by any JSONL tool (pandas, jq, custom scripts)
- Each iteration is self-contained (no header row)

Cons:
- Larger than CSV (human-readable KB per row vs CSV's ~200 bytes)
- Requires JSON serialization dependency or Julia-native serializer
- Not directly viewable as a table without conversion

**Approach details**: Use Julia's `JSON3.jl` or stdlib `Serialization` with a thin JSONL wrapper. File path: `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl`. Opened once in append mode, flushed per iteration. A separate metadata file (`metrics_meta.json`) records the schema version and field descriptions.

#### 2. MCTS Diagnostics Return from run_training_iteration

**Option A: Extend `TrainingResult`**

Add fields for MCTS diagnostics aggregates to `TrainingResult`. This is the simplest change but couples `Training.jl` to the full set of aggregate stats.

```julia
struct TrainingResult
    avg_loss::Float32
    avg_policy_loss::Float32
    avg_value_loss::Float32
    avg_grad_norm::Float32
    avg_pred_entropy::Float32
    avg_target_entropy::Float32
    avg_game_len::Float64
    replay_pct::Float64
    # NEW:
    kl_mean::Float32
    kl_median::Float32
    top1_pct::Float32
    top2_pct::Float32
    top3_pct::Float32
    root_conf_mean::Float32
    l1_mean::Float32
end
```

Pros: Simple, typed, no indirection. Cons: Struct grows with each new metric, every caller must handle all fields.

**Option B: Add a diagnostics bag (Dict or NamedTuple)**

Return a second value from `run_training_iteration` — an untyped bag of diagnostic accumulators.

```julia
function run_training_iteration(...)
    # ... existing computation ...
    # ... existing return ...
    return TrainingResult(...), mcts_diagnostics
end
```

Pros: No struct changes for existing callers, extensible. Cons: Changes call signature, caller must destructure.

**Recommendation**: Option A (extend `TrainingResult`) — the struct is internal to the training module, only used as a return type. Adding fields is backward compatible (old code accessing `.avg_loss` still works). The user's requirement says "no changes to Training.jl behavior" but adding read-only return fields doesn't change behavior.

#### 3. Root Q-value from MCTS

The `search_with_stats` function currently returns `(best_action, counts, max_confidence)`. To add the searched root value (Q), we'd modify the return to include `root.value_sum[] / root.visits[]` after the search loop (after line 175 in MCTS.jl). This requires a change to `MCTS.jl` — the current constraint says no changes to `MCTS.jl`. However, the root Q is a **read-only observation** that doesn't change algorithm behavior, so this constraint should be re-examined for this specific change.

### Recommendation

1. **Persistence format**: JSONL — the schema flexibility, crash-safety, and tooling compatibility outweigh the file size cost. Keep CSV for backward compatibility; add JSONL alongside it.

2. **MCTS diagnostics return**: Extend `TrainingResult` with aggregate MCTS diagnostic fields. This is the minimal-change approach and preserves type safety.

3. **Root Q from MCTS**: Modify `search_with_stats` to return root Q-value. The "no changes to MCTS.jl" constraint from the original spec was context-specific (the training-progress change) and should not block a read-only observation that costs nothing.

4. **Promotion record enrichment**: Add `gap_since_last`, `total_promotions_at_event`, and Elo-at-promotion to `PromotionRecord`. These are metadata fields that can be derived at promotion time and stored for reference.

5. **Separation of concerns**: Keep JSONL writing in `train.jl` (caller owns the IO), not in `Metrics.jl` or `Training.jl`. The modules produce values; the script persists them.

### Risks

- **JSONL performance**: One JSON object per iteration (~1KB) at 50-byte line overhead is negligible. Risk is low.
- **MCTS.jl change**: The "no changes to MCTS.jl" constraint needs re-examination. A read-only return change is trivial but violates the letter of the constraint.
- **TrainingResult extension**: Adding fields to a struct that is also used for the `isfinite` guard (`Base.isfinite(r::TrainingResult)`) is safe as long as the first field (avg_loss) stays.
- **CSV vs JSONL dual-write**: Dual persistence means two file descriptors open. Training iterations are I/O bound at batch time, not CSV write time — risk is low.

### Ready for Proposal

Yes
