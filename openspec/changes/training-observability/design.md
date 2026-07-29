# Design: Training Observability

## Technical Approach

Four-layered additive instrumentation: (1) extend `TrainingResult` to carry MCTS aggregates that are currently printed and discarded, (2) add a read-only root Q return to `search_with_stats`, (3) enrich `PromotionRecord` and Elo mechanics in `Metrics.jl`, (4) inject a JSONL writer, convergence detection, health dashboard, and warnings into `train.jl` — all reusing existing forward passes. CSV unchanged.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| **TrainingResult extension** | Add 7 MCTS aggregate fields | Separate diagnostics bag | Same pattern as existing fields; backward-compatible (old `.avg_loss` access unchanged). Internal struct, single internal caller |
| **JSONL library** | JSON3.jl | JSON.jl, Serialization | JSON3.jl produces compact output and handles Julia types natively. User accepted implicitly |
| **Root Q return** | 4th return from `search_with_stats` | Separate query call | Zero-cost computation (`value_sum / visits` already loaded in registers). Read-only — no algorithm path change. Update `search` wrapper and `collect_selfplay_data` destructuring too |
| **PromotionRecord enrichment** | 5 new fields directly on struct | Separate enrichment table | Self-contained records — no join needed. TOML serialization fields can be `nothing`-guarded for resume compat |
| **Elo K default** | 64 | Keep 32 | Matches typical AlphaZero literature values for faster convergence signal |
| **Convergence detection** | Passive print in `train.jl` main loop | New module | Print-only per O6 constraint. Moving window (20 iters) stored in a mutable ring buffer. No behavioral feedback |
| **Δ-metrics / moving averages** | Computed in `train.jl`, JSONL-only | In `Metrics.jl` | Avoid serializing previous iteration state into module. `train.jl` holds prior values in local vars |

## Data Flow

```
Self-play ──→ run_training_iteration ──→ TrainingResult (incl. MCTS aggregates)
                    │                              │
                    │    train.jl main loop         │
                    ▼                              ▼
          ┌──────────────────────────────────────────┐
          │  Accumulate: drift KL, Elo, value cal,  │
          │  Δ-metrics, moving averages, convergence │
          └──────────┬───────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     CSV (same)   JSONL writer  Print diagnostics
     append-row   JSON3.write   convergence/health
     per-iter     flush+append   warnings
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/Awale/Training.jl` | Modify | Add 7 MCTS aggregate fields to `TrainingResult`. Populate them in `run_training_iteration` return paths (line 429, line 435) |
| `src/Awale/MCTS.jl` | Modify | `search_with_stats` returns 4-tuple `(action, counts, max_conf, root_q)`. `root_q = root.value_sum[] / root.visits[]`. Update `search` wrapper and all destructuring callers |
| `src/Awale/Metrics.jl` | Modify | Add 5 fields to `PromotionRecord`. Change Elo K default to 64. Add `compute_value_calibration()`. Add `print_historical_summary()`. Rename "Replay Coverage" → "Replay Fill %" in all print strings |
| `train.jl` | Modify | Open JSONL file handle at setup. Per-iteration: compute value calibration, Δ-metrics, moving averages, convergence checks, health line, warnings. JSONL entry with ALL fields. Dual-write alongside CSV |
| `src/Awale/Utils.jl` | None | No changes needed |
| `docs/metrics.md` | Create | Formula + interpretation + cost per metric (per O9) |

## Interfaces

### TrainingResult (extended)

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
    # NEW — MCTS diagnostic aggregates
    kl_mean::Float32
    kl_median::Float32
    top1_pct::Float32
    top2_pct::Float32
    top3_pct::Float32
    root_conf_mean::Float32
    l1_mean::Float32
end
```

### MCTS.search_with_stats (extended return)

```julia
# Before: return best_action, counts, maximum(counts)
# After:
root_q = root.value_sum[] / max(1, root.visits[])
return best_action, counts, maximum(counts), root_q
```

All three call sites (`search`, `collect_selfplay_data`, num_sims ≤ 0 early return) updated to destructure 4 values.

### PromotionRecord (enriched)

```julia
struct PromotionRecord
    iteration::Int
    win_rate_vs_best::Float64
    wins::Int
    losses::Int
    draws::Int
    random_anchor_wr::Union{Nothing, Float64}
    promotion_score::Float64
    # NEW:
    gap_since_last::Int
    total_promotions_at_event::Int
    elo_candidate::Float64
    elo_best::Float64
    timestamp::String
end
```

### JSONL entry shape (per iteration)

```
{"metric_version":"1.0.0","training_version":"...","git_commit":"abc123",
 "architecture":"...","config_hash":"e3b0c442",
 "iter":1,"timestamp":"2026-07-29T13:00:00Z","duration_s":45.2,
 "avg_loss":4.21,"policy_loss":2.10,"value_loss":2.11,
 "grad_norm":3.45,"pred_entropy":0.89,"target_entropy":0.72,
 "param_update_norm":0.52,"replay_fill_pct":0.4,"avg_game_len":34.2,
 "baseline_wr":95.0,"candidate_vs_best_wr":null,
 "promoted":false,"elo_candidate":1500.0,"elo_best":1500.0,
 "mcts_kl_mean":0.45,"mcts_kl_median":0.38,
 "mcts_top1_pct":42.5,"mcts_top2_pct":68.1,"mcts_top3_pct":82.3,
 "mcts_root_conf_mean":0.32,"mcts_l1_mean":0.41,
 "drift_kl":0.12,"mcts_root_q":0.35,
 "value_cal_mae":0.28,"value_cal_pearson":0.62,"value_cal_spearman":0.58,
 "elo_expected":0.46,"elo_actual":0.55,"elo_delta":5.76,"elo_upset":0.09,
 "delta_kl":0.03,"ma_loss_5":4.15,"ma_loss_10":null,"ma_loss_20":null}
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| **Unit — TrainingResult** | New fields default/set correctly, `isfinite` unchanged | Extend existing `TrainingResult` construction test. New fields zero on no-data path |
| **Unit — MCTS** | `search_with_stats` returns `root_q = root.value_sum / root.visits` | After 10 sims, Q ∈ [-1, 1] |
| **Unit — Metrics.jl** | Enriched `PromotionRecord` serialization roundtrip (TOML) | Construct record with all fields, serialize, deserialize, assert equality |
| **Unit — Metrics.jl** | Elo with K=64 vs K=32 produces larger Δ | 55% WR, same rating → Δ₆₄ > Δ₃₂ |
| **Unit — Metrics.jl** | Value calibration from known v_hat and returns | MAE=0, Pearson=1 on perfect predictions |
| **Unit — Metrics.jl** | Historical promo summary formatting | Mock ProgressTracker with N records, assert print output matches spec |
| **Integration — JSONL writer** | File created, appended, flush-after-write | Run 3 iterations, verify file has 3 valid JSON lines, no header |
| **Integration — Dual-write** | CSV unchanged + JSONL written alongside | Run 1 iteration, assert CSV row present + JSONL line present with same metric values |
| **Integration — Rename** | All print labels say "Replay Fill %" | Grep for "Replay Coverage" — zero matches in print paths |

## Open Questions

- [ ] Resumability of JSONL: on resume, should we append (current iteration adds line N+1) or re-create (start fresh)? Proposal says append-mode — consistent with CSV.
- [ ] `value_calibration` for empty batch (0 positions): should output NaN or skip? NaN aligns with existing `TrainingResult` zero-value fallback (line 435).
