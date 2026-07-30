# Design: Instrumentation Consolidation

## Technical Approach

Eight-area consolidation: bug fixes (3 targeted ≤5-line edits), docs restructure, health dashboard documentation, value calibration accumulation, Search Gain, persistence audit, interpretation guide, and metric classification. Zero algorithm, MCTS, or behavioral changes. All changes are data-pipeline fixes and doc improvements.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| **Distributional data format** | Flat JSONL fields (`kl_p25`, `kl_p50`, `kl_p75`) | JSON arrays `"kl_percentiles": [...]` | Flat fields are queryable in any JSON parser; arrays require position-indexing. Post-hoc aggregation on flat fields is trivial. Cost: 12 extra scalar fields per iteration (~96 bytes) |
| **Search Gain computation** | Compute in `train.jl` from two TrainingResult means | Compute in `run_training_iteration` or MCTS | `search_gain = root_q_mean - network_value_mean` is a scalar difference. Keeping it in `train.jl` avoids adding a redundant field and keeps TrainingResult as raw-observation struct |
| **Multi-signal Health Dashboard** | Top1 ≤60% AND KL ≤ ε_kl AND policy distance ≤ ε_pd → Srch:LOW; else HIGH | Single-signal (current: Top1 < 80% → HIGH) | Three conjunctions prevent false HIGH when only one signal is strong. Thresholds: Top1 ≤60%, KL ≤0.15, L1 ≤0.20. Rationale documented in metrics.md |
| **5th return propagation** | Update `search` wrapper + `collect_selfplay_data` destructuring to 5-value unpack | Separate API call for raw value | Zero extra inference cost (value already on line 141). Compiler catches incomplete destructuring |
| **raw_network_value origin** | Capture `predict_inference` 2nd return at MCTS.jl:141 | `root.value` after first eval | Line 141 `logits, _ = predict_inference(...)` already computes the value and discards it — just name it |
| **Param stability buffer** | New `ma_buf_param_update` ring buffer in `train.jl` | Reuse `ma_buf_policy_loss` | Current bug: BUG-C uses policy loss variance as proxy for param update magnitude. Dedicated buffer holds actual `param_update_norm` values |

## Data Flow

```
Self-play ──→ search_with_stats ──→ collect_selfplay_data ──→ run_training_iteration
                  │ (5 values)      │ (5 values: +raw_val)     │
                  ▼                  ▼                          ▼
          root_q + raw_net_val  all_root_q + all_raw_vals   TrainingResult
                                                               │ root_q_mean
                                                               │ network_value_mean
                                                               │ +15 distributional fields
                                                               ▼
                                                          train.jl
                                                          ├─ search_gain = root_q_mean - network_value_mean
                                                          ├─ calib_data via append! (full iteration)
                                                          ├─ ma_buf_param_update ← param_update_norm
                                                          └─ JSONL: 25 new fields
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/Awale/MCTS.jl` | Modify | 5th return from `search_with_stats`: capture `root_value` at line 141. Return `(action, counts, max_conf, root_q, root_value)`. Update `search` wrapper (line 94) and early-return (line 163) |
| `src/Awale/Training.jl` | Modify | Add 17 fields to `TrainingResult` (root_q_mean, network_value_mean, 15 distributional stats). Change line 323-324 from `=` to `append!`. Add `all_raw_values` accumulator (line 298 sibling). Compute means/percentiles in return path |
| `train.jl` | Modify | BUG-A: line 1002 replace `mean(calib_data.v_pred)` with `training_result.root_q_mean`. BUG-C: line 943 replace `ma_buf_policy_loss` with new `ma_buf_param_update`. Add `ma_buf_param_update` init + push (line 911 sibling). JSONL: add 25 new fields (search_gain, health_state, stability_string, warnings string, all distributional fields). Multi-signal Srch health (line 955) |
| `docs/metrics.md` | Rewrite | 8-section schema per metric (Definition/Formula/Variables/Implementation/Cost/Interpretation/Typical/Notes). Theory vs Empirical separation. Log base: ln. Top-K fix: set membership. New sections: Health Dashboard (all states + thresholds + rationale), Metric Guide (4 patterns), Classification (7 categories), Search Gain |
| `openspec/specs/training-observability/spec.md` | Delta | CR-O1, CR-O2, CR-O3 already present |
| `test/test_training_pipeline.jl` | Modify | Update TrainingResult constructor calls (lines 1051-1052). Update search_with_stats destructuring test calls (lines 313-334) — Julia handles extra returns silently, no breakage |

## Interfaces

### TrainingResult (extended — 17 new fields, 32 total)

```julia
struct TrainingResult
    # Existing 15 fields unchanged
    avg_loss::Float32; avg_policy_loss::Float32; avg_value_loss::Float32
    avg_grad_norm::Float32; avg_pred_entropy::Float32; avg_target_entropy::Float32
    avg_game_len::Float64; replay_pct::Float64
    kl_mean::Float32; kl_median::Float32; top1_pct::Float32
    top2_pct::Float32; top3_pct::Float32; root_conf_mean::Float32; l1_mean::Float32
    # NEW — BUG-B fix: root Q mean
    root_q_mean::Float32
    # NEW — Search Gain support
    network_value_mean::Float32
    # NEW — Distributional (persistence audit)
    kl_p25::Float32; kl_p50::Float32; kl_p75::Float32
    l1_p25::Float32; l1_p50::Float32; l1_p75::Float32
    entropy_mean::Float32; entropy_p25::Float32; entropy_p50::Float32; entropy_p75::Float32
    root_conf_p25::Float32; root_conf_p50::Float32; root_conf_p75::Float32
end
```

### MCTS.search_with_stats (5th return)

```julia
# MCTS.jl:141 — capture value
logits, root_value = predict_inference(mcts.model, root.state)
# root_value is Float32 scalar from Model.jl:487 (value[1])

# Line 138 early return (num_sims ≤ 0 or no actions)
return 0, zeros(Float32, 6), 0.0f0, 0.0f0, 0.0f0

# Line 163 num_sims ≤ 0 after expansion
return best_action, policy, maximum(policy), 0.0f0, 0.0f0

# Line 184-185 normal return
root_q = root.value_sum[] / max(1, root.visits[])
return best_action, counts, maximum(counts), root_q, root_value
```

### Search wrapper (MCTS.jl:93)

```julia
action, _, _, _, _ = search_with_stats(...)
```

### collect_selfplay_data (Training.jl:136)

```julia
_, pi_target, root_conf, root_q, raw_value = search_with_stats(...)
push!(all_raw_values, raw_value)
# Return at line 148: add all_raw_values as 4th return
return backfill_value_targets(samples, reward(state)), root_confidences, all_root_q, all_raw_values
```

### Calibration accumulation (Training.jl:298-324)

```julia
# Lines 298-299 — unchanged (empty init)
last_v_pred = Float32[]
last_v_target = Float32[]

# Lines 323-324 — changed from = to append!
append!(last_v_pred, vec(step_result.after_values))
append!(last_v_target, vec(step_result.v_target))
```

### JSONL entry additions (train.jl, ~line 1037)

```julia
# After existing dict entries, append:
"root_q_mean" => Float64(round(training_result.root_q_mean, digits=6)),
"network_value_mean" => Float64(round(training_result.network_value_mean, digits=6)),
"search_gain" => Float64(round(root_q_mean - network_value_mean, digits=6)),
# Distributional
"kl_p25" => Float64(round(training_result.kl_p25, digits=6)),
"kl_p50" => Float64(round(training_result.kl_p50, digits=6)),
"kl_p75" => Float64(round(training_result.kl_p75, digits=6)),
"l1_p25" => Float64(round(training_result.l1_p25, digits=6)),
"l1_p50" => Float64(round(training_result.l1_p50, digits=6)),
"l1_p75" => Float64(round(training_result.l1_p75, digits=6)),
"entropy_mean" => Float64(round(training_result.entropy_mean, digits=6)),
"entropy_p25" => Float64(round(training_result.entropy_p25, digits=6)),
"entropy_p50" => Float64(round(training_result.entropy_p50, digits=6)),
"entropy_p75" => Float64(round(training_result.entropy_p75, digits=6)),
"root_conf_p25" => Float64(round(training_result.root_conf_p25, digits=6)),
"root_conf_p50" => Float64(round(training_result.root_conf_p50, digits=6)),
"root_conf_p75" => Float64(round(training_result.root_conf_p75, digits=6)),
# Dashboard states
"health_state" => health_line,
"stability_string" => stability_str,
"warnings" => warnings_str,
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit — MCTS | `search_with_stats` returns 5 values, last is Float32 ∈ [-1,1] for num_sims>0 | Extend existing `@testset "policy-only..."` |
| Unit — TrainingResult | New 17 fields default to zero on no-data path | Extend existing `@testset "no-data path fields are zero"` |
| Unit — Training | `root_q_mean` matches mean of collected root_q across games | Integration: 1 game, 1 sim, assert `result.root_q_mean ≈ mean(all_q)` |
| Unit — Training | Calibration accumulates full iteration (append! not =) | Mock 3 batches: assert calibration computed on all 3 not just last |
| Unit — Training | `network_value_mean` populated from raw_network_value returns | Extend search_with_stats test |
| Unit — train.jl | `search_gain = root_q_mean - network_value_mean` | Assert in JSONL output |
| Integration — JSONL | New fields appear with correct values in output | Run 1 iter, read JSONL, verify all 25 new fields present |
| Existing tests | All 432 existing tests pass unchanged | `Pkg.test()` |

## Health Dashboard Multi-Signal States

| State | Signal | Threshold | Rationale |
|-------|--------|-----------|-----------|
| **Srch:HIGH** | Top1 ≤60% AND KL ≥0.15 AND L1 ≥0.20 | All three below threshold? LOW; else HIGH | Single-signal (current Top1<80%) triggers false HIGH. Triple conjunction ensures search is genuinely providing new information across policy divergence, ranking, and distribution mass |
| **Net:BOOTSTRAP** | Replay fill < start_threshold | replay_fill_pct < 10% | Buffer too small for representative sampling — skip learning state assessment |
| **ValCal:WARN** | MAE > 0.5 OR Pearson < 0.3 | Either threshold breached | Value head poorly calibrated |
| **ValCal:OK** | MAE ≤0.5 AND Pearson ≥0.3 | Both thresholds satisfied | Healthy value calibration |

## Open Questions

- [ ] **None** — all decisions resolved in design phase.

## Migration / Rollout

No migration required. CSV unchanged. JSONL adds fields — post-hoc analysis scripts that deserialize into typed structs need field additions, but raw JSON consumers handle extra fields transparently.
