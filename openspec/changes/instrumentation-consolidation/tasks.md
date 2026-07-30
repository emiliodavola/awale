# Tasks: Instrumentation Consolidation

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~450 |
| 400-line budget risk | Medium |
| Chained PRs recommended | Yes |
| Suggested split | PR 1: Code (T1–T9, T12) ~150 lines → PR 2: Docs (T10–T11) ~350 lines |
| Delivery strategy | auto-forecast |
| Chain strategy | feature-branch-chain |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: Medium

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Code fixes + Search Gain + calibration + persistence + tests | PR 1 | Base = `feat/instrumentation-consolidation`. ~150 lines, self-contained |
| 2 | Docs restructure (schema, classification, Health Dashboard) | PR 2 | Base = PR 1 branch. ~350 lines additive, requires code to be merged first for formula verification |

## Dependency Graph

```
T1 (MCTS 5th return) → T2 (TrainingResult fields) → T3 (Search Gain train.jl)
                                                      → T4 (JSONL mcts_root_q fix)
                                                      → T8 (JSONL persistence audit)
T5 (root Q propagation) → T4
T6 (param_window fix) ── independent ──→ T12
T7 (calibration append!) ── independent ──→ T12
T9 (multi-signal dashboard) → T12
T10 (docs restructure) ── independent ──→ T12
T11 (dashboard docs) → T10
T12 (test updates) depends on T1–T9
```

## Per-Task Breakdown

### Phase 1: Foundation (MCTS + TrainingResult)

- [x] **T1** — `src/Awale/MCTS.jl`: Capture `root_value` at line 141 (`logits, root_value = ...`). Return 5 values in `search_with_stats` (all 4 return sites). Update `search` wrapper destructuring to `_, _, _, _, _`. **~3 lines | Deps: none | CR-O1**
- [x] **T2** — `src/Awale/Training.jl`: Add 22 fields to `TrainingResult` (root_q_mean, network_value_mean, kl_p25/p50/p75/p95, l1_p25/p50/p75/p95, entropy_mean/min/max/p25/p50/p75/p95, root_conf_min/max/p25/p50/p75/p95). Add `all_raw_values::Float32[]` accumulator in `run_training_iteration`. Compute means/percentiles in the return block. Pass `all_raw_values` as 4th return from `collect_selfplay_data`. Update `collect_selfplay_data` destructuring to 5-value unpack. **~30 lines | Deps: T1 | CR-O1, O1**

### Phase 2: Bug Fixes & Calibration

- [x] **T3** — `train.jl`: Compute `search_gain = root_q_mean - network_value_mean` after iteration. Write `root_q_mean`, `network_value_mean`, `search_gain` to JSONL. **~3 lines | Deps: T2 | CR-O1**
- [x] **T4** — `train.jl`: Replace `mean(calib_data.v_pred)` with `training_result.root_q_mean`. Add `mcts_network_raw` JSONL field with `training_result.network_value_mean`. **~2 lines | Deps: T2, T5 | O1**
- [x] **T5** — `src/Awale/Training.jl`: `all_root_q` accumulated at line 273 (already `append!`) — ensured `root_q_mean` is computed from it and stored in `TrainingResult`. Added sibling computation for `all_root_q` and `all_raw_values`. **~3 lines | Deps: T1 | O1, CR-O1**
- [x] **T6** — `train.jl`: Initialize `ma_buf_param_update = Float32[]` (line 739 sibling). Push `param_update_norm` into it. Replace `ma_buf_policy_loss` with `ma_buf_param_update` in convergence check with adjusted threshold (1e-8 variance). **~5 lines | Deps: none | O6**
- [x] **T7** — `src/Awale/Training.jl`: Change `last_v_pred = step_result.after_values` and `last_v_target = step_result.v_target` from overwrite to `append!(last_v_pred, vec(...))` / `append!(last_v_target, vec(...))`. **~2 lines | Deps: none | CR-M1**

### Phase 3: Integration (train.jl wiring)

- [x] **T8** — `train.jl` JSONL dict assembly: Add 27 new fields: `search_gain`, `root_q_mean`, `network_value_mean`, `mcts_network_raw`, `kl_p25/p50/p75/p95`, `l1_p25/p50/p75/p95`, `entropy_mean/min/max/p25/p50/p75/p95`, `root_conf_min/max/p25/p50/p75/p95`, `net_health`, `srch_health`, `drift_health`, `valcal_health`, `stability_kl`, `stability_drift`, `stability_top1`, `stability_param`, `warning_count`, `warning_messages`. Refactored warnings to capture in array. **~40 lines | Deps: T2, T7 | O1**
- [x] **T9** — `train.jl`: Upgrade `srch_health` from single-signal (`top1_pct < 80%`) to multi-signal: `top1_pct <= 60% && kl_mean <= 0.15 && l1_mean <= 0.20 ? "LOW" : "HIGH"`. Added inline threshold constants with rationale comments. **~6 lines | Deps: T2 | O7**

### Phase 4: Documentation

- [x] **T10** — `docs/metrics.md`: Full restructure. Every metric gets 8-field schema (Definition/Formula/Variables/Implementation/Cost/Interpretation/Observations/Notes). Add top-level classification (7 categories). Add `## Metric Guide` with 4 combination patterns (CR-O2). Fix Top-K formula to set membership. Specify log base (ln). Separate theory from empirical. Add Search Gain section. **~350 lines | Deps: none | O9, CR-O2, CR-O3**
- [x] **T11** — `docs/metrics.md` §Health Dashboard: Document all states — Net:ACTIVE/STALLED/BOOTSTRAP, Srch:HIGH/LOW, Drift:LOW/MEDIUM/HIGH, ValCal:OK/HIGH/N/A, stability:ACTIVE/STALLED/BOOTSTRAP. Every threshold with rationale. Document BOOTSTRAP (replay < 10%). **~30 lines integrated into T10 | Deps: T10 | O7, O6**

### Phase 5: Testing

- [x] **T12** — Updated 3 test files: `test/test_jsonl_observability.jl` (new fields + Search Gain + JSONL serialization tests); `test/test_metrics_enrichment.jl` (no change needed); `test/test_training_pipeline.jl` (extended TrainingResult constructor in no-data path test with 22 new fields; extended `search_with_stats` destructuring in tests to 5-value unpack; added assertions for new TrainingResult fields). All 612 tests pass (432 existing + new tests). **~60 lines | Deps: T1–T9 | All**

## Requirement Mapping

| Spec Req | Tasks | Description |
|----------|-------|-------------|
| O1 | T4, T5, T8, T2 | JSONL root Q fix + full persistence (25 new fields) |
| O6 | T6 | Param stability check fix (ma_buf_param_update) |
| O7 | T9, T11 | Multi-signal dashboard + documented states |
| O9 | T10 | Doc schema (8-field per metric, Top-K fix, log base) |
| CR-O1 | T1, T2, T3 | Search Gain (5th return + TrainingResult + JSONL) |
| CR-O2 | T10 | Interpretation guide (4 patterns in docs) |
| CR-O3 | T10 | Metrics classification (7 categories in docs) |
| CR-M1 | T7 | Value calibration accumulation (append! full iter) |
| Verification | T12 | All tests pass, new fields validated |
