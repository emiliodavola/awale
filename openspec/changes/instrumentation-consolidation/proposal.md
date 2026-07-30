# Proposal: instrumentation-consolidation

## Intent

Fix 3 critical data bugs in the training instrumentation merged in PR#60, restructure `docs/metrics.md` to separate mathematical definition from empirical observation, accumulate value calibration across the full iteration, implement Search Gain, and verify full persistence — all with zero algorithm, MCTS, or behavioral changes.

## Scope

### In Scope

| Phase | Deliverables |
|-------|-------------|
| **1 — Bug fixes** | Fix JSONL `mcts_root_q` (stores mean network pred, not MCTS Q). Fix discarded root Q in `collect_selfplay_data`. Fix stability `Param:` convergence check (uses policy loss variance instead of param update norm) |
| **2 — Docs restructure** | Every metric: Definition, Formula, Variables, Implementation, Cost, Interpretation, Typical observations, Notes. Separate theory (`$` facts) from observation (project-specific). Fix Top-K formula (set intersection → implementation). Specify log base (ln). Root Q: document vs network baseline. Health Dashboard: all states with calculation + thresholds + rationale |
| **3 — Health Dashboard** | Multi-signal states (e.g. Srch:HIGH based on Top1 AND KL AND L1). Fix ValCal HIGH undocumented. Document BOOTSTRAP state + magic thresholds |
| **4 — Value calibration** | Accumulate across full iteration (append! instead of overwrite). Zero extra forwards. Full MAE/Pearson/Spearman for the iteration |
| **5 — Search Gain** | 5th return from `search_with_stats`: raw network value at root. Delta = MCTS_root_Q - network_raw_value. Persist to JSONL. Document meaning + cost |
| **6 — Persistence audit** | Every console-only metric must reach JSONL: KL percentiles, L1 percentiles, entropy distribution stats, root confidence stats, Top-3. Warning drift threshold mismatch (mean < 1e-6 vs variance < 0.0001) documented |
| **7 — Interpretation guide** | Metric combinations (4 cases from original prompt). How to read a full training run from metrics alone |
| **8 — Classification** | Reorganize all metrics into: Optimization, Policy Learning, Search, Value, Replay, Promotion, Network Evolution |

### Out of Scope

No changes to algorithm, MCTS behavior, training loop control flow, optimizer, scheduler, replay buffer, hyperparameters, promotion criteria, percentiles beyond what exists, new policy metrics, Elo variants, new console blocks, console block reordering.

## Capabilities

### New
None — consolidation only, no new capability boundaries.

### Modified
- `training-metrics`: Delta for docs restructuring schema, value calibration accumulation, Search Gain metric, Health Dashboard state definitions, metric classification.
- `training-observability`: Delta for JSONL root_q fix, stability convergence fix, persistence audit (console-only → JSONL), Health Dashboard multi-signal states.

## Approach

| Area | Approach |
|------|----------|
| Bug fixes | 3 targeted edits: (1) replace network pred with `root_q` in JSONL assembly (train.jl), (2) add `root_q_mean` to `TrainingResult`, (3) replace `ma_buf_policy_loss` with actual param update norm sliding window. All ≤ 5 line changes |
| Docs restructure | Rewrite `docs/metrics.md` with new per-metric schema. New `## Metric Guide` section for combinations. Keep cost table. Move empirical ranges to `## Empirical Observations` appendix |
| Value calibration | In `train.jl` iteration loop: change `last_v_pred = step_result.after_values` to `append!`. After loop ends: compute calibration on full concatenated vectors |
| Search Gain | Add 5th return `raw_network_value` to `search_with_stats` (store `root.value` from first MCTS node expansion). Compute `search_gain = mcts_root_q - raw_network_value` in train.jl. Add `root_q_mean` + `search_gain` to `TrainingResult` and JSONL |
| Persistence audit | Add missing distributional fields to JSONL dict assembly. Verify every printed metric has a JSONL field. No structural changes |
| Health Dashboard | Modify each state from single-signal to multi-signal heuristics. Document each state's calculation + threshold rationale in docs |

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/Awale/Training.jl` | Modified | `TrainingResult` adds `root_q_mean`, `search_gain`; value calibration accumulation (append!) |
| `src/Awale/MCTS.jl` | Modified | `search_with_stats` 5th return (`raw_network_value`) |
| `src/Awale/Metrics.jl` | Modified | Stability `Param:` convergence check fix |
| `train.jl` | Modified | JSONL root_q fix, Search Gain, calibration accumulation, persistence audit fields |
| `docs/metrics.md` | Restructured | New schema, separation theory/observation, Health Dashboard docs, interpretation guide, classification |
| `openspec/specs/training-metrics/spec.md` | Delta updates | Value calibration, Search Gain, doc schema |
| `openspec/specs/training-observability/spec.md` | Delta updates | Root Q fix, convergence check fix, persistence audit, multi-signal dashboard |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Docs restructure accidentally changes metric formulas | Low | All formulas verified against implementation before edit. Code is source of truth |
| Value calibration append! changes memory profile | Low | Concatenating ~10K Float32s per iter → ~40 KB peak, negligible |
| Search Gain 5th return breaks callers | Low | Update `search` wrapper and all destructuring sites. Compiler catches mismatches |
| 450-line budget exceeded | Medium | 3 bug fixes (≈15 lines) + calibration (≈5 lines) + Search Gain (≈15 lines) + docs (~250 lines). Docs dominates. Consider chained PR if needed |

## Rollback Plan

Revert targeted commits for each bug fix. Docs changes are pure additive — revert `docs/metrics.md` to pre-consolidation version. All changes are data-only fixes and doc improvements; no behavioral rollback needed.

## Success Criteria

- [ ] JSONL `mcts_root_q` correctly stores MCTS root Q (not network prediction) — verified by comparing to raw `root.value_sum/root.visits`
- [ ] `root_q_mean` appears in `TrainingResult` and JSONL — no longer discarded
- [ ] Stability `Param:` convergence check uses actual param update norm
- [ ] Value calibration computed on full iteration data — not last batch only
- [ ] Search Gain documented and persisted — delta between MCTS root Q and raw network value
- [ ] Every console-only metric has matching JSONL field
- [ ] `docs/metrics.md` uses the new schema for every metric — definition (always valid) separated from observations (project-specific)
- [ ] Top-K formula in docs matches implementation (MCTS top-1 in network top-K)
- [ ] Health Dashboard documents all states with thresholds and rationale
- [ ] All 432 existing tests pass unchanged
