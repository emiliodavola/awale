# Delta for Consolidation

Consolidation changes across `training-observability` and `training-metrics` — bug fixes, docs restructure, value calibration accumulation, Search Gain, persistence audit, Health Dashboard multi-signal, interpretation guide, and metric classification. Zero algorithm, MCTS, or training behavior changes.

## ADDED Requirements

### Domain: training-observability

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| CR-O1 | **Search Gain**: `search_with_stats` MUST return a 5th value: `raw_network_value` (root network value from first MCTS expansion before backup). TrainingResult MUST carry `root_q_mean` and `search_gain`. Search Gain = MCTS_root_Q − raw_network_value. JSONL MUST persist both `root_q_mean` and `search_gain` | MUST | Search batch completes → `search_gain` stored in TrainingResult, written to JSONL. Zero extra forward passes — reuses root.value from first MCTS node expansion |
| CR-O2 | **Interpretation guide**: `docs/metrics.md` MUST include a `## Metric Guide` section with 4+ metric combination patterns: converging network (KL↓, Top1↗, loss↓), active search (Top1 moderate, KL moderate, Drift > ε_kld), rapid changes (Δ-metrics ↑, grad_norm ↑), stalled (all variances → 0 or model unchanging) | MUST | Reader opens docs → finds all 4 patterns with their interpretations and what to do in each case |
| CR-O3 | **Metrics classification**: All metrics in `docs/metrics.md` MUST be explicitly grouped into one of 7 categories: Optimization (loss, grad_norm, param_update), Policy Learning (KL, Top1, entropy, policy distance), Search (root Q, search gain, root confidence), Value (value loss, value calibration, v_pred stats), Replay (replay fill %, game length), Promotion (win rates, Elo, promotion gap), Network Evolution (drift, param update norm) | MUST | Reader inspects any metric section → classification label visible. Total of 7 groups, each metric appears under exactly one group |

### Domain: training-metrics

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| CR-M1 | **Value calibration accumulation**: Value calibration MUST `append!` `v_pred` and `v_target` across ALL batches in the iteration (not only the last batch). MAE, Pearson r, Spearman ρ MUST be computed on the concatenated vectors. MUST reuse existing forward pass — zero extra inference | MUST | 10 batches of 1000 samples → calibration computed on all 10000 concatenated samples, not last 1000 only. No forward pass beyond normal training |

## MODIFIED Requirements

### Domain: training-observability

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| O1 | **JSONL persistence** (root Q fix + full persistence): One JSON object per iteration at `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl`. Append-mode, flush per iteration, no header row. `mcts_root_q` MUST store actual MCTS root Q (`root.value_sum / root.visits`), not the mean network prediction. JSONL MUST also include: `root_q_mean`, `search_gain`, `kl_p25`, `kl_p50`, `kl_p75`, `l1_p25`, `l1_p50`, `l1_p75`, `entropy_mean`, `entropy_p25`, `entropy_p50`, `entropy_p75`, `root_conf_mean`, `root_conf_p25`, `root_conf_p50`, `root_conf_p75`, `top_3`, `health_state`, `stability_string`, `warnings`. (Previously: mcts_root_q stored mean network prediction; distributional stats and health states were console-only) | MUST | Iter completes → mcts_root_q = root.value_sum/root.visits. All printed distributional stats and dashboard states appear as JSONL fields |
| O6 | **Convergence detection** (stability fix): Passive indicators only — sliding window (default 20 iter) checks: KL stability (variance < εₖₗ), Drift stability (variance < ε_d), Top1 stability (variance < ε_t1), param update magnitude (< ε_p using actual param update norm `\|\|θ_new−θ_old\|\|₂`, not policy loss variance). Prints ACTIVE/STALLED per metric. (Previously: used ma_buf_policy_loss variance for `Param:` convergence check) | SHOULD | KL variance < εₖₗ for 20 iters → `KL: STALLED`. `Param:` check uses param_update_norm sliding window, not ma_buf_policy_loss variance |
| O7 | **Health dashboard** (multi-signal + full states): Compact single printed line per iteration: Network learning (ACTIVE/STALLED/BOOTSTRAP), Search usefulness (HIGH/LOW using Top1 AND KL AND Policy Distance thresholds simultaneously), Model drift (LOW/MEDIUM/HIGH from drift KL), Promotion frequency, Replay fill %, Value calibration (OK/WARN/N/A). All states MUST include: ACTIVE, STALLED, HIGH, LOW, BOOTSTRAP, OK, N/A. BOOTSTRAP triggers when replay buffer < minimum fill threshold. ValCal state MUST be documented. Every threshold MUST include rationale in `docs/metrics.md`. (Previously: single-signal Srch:HIGH from Top1 alone; ValCal state undocumented; thresholds without rationale) | SHOULD | Top1 > 60% AND KL > ε_kl AND policy distance > ε_pd → Srch:HIGH. Replay fill < start threshold → Net:BOOTSTRAP, all other states N/A |
| O9 | **Cost documentation** (doc schema): Every metric MUST document: Definition (mathematical), Formula (LaTeX), Variables (inputs/sources), Implementation (code path), Computational Cost (O expression + measured μs + bytes/iter), Interpretation (what high/low means), Typical observations (empirical ranges from project runs), Notes (edge cases, caveats). Theory MUST be clearly separated from empirical observations. Log base MUST be explicitly ln (natural log). Top-K formula MUST read: "MCTS's top-1 action is in network's top-K". (Previously: schema was time/memory/frequency/reuse only; Top-K described as set-intersection) | MUST | Each metric entry has all 8 schema fields. Top-K formula matches implementation: set membership not intersection. Log base spelled out as natural log |
