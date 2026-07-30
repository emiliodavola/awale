# Training Observability Specification

## Purpose

Capture every per-iteration metric that is currently printed then discarded (MCTS diagnostics, network drift) into persistent JSONL, enrich promotion records with event context, and provide passive convergence detection, health assessment, and diagnostic warnings — all with zero behavioral change to training, MCTS, or self-play.

## Requirements

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| O1 | **JSONL persistence** (root Q fix + full persistence): One JSON object per iteration at `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl`. Append-mode, flush per iteration, no header row. `mcts_root_q` MUST store actual MCTS root Q (`root.value_sum/root.visits`), not network prediction. JSONL MUST also include: `root_q_mean`, `search_gain`, `kl_p25`, `kl_p50`, `kl_p75`, `l1_p25`, `l1_p50`, `l1_p75`, `entropy_mean`, `entropy_p25`, `entropy_p50`, `entropy_p75`, `root_conf_mean`, `root_conf_p25`, `root_conf_p50`, `root_conf_p75`, `top_3`, `health_state`, `stability_string`, `warnings` | MUST | Iter 1 writes `{"iter":1,...}` — file missing → created transparently. Iter 50 appends line 50. `mcts_root_q` verified equal to root.value_sum/root.visits. All printed distributional stats present as JSONL fields |
| O2 | **Crash safety**: Single-line-per-iteration write is atomic at OS page level when flushed. No temporary file, no atomic rename | MUST | Crash mid-write at iter 50 → last valid line is iter 49; partial line ignored on read |
| O3 | **Versioning**: Every entry MUST include `metric_version` (schema semver), `training_version` (from release), `git_commit` (SHA), `architecture` (arch tag), `config_hash` (SHA256 of active config) | MUST | Entry at iter 10 contains `"metric_version":"1.0.0","git_commit":"abc123","config_hash":"e3b0c442"` |
| O4 | **Promotion enrichment**: `PromotionRecord` MUST add `gap_since_last::Int`, `total_promotions_at_event::Int`, `elo_candidate::Float64`, `elo_best::Float64`, `timestamp::String` (ISO 8601) | MUST | Promotion #3 at iter 100, gap 35 from #2, candidate 1540, best 1520 → record populated with all 5 enrichment fields |
| O5 | **Historical promo summary**: On promotion, compute and print: total promotions, avg/median/best/closest winrate, largest gap, longest streak, avg gap | SHOULD | After promo #5: `Promo #5 summary: total=5 avg_wr=57.2% median=58.1% best=62.0% closest=+1.2% lgst_gap=50 lgst_streak=12 avg_gap=28` |
| O6 | **Convergence detection** (stability fix): Passive indicators only — sliding window (default 20 iter) checks: KL stability (variance < εₖₗ), Drift stability (variance < ε_d), Top1 stability (variance < ε_t1), param update magnitude (< ε_p using actual `||θ_new−θ_old||₂` norm, not policy loss variance). Prints ACTIVE/STALLED per metric | SHOULD | KL variance < εₖₗ for 20 iters → `KL: STALLED`. `Param:` check uses param_update_norm sliding window, not ma_buf_policy_loss |
| O7 | **Health dashboard** (multi-signal + full states): Compact single printed line per iteration: Network learning (ACTIVE/STALLED/BOOTSTRAP), Search usefulness (HIGH/LOW from Top1 AND KL AND Policy Distance simultaneously), Model drift (LOW/MEDIUM/HIGH from drift KL), Promotion frequency, Replay fill %, Value calibration (OK/WARN/N/A). All states: ACTIVE, STALLED, HIGH, LOW, BOOTSTRAP, OK, N/A. BOOTSTRAP triggers when replay buffer < minimum fill. ValCal state documented with rationale | SHOULD | Top1>60% AND KL>ε_kl AND pd>ε_pd → `Srch:HIGH`. Buffer<start → `Net:BOOTSTRAP` others N/A |
| O8 | **Diagnostic warnings**: Print warnings when Top-1 > 95% (policy near-deterministic), Drift ≈ 0 (< ε, model unchanging), param update > 5× rolling mean (potential instability). No circuit breaking, no behavioral change | SHOULD | Top-1 = 97% → `⚠ Top-1 agreement 97% > 95% — policy near-deterministic` |
| O9 | **Cost documentation** (doc schema): Every metric MUST document: Definition (mathematical), Formula (LaTeX), Variables (inputs/sources), Implementation (code path), Computational Cost (O expr + μs + bytes), Interpretation (meaning of high/low), Typical observations (project-specific empirical ranges), Notes (edge cases). Theory MUST be separated from empirical. Log base MUST be explicit ln. Top-K: "MCTS's top-1 action in network's top-K" | MUST | Each metric entry has all 8 fields. Top-K matches implementation (set membership). Log base spelled as natural log |
| CR-O1 | **Search Gain**: `search_with_stats` MUST return 5th value `raw_network_value` (root network value from first MCTS expansion). TrainingResult MUST carry `root_q_mean` and `search_gain`. Search Gain = MCTS_root_Q − raw_network_value. JSONL persists both | MUST | Search completes → search_gain stored in TrainingResult and JSONL. Zero extra forwards — reuses root.value from first node expansion |
| CR-O2 | **Interpretation guide**: `docs/metrics.md` MUST include `## Metric Guide` with 4+ combination patterns: converging network (KL↓, Top1↗, loss↓), active search (Top1 moderate, KL moderate, Drift > ε_kld), rapid changes (Δ-metrics ↑, grad_norm ↑), stalled (all variances → 0) | MUST | Reader opens docs → finds all 4 patterns with interpretations |
| CR-O3 | **Metrics classification**: All metrics MUST be grouped into 7 categories: Optimization, Policy Learning, Search, Value, Replay, Promotion, Network Evolution | MUST | Each metric section in docs includes its classification group label |

## Constraints

- No changes to training algorithm, MCTS behavior, self-play, promotion criteria, hyperparameters, optimizer, scheduler, or replay buffer
- Convergence detection, health dashboard, and warnings are purely additive observation — MUST NOT alter control flow or training decisions
- CSV persistence from `training-metrics` MUST remain unchanged; JSONL is additive
- JSONL MUST use `JSON3.jl`; file opened once in append mode, held open for duration of training
- All existing tests MUST pass unchanged
