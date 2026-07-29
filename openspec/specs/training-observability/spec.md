# Training Observability Specification

## Purpose

Capture every per-iteration metric that is currently printed then discarded (MCTS diagnostics, network drift) into persistent JSONL, enrich promotion records with event context, and provide passive convergence detection, health assessment, and diagnostic warnings — all with zero behavioral change to training, MCTS, or self-play.

## Requirements

| # | Requirement | MUST/SHOULD | Scenario |
|---|-------------|-------------|----------|
| O1 | **JSONL persistence**: One JSON object per iteration at `checkpoints/<arch>/log/metrics_<arch>_<release_id>.jsonl`. Append-mode, flush per iteration, no header row | MUST | Iter 1 writes `{"iter":1,...}` — file missing → created transparently. Iter 50 appends line 50 |
| O2 | **Crash safety**: Single-line-per-iteration write is atomic at OS page level when flushed. No temporary file, no atomic rename | MUST | Crash mid-write at iter 50 → last valid line is iter 49; partial line ignored on read |
| O3 | **Versioning**: Every entry MUST include `metric_version` (schema semver), `training_version` (from release), `git_commit` (SHA), `architecture` (arch tag), `config_hash` (SHA256 of active config) | MUST | Entry at iter 10 contains `"metric_version":"1.0.0","git_commit":"abc123","config_hash":"e3b0c442"` |
| O4 | **Promotion enrichment**: `PromotionRecord` MUST add `gap_since_last::Int`, `total_promotions_at_event::Int`, `elo_candidate::Float64`, `elo_best::Float64`, `timestamp::String` (ISO 8601) | MUST | Promotion #3 at iter 100, gap 35 from #2, candidate 1540, best 1520 → record populated with all 5 enrichment fields |
| O5 | **Historical promo summary**: On promotion, compute and print: total promotions, avg/median/best/closest winrate, largest gap, longest streak, avg gap | SHOULD | After promo #5: `Promo #5 summary: total=5 avg_wr=57.2% median=58.1% best=62.0% closest=+1.2% lgst_gap=50 lgst_streak=12 avg_gap=28` |
| O6 | **Convergence detection**: Passive indicators only — sliding window (default 20 iter) checks: KL stability (variance < εₖₗ), Drift stability (variance < ε_d), Top1 stability (variance < ε_t1), param update magnitude (< ε_p). Prints ACTIVE/STALLED per metric | SHOULD | KL variance < εₖₗ for 20 iters → `KL: STALLED`. No training parameter, checkpoint, or schedule changes |
| O7 | **Health dashboard**: Compact single printed line per iteration: Network learning (ACTIVE/STALLED), Search usefulness (HIGH/LOW from Top-1), Model drift (LOW/MEDIUM/HIGH from drift KL), Promotion frequency, Replay fill %, Value calibration (OK/WARN) | SHOULD | `Net:ACTIVE Srch:HIGH Drift:LOW Promo:1/50 Rply:82% ValCal:OK` |
| O8 | **Diagnostic warnings**: Print warnings when Top-1 > 95% (policy near-deterministic), Drift ≈ 0 (< ε, model unchanging), param update > 5× rolling mean (potential instability). No circuit breaking, no behavioral change | SHOULD | Top-1 = 97% → `⚠ Top-1 agreement 97% > 95% — policy near-deterministic` |
| O9 | **Cost documentation**: Every metric MUST document time cost (O expression + measured μs), memory cost (bytes/iter), frequency (every iter / on-event), and reuse opportunities in `docs/metrics.md` | MUST | MCTS KL: `time=O(batch·samples) ~50μs, mem=~4KB, freq=every iter, reuse=already in forward pass` |

## Constraints

- No changes to training algorithm, MCTS behavior, self-play, promotion criteria, hyperparameters, optimizer, scheduler, or replay buffer
- Convergence detection, health dashboard, and warnings are purely additive observation — MUST NOT alter control flow or training decisions
- CSV persistence from `training-metrics` MUST remain unchanged; JSONL is additive
- JSONL MUST use `JSON3.jl`; file opened once in append mode, held open for duration of training
- All existing tests MUST pass unchanged
