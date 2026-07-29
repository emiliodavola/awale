# 07_training: Current training loop and checkpoint policy

This spec describes the **current** training behavior, not an idealized future checkpointing system.

## Quick path

1. Self-play generates `(state, π, z)` samples.
2. Samples accumulate in a ring replay buffer.
3. Each iteration runs `updates_per_iteration` gradient updates from replay batches.
4. Training writes `model_last.bin`, `model_best.bin`, `model_final.bin`, optional numbered snapshots, and `training_state.toml`.

## Data flow

- Replay items are:
  - `state::GameState`
  - `pi_target::Vector{Float32}`
  - `z_target::Float32`
- Self-play uses:
  - root noise enabled during self-play search
  - temperature-based action sampling for early moves
  - value-target backfilling by alternating perspective signs backward through the game

## Optimization contract

- Batches are sampled with `sample_batch(buffer, batch_size, rng)`.
- `train_step` optimizes policy cross-entropy + value MSE.
- `run_training_iteration` performs:
  - `n_games` self-play games
  - replay insertion
  - `updates_per_iteration` replay updates

## Checkpoint policy

| Artifact | Meaning |
|---|---|
| `model_last.bin` | latest model after each iteration |
| `model_best.bin` | best model by win rate vs `RandomAgent` |
| `model_final.bin` | terminal artifact for the configured run |
| `model_iter_N.bin` | numbered milestone snapshot |
| `training_state.toml` | lightweight resume state: `resume_contract`, `last_iter`, `best_selection_score` (`best_win_rate` legacy fallback) |

### Numbered snapshot rule

Numbered snapshots are saved only when:
- iteration `== 1`
- iteration is a power of two
- iteration is a multiple of `checkpoint_every`

The final run state is represented by `model_final.bin`, not by forcing a numbered final snapshot.

## Resume semantics

- Preferred resume path: `model_last.bin` + `training_state.toml`.
- `training_state.toml` records `resume_contract = "weights-only"` so the current contract is explicit.
- Resume is intentionally weights-only: the optimizer is recreated, the replay buffer starts fresh, and RNG state is not persisted.
- Legacy files that only contain `last_iter` and `best_win_rate` still load correctly.
- Legacy fallback: highest detected `model_iter_N.bin` if the lightweight state file is absent.

## Runtime determinism knobs

- `training.initial_model_seed` controls fresh model initialization.
- `training.bootstrap_rng_seed` controls the main training RNG bootstrap.
- `training.max_turns` is the shared turn cap used by self-play, evaluation, arena, and interactive play paths.

## Training metrics tracking

Per-iteration metrics are collected by `src/Awale/Metrics.jl` (registered via `Awale.jl`) and injected into `train.jl`'s main loop.

| Metric | Mechanism | Persistence |
|--------|-----------|-------------|
| Parameter update norm | `Flux.destructure` before/after `run_training_iteration` | CSV + diagnostics |
| Elo rating (candidate vs best) | `EloTracker` struct; candidate updates each iter, best freezes until promotion | CSV column + printed |
| Promotion tracking | `ProgressTracker` (gaps, streaks, total), `PromotionRecord` per event | `promotion_history.toml` |
| Learning curve | 15-column CSV row per iteration | `learning_curve_<arch>_<release_id>.csv` |

### Resume behavior

- CSV appends (no duplicate header rewritten on resume).
- `promotion_history.toml` is reloaded; `ProgressTracker` state restored.
- `EloTracker` resets to (1500, 1500) — weights-only contract.

### File locations

| File | Path (relative to checkpoint namespace) |
|------|----------------------------------------|
| Learning curve CSV | `log/learning_curve_<arch>_<release_id>.csv` |
| Promotion history | `promotion_history.toml` (same dir as `training_state.toml`) |

## Determinism boundary

- Fresh training runs initialize the model in `train.jl` from `training.initial_model_seed`, so the starting weights are reproducible for any fixed configured seed.
- Training iterations, replay sampling, selection, baseline evaluation, and checkpoint arena runs already use explicit RNGs.
- This boundary does **not** include resume continuity: the optimizer, replay buffer, and RNG state are still recreated on resume.

## Important limitation

The repo still does **not** persist optimizer state, replay-buffer state, RNG state, commit hash, or exact deterministic continuation metadata. That is a conscious limit of the current checkpoint design.

Checkpoint `.bin` files are also treated as **trusted-local-only** artifacts: they are produced by this repo and loaded from the local workspace. The current implementation uses Julia `Serialization` for that internal workflow.

## MCTS Diagnostics (Intra-Iteration Telemetry)

The training loop emits 7 read-only diagnostics metrics per iteration that quantify MCTS vs network policy alignment. All metrics observe without modifying the training loop.

### Metrics

| # | Metric | Description | Data Source |
|---|--------|-------------|-------------|
| R1 | KL Divergence | Per-position KL(target \|\| network) from MCTS target and post-update network policy | `Y_pi`, `after_probs` in `train_step` |
| R2 | Top-K Agreement | Top-1/2/3 argmax agreement rate between target and network policies | `Y_pi`, `after_probs` in `train_step` |
| R3 | Root Confidence | Max visit-count probability from each game's MCTS root policy | `search_with_stats` in `collect_selfplay_data` |
| R4 | Target Entropy (Enhanced) | Distributional stats (mean, median, P25/P75/P95, min, max) of per-sample target entropy | `Y_pi` in `train_step` |
| R5 | Policy Distance (L1) | Normalized L1 = sum(\|pi_network - pi_target\|) / 2, in [0, 1] | `after_probs`, `Y_pi` in `train_step` |
| R7 | Network Drift | KL(softmax(logits_iter1) \|\| softmax(logits_current)) over fixed reference state set | One extra `predict_batch_inference` call per iteration |

### Constraints

- **C1**: Metrics 1–5 MUST derive entirely from values already computed inside `train_step`. Zero new forward passes.
- **C2**: Network Drift adds exactly ONE `predict_batch_inference` call per iteration (plus one at iteration 1 for reference generation).
- **C3**: No algorithm changes to `MCTS.jl`, `Model.jl`, `ReplayBuffers.jl`, `Env.jl`, or `State.jl`.
- **C4**: Adding diagnostics MUST NOT change numerical outputs (loss values, gradients, model weights must be bit-identical to baseline).

### Output Format

After the existing Diagnostics block in `run_training_iteration`:

```
──────────────────────────────────────────────
MCTS Diagnostics
Avg KL(target || network)   Median   Max   P25 / P75 / P95
Top-1 agreement: XX.X%   Top-2: XX.X%   Top-3: XX.X%
Root confidence — Mean / Median / P25 / P50 / P75 / P95 / Min / Max
Policy distance (L1) — Mean / Median / P25 / P75 / P95
──────────────────────────────────────────────
```

After the training loop in `train.jl`:

```
──────────────────────────────────────────────
Network Drift: X.XXXX
──────────────────────────────────────────────
```

### Percentile Definition

All percentiles (P25, P50, P75, P95) MUST use linear interpolation between sorted values (Julia `Statistics.quantile` default).

### Spec Deviations (Intentional Improvements)

1. **Network drift**: Implemented as per-iteration delta (not cumulative) — provides a more useful signal for tracking short-term training dynamics.
2. **Root confidence**: Accumulated per-position (not per-game) — gives a richer distribution over all root visit counts.
3. **Reference set**: Sourced from random play (not self-play) — avoids conflating policy bias with drift measurement.

Detailed specification: `spec/changes/mcts-diagnostics/specs/mcts-diagnostics/spec.md`

## Testing checklist

- [ ] replay buffer receives self-play data
- [ ] updates run from replay batches
- [ ] resume continues from `last_iter + 1`
- [ ] milestone snapshots follow the automatic rule
- [ ] completed runs are recognized via `model_final.bin` + state
