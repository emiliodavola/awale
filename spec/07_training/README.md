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

## Determinism boundary

- Fresh training runs initialize the model in `train.jl` from `training.initial_model_seed`, so the starting weights are reproducible for any fixed configured seed.
- Training iterations, replay sampling, selection, baseline evaluation, and checkpoint arena runs already use explicit RNGs.
- This boundary does **not** include resume continuity: the optimizer, replay buffer, and RNG state are still recreated on resume.

## Important limitation

The repo still does **not** persist optimizer state, replay-buffer state, RNG state, commit hash, or exact deterministic continuation metadata. That is a conscious limit of the current checkpoint design.

Checkpoint `.bin` files are also treated as **trusted-local-only** artifacts: they are produced by this repo and loaded from the local workspace. The current implementation uses Julia `Serialization` for that internal workflow.

## Testing checklist

- [ ] replay buffer receives self-play data
- [ ] updates run from replay batches
- [ ] resume continues from `last_iter + 1`
- [ ] milestone snapshots follow the automatic rule
- [ ] completed runs are recognized via `model_final.bin` + state
