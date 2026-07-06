# 08_evaluation: Current evaluation workflow

This spec documents the **current** evaluation workflow used by the repository.

## Quick path

1. Use `baseline_eval.jl` for quick sanity checks.
2. Use `checkpoint_arena.jl` for checkpoint-vs-checkpoint comparison.
3. Interpret arena results before making architecture decisions.

## Baselines currently implemented

- `RandomAgent`
- `HeuristicAgent` (greedy immediate-capture heuristic)
- `ModelAgent` backed by MCTS with configurable simulation budget

## Baseline evaluation script

`baseline_eval.jl` currently evaluates:
- Model vs `RandomAgent`
- Model vs `HeuristicAgent`
- `HeuristicAgent` vs `RandomAgent`

This is a sanity check, not the main research signal once the model surpasses weak baselines.

## Checkpoint arena

`checkpoint_arena.jl` is the main comparison tool for training progress.

Current contract:
- compares **numeric checkpoints in sorted consecutive order** by default
- evaluates each pairing at `0`, `50`, and `200` simulations per side
- alternates colors
- uses a reproducible opening suite instead of always starting from one opening

### Opening suite

- fixed seed
- openings generated from `0`, `2`, `4`, and `6` random legal plies
- default suite size: `16` positions

## Determinism expectations

- Baseline and arena evaluation disable root noise for model search.
- Reproducible openings reduce single-opening bias.
- Results are still matchup-based empirical evidence, not Elo/Glicko ratings.

## Explicit non-goals in the current repo state

- No minimax baseline yet.
- No Elo/Glicko pipeline yet.
- No per-game artifact logging system yet.
- No canonical-position regression suite yet.

## Testing checklist

- [ ] alternating-color evaluation is deterministic for fixed setup
- [ ] opening suite is reproducible for fixed seed
- [ ] arena ignores special checkpoints (`last`, `best`, `final`) for the default numeric chain
- [ ] missing checkpoint directory fails gracefully
