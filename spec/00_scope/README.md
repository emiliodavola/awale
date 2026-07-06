# 00_scope: Goals and project scope

## Goals

- Build a deterministic Awale RL system in Julia.
- Keep core game/search/training contracts explicit and testable.
- Use specs to document the current behavior and intended constraints of the system.
- Prefer correctness and trustworthy evaluation before architectural escalation.

## Out of scope for the current repo state

- complex architectures beyond the current strong-MLP phase
- Elo/Glicko infrastructure
- fully reproducible checkpoint metadata with optimizer/RNG continuation
- GPU-specific optimization work before stronger experimental evidence exists

## Deliverables

- maintained specs for environment, model, training, evaluation, and testing
- deterministic environment API
- active test suite for invariants, variants, and training pipeline behavior
- reproducible-enough checkpoint comparison workflow via `checkpoint_arena.jl`

## Documentation rule

When implementation changes invalidate a documented contract, update the matching spec in the same unit of work.
