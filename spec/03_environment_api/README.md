# 03_environment_api: API contracts and transition semantics

This spec documents the **current** environment contract implemented in `src/Awale/Env.jl`.

## Quick path

1. Use `legal_actions(s)` to get legal local actions.
2. Pass one of those actions to `transition(s, action)`.
3. Use `is_terminal(s)` and `reward(s)` only on the resulting state contract.

## Core API

- `legal_actions(s::GameState)::Vector{Int}`
  - Returns legal actions as **local-side indices `1..6`** relative to `s.to_move`.
- `transition(s::GameState, action::Int)::GameState`
  - Pure transition. Returns a new immutable state.
  - Throws an error when `action ∉ legal_actions(s)`.
- `is_terminal(s::GameState)::Bool`
  - Terminal on capture threshold, repetition policy, or no legal moves.
- `reward(s::GameState)::Float32`
  - Terminal reward in `{-1.0, 0.0, +1.0}` from the **current-player perspective**.
- `local_to_global(action::Int, s::GameState)::Int`
  - Maps local action `1..6` to absolute pit index `1..12`.

## Transition semantics

1. Validate `action ∈ legal_actions(s)`.
2. Map local action to absolute pit index.
3. Sow seeds forward around the 12 pits.
4. Apply backward capture logic on opponent pits according to the current rules implementation.
5. Update captured counts and switch `to_move`.
6. Recompute `history_hash` from the canonicalized next state.
7. Extend `history_hashes` with the previous state's hash.

## Variant rules currently modeled

| Area | Current contract |
|---|---|
| Forced feeding | `forced_feeding = :require_feed` filters `legal_actions` down to feeding moves when such moves exist |
| Starvation prevention | `starvation = :prevent_starvation` filters out starving moves when non-starving alternatives exist |
| Repetition | `:draw_on_repeat`, `:revert`, `:score_diff` |

## Guarantees

- No mutation of the input state.
- Deterministic output for the same state and action.
- Seed conservation must hold after every legal transition.
- Illegal actions must fail loudly.

## Testing checklist

- [ ] `legal_actions` only returns values in `1..6`
- [ ] `transition` preserves seed conservation
- [ ] illegal actions throw
- [ ] starvation / forced-feeding variants filter actions as expected
