03_environment_api: API contracts and transition semantics

Design goals
- Pure, side-effect-free transition function
- Clear, typed contracts suitable for property testing and formal reasoning
- Deterministic behavior with explicit error handling for illegal moves

Core API (Julia signatures suggested)
- function legal_actions(s::GameState)::Vector{Int}
  - Returns the list of legal actions expressed as local-side indices 0..5 (relative to to_move)

- function transition(s::GameState, action::Int)::Union{GameState, ThrowError}
  - Pure function; does not mutate s
  - On success returns new GameState after applying sowing and capture rules and updating captured counts, to_move, history_hash
  - On illegal action throws an explicit IllegalAction error (or returns Result{GameState,Error} type)

- function is_terminal(s::GameState)::Bool
  - Terminal on: no legal moves for to_move (depending on variant), or game-end conditions (e.g., all seeds captured), or repetition terminal policy

- function reward(s::GameState)::Float32
  - Terminal-only; returns outcome in {-1.0, 0.0, +1.0} from current-player perspective (or use absolute scoreboard difference normalized)

Transition semantics (pseudocode)
1. Validate action ∈ legal_actions(s)
2. Compute absolute pit index idx = local_to_global(action, s.to_move)
3. seeds = s.board[idx]; require seeds > 0
4. new_board = s.board with s.board[idx] = 0
5. pos = idx; while seeds > 0:
     pos = (pos + 1) % 12
     new_board[pos] += 1
     seeds -= 1
6. apply captures according to config: check opponent pits backwards from pos; collect captured seeds if pits in {2,3} (after sowing) and variant allows
7. If move causes starvation and variant forbids starving moves, treat as illegal (revert to pre-move state and reject)
8. Compute new captured counts and new to_move = 3 - s.to_move
9. Compute canonical_hash of canonicalize(new_state) and set history_hash
10. Return GameState(new_board, new_to_move, new_captured, history_hash, s.config)

Contracts and guarantees
- No mutation: inputs unmodified
- Deterministic outputs for same inputs
- Preserves invariants (seed conservation)
- Throws explicit errors for illegal/invalid actions

Error handling
- Define exception types: IllegalAction, StarvationViolation, InvalidState, SerializationError
- Prefer exceptions for incorrect calls; tests must assert thrown exceptions for illegal inputs

Testing hooks
- Allow injection of a deterministic RNG for any tie-breaking (for example in move selection utilities)
- Fixtures must be able to construct GameState from small textual boards for property tests