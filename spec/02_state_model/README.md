02_state_model: State representation, canonicalization, and invariants

Mathematical state
- Board vector: b ∈ N^{12} (non-negative integers)
- Current player: p ∈ {+1,-1} or {1,2} (choose one canonical representation)
- Captured scores: c = (c_1,c_2) ∈ N^2
- Variant flags: V (immutable or persistent config attached to state)
- Repetition history: H = set of previously seen canonical hashes
- Full state: s = (b,p,c,V,H)

Canonicalization
- Canonical state view MUST be from the current player's perspective.
- Define `canonicalize(s)` so that:
  - when `to_move == 1`, the board is kept in place and `to_move` is normalized to `1`;
  - when `to_move == 2`, the board is rotated by 6 pits and captured scores are swapped so the canonical view is always from Player 1's perspective.
- This canonical form is required for learning inputs and repetition/transposition keys.
- `canonicalize` must be idempotent: `canonicalize(canonicalize(s)) == canonicalize(s)`

Type contracts (Julia style)
- immutable struct GameConfig
    starvation::Symbol  # :allow_capture | :prevent_starvation
    grand_slam::Symbol  # :allow | :forbid | :special
    repetition::Symbol  # :draw_on_repeat | :revert | :score_diff
    forced_feeding::Symbol  # :allow_move_feeding | :require_feed
  end

- immutable struct GameState
    board::SVector{12,UInt8}  # use StaticArrays for compactness
    to_move::Int8             # 1 or 2
    captured::NTuple{2,UInt8}
    history_hash::UInt64      # canonicalized state hash for repetition detection
    config::GameConfig
    history_hashes::Set{UInt64}  # previous canonical hashes for repetition detection
  end

Invariants (MUST ALWAYS HOLD)
1. Seed conservation: sum(board) + captured[1] + captured[2] == 48
2. Non-negativity: ∀ i, board[i] ≥ 0
3. Turn validity: to_move ∈ {1,2}
4. Serialization consistency: deserialize(serialize(s)) == s

Serialization
- Define serialize(s)::Vector{UInt8} that encodes board, to_move, captured, and variant flags in a stable, versioned binary layout.
- Hashing: use a stable 64-bit hash (e.g., SipHash or xxhash64 with fixed seed) on serialize(canonicalize(s)) for repetition detection and transposition tables.

Determinism
- All state transitions and hashes must be deterministic across platforms when using the same Julia version; document any architecture-dependent behavior.

Performance notes
- Prefer StaticArrays.SVector{12,UInt8} for board storage in GameState to minimize allocations.
- Avoid mutable fields; construct new GameState for each transition.

Testable properties
- canonicalize maintains invariants and is idempotent
- serialize/deserialize roundtrip exactness
- seed conservation after any legal transition
- immutability guarantees (no in-place modification of input state)