"""
    State

Immutable game state for Awale: board representation, canonicalization,
serialization, and invariant validation.
"""
module State

using StaticArrays: SVector
using ..Utils: fnv1a64

export GameConfig, GameState, initial_state, canonicalize, serialize_state, deserialize_state, hash_state, validate_invariants, encode_state

"""
    GameConfig

Configuration flags for Awale rule variants.

# Fields
- starvation::Symbol — :allow_capture or :prevent_starvation
- grand_slam::Symbol — :allow, :forbid, or :special
- repetition::Symbol — :draw_on_repeat, :revert, or :score_diff
- forced_feeding::Symbol — :allow_move_feeding or :require_feed
"""
struct GameConfig
    starvation::Symbol
    grand_slam::Symbol
    repetition::Symbol
    forced_feeding::Symbol
end

GameConfig(; starvation::Symbol=:allow_capture, grand_slam::Symbol=:allow, repetition::Symbol=:draw_on_repeat,
forced_feeding::Symbol=:require_feed) =
    GameConfig(starvation, grand_slam, repetition, forced_feeding)

"""
    GameState

Immutable representation of a complete Awale game position.

# Fields
- board::SVector{12,UInt8} — seed counts per pit (indices 1-6: player 1, 7-12: player 2)
- to_move::Int8 — 1 or 2
- captured::NTuple{2,UInt8} — seeds captured by each player
- history_hash::UInt64 — computed on canonicalized serialize for repetition detection
- config::GameConfig — rule variant flags
- history_hashes::Set{UInt64} — set of past history hashes for repetition detection
"""
struct GameState
    board::SVector{12,UInt8}
    to_move::Int8
    captured::NTuple{2,UInt8}
    history_hash::UInt64
    config::GameConfig
    history_hashes::Set{UInt64}
end

"""
    initial_state(config::GameConfig=GameConfig()) -> GameState

Create a new game with 4 seeds per pit (48 total), player 1 to move,
and no captures. History is initialized with the starting state hash.
"""
function initial_state(config::GameConfig=GameConfig())::GameState
    board = SVector{12,UInt8}(ntuple(_ -> UInt8(4), 12))
    to_move = Int8(1)
    captured = (UInt8(0), UInt8(0))
    history = Set{UInt64}()

    h = hash_state(GameState(board, to_move, captured, UInt64(0), config, history))
    return GameState(board, to_move, captured, h, config, history)
end

rotate_board(b::SVector{12,UInt8}, k::Int) = SVector{12,UInt8}(ntuple(i -> b[mod1(i + k, 12)], 12))

"""
    canonicalize(s::GameState) -> GameState

Return the canonical representation of `s` from player 1's perspective.
If `s.to_move == 2`, the board is rotated 180° (6 pits) and captures are swapped.
"""
function canonicalize(s::GameState)::GameState
    history = copy(s.history_hashes)

    if s.to_move == 1
        canonical_state = GameState(s.board, Int8(1), s.captured, UInt64(0), s.config, history)
        h = hash_state(canonical_state)
        return GameState(s.board, Int8(1), s.captured, h, s.config, history)
    end

    board = rotate_board(s.board, 6)
    captured = (s.captured[2], s.captured[1])
    canonical_state = GameState(board, Int8(1), captured, UInt64(0), s.config, history)
    h = hash_state(canonical_state)
    return GameState(board, Int8(1), captured, h, s.config, history)
end

const CONFIG_ORDER = (:starvation, :grand_slam, :repetition, :forced_feeding)
const STARVATION_MAP = Dict(:allow_capture => UInt8(0), :prevent_starvation => UInt8(1))
const GRANDSLAM_MAP = Dict(:allow => UInt8(0), :forbid => UInt8(1), :special => UInt8(2))
const REPETITION_MAP = Dict(:draw_on_repeat => UInt8(0), :revert => UInt8(1), :score_diff => UInt8(2))
const FEEDING_MAP = Dict(:allow_move_feeding => UInt8(0), :require_feed => UInt8(1))
const STARVATION_MAP_REV = Dict(v => k for (k, v) in STARVATION_MAP)
const GRANDSLAM_MAP_REV = Dict(v => k for (k, v) in GRANDSLAM_MAP)
const REPETITION_MAP_REV = Dict(v => k for (k, v) in REPETITION_MAP)
const FEEDING_MAP_REV = Dict(v => k for (k, v) in FEEDING_MAP)

"""
    serialize_state(s::GameState) -> Vector{UInt8}

Serialize `s` to a compact byte vector for hashing and transmission.
Format: version(1) + board(12) + to_move(1) + captured(2) + config(4).
"""
function serialize_state(s::GameState)::Vector{UInt8}
    buf = Vector{UInt8}(undef, 1 + 12 + 1 + 2 + 4)

    idx = 1
    buf[idx] = UInt8(1); idx += 1
    for i in 1:12
        buf[idx] = s.board[i]; idx += 1
    end
    buf[idx] = UInt8(s.to_move); idx += 1
    buf[idx] = s.captured[1]; idx += 1
    buf[idx] = s.captured[2]; idx += 1
    buf[idx] = get(STARVATION_MAP, s.config.starvation, UInt8(255)); idx += 1
    buf[idx] = get(GRANDSLAM_MAP, s.config.grand_slam, UInt8(255)); idx += 1
    buf[idx] = get(REPETITION_MAP, s.config.repetition, UInt8(255)); idx += 1
    buf[idx] = get(FEEDING_MAP, s.config.forced_feeding, UInt8(255)); idx += 1
    return buf
end

"""
    deserialize_state(bytes::Vector{UInt8}) -> GameState

Reconstruct a `GameState` from the byte vector produced by `serialize_state`.
Throws `ArgumentError` if the payload is truncated or has an unknown version.
"""
function deserialize_state(bytes::Vector{UInt8})::GameState
    if length(bytes) < 1 + 12 + 1 + 2 + 4
        throw(ArgumentError("byte vector too short for GameState"))
    end
    i = 1
    version = bytes[i]; i += 1
    if version != UInt8(1)
        throw(ArgumentError("unsupported serialization version: $version"))
    end

    board = ntuple(k -> UInt8(bytes[i + k - 1]), 12)
    i += 12
    to_move = Int8(bytes[i]); i += 1
    captured = (bytes[i], bytes[i + 1]); i += 2
    b1 = bytes[i]; i += 1
    b2 = bytes[i]; i += 1
    b3 = bytes[i]; i += 1
    b4 = bytes[i]; i += 1

    starvation = get(STARVATION_MAP_REV, b1, :allow_capture)
    grand_slam = get(GRANDSLAM_MAP_REV, b2, :allow)
    repetition = get(REPETITION_MAP_REV, b3, :draw_on_repeat)
    # Preserve legacy serialized states that used the older allow_move_feeding default.
    forced_feeding = get(FEEDING_MAP_REV, b4, :allow_move_feeding)
    cfg = GameConfig(starvation, grand_slam, repetition, forced_feeding)
    state = GameState(board, to_move, captured, UInt64(0), cfg, Set{UInt64}())
    h = hash_state(canonicalize(state))
    return GameState(board, to_move, captured, h, cfg, Set{UInt64}())
end

"""
    hash_state(s::GameState) -> UInt64

Deterministic FNV-1a 64-bit hash of the serialized representation.
"""
function hash_state(s::GameState)::UInt64
    return fnv1a64(serialize_state(s))
end

"""
    validate_invariants(s::GameState) -> Bool

Check that seed conservation (sum = 48) holds and to_move is 1 or 2.
"""
function validate_invariants(s::GameState)::Bool
    total = zero(UInt16)
    for i in 1:12
        total += UInt16(s.board[i])
    end
    total += UInt16(s.captured[1])
    total += UInt16(s.captured[2])
    if total != UInt16(48)
        return false
    end
    if !(s.to_move == 1 || s.to_move == 2)
        return false
    end
    return true
end

"""
    encode_state(s::GameState) -> Matrix{Float32}

Encode state into a structured tensor for neural network input.
Returns a Matrix{Float32}(4, 12) with channels:
  1: board seeds normalized to [0, 1]
  2: player indicator (1.0 for player 1, 0.0 for player 2)
  3: player 1 captures normalized
  4: player 2 captures normalized
"""
function encode_state(s::GameState)::Matrix{Float32}
    x = Matrix{Float32}(undef, 4, 12)

    for i in 1:12
        x[1, i] = Float32(s.board[i]) / 48.0f0
    end

    player_val = s.to_move == 1 ? 1.0f0 : 0.0f0
    for i in 1:12
        x[2, i] = player_val
    end

    cap1 = Float32(s.captured[1]) / 48.0f0
    for i in 1:12
        x[3, i] = cap1
    end

    cap2 = Float32(s.captured[2]) / 48.0f0
    for i in 1:12
        x[4, i] = cap2
    end

    return x
end

end # module