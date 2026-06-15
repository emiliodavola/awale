<<<<<<< ours
module State

using ..Utils: fnv1a64

export GameConfig, GameState, initial_state, canonicalize, serialize_state, deserialize_state, hash_state, validate_invariants

"""
GameConfig: variant flags for Awale rules
"""
struct GameConfig
    starvation::Symbol
    grand_slam::Symbol
    repetition::Symbol
    forced_feeding::Symbol
end

GameConfig(;starvation::Symbol=:allow_capture, grand_slam::Symbol=:allow, repetition::Symbol=:draw_on_repeat, forced_feeding::Symbol=:allow_move_feeding) =
    GameConfig(starvation, grand_slam, repetition, forced_feeding)

"""
GameState: immutable representation of the game
- board: NTuple{12,UInt8}
- to_move: Int8 (1 or 2)
- captured: NTuple{2,UInt8}
- history_hash: UInt64 (computed on canonicalized serialize)
- config: GameConfig
"""
struct GameState
    board::NTuple{12,UInt8}
    to_move::Int8
    captured::NTuple{2,UInt8}
    history_hash::UInt64
    config::GameConfig
end

# Create canonical initial state: 4 seeds per pit, player 1 to move
function initial_state(config::GameConfig=GameConfig())::GameState
    board = ntuple(i->UInt8(4), 12)
    to_move = Int8(1)
    captured = (UInt8(0), UInt8(0))
    s = GameState(board, to_move, captured, UInt64(0), config)
    h = hash_state(canonicalize(s))
    return GameState(board, to_move, captured, h, config)
end

# Helper: rotate board by k positions (positive k rotates left)
rotate_board(b::NTuple{12,UInt8}, k::Int) = ntuple(i->b[ mod1(i + k, 12) ], 12)

# Canonicalize: rotate so current player's pits are indices 1..6 and to_move set to 1
function canonicalize(s::GameState)::GameState
    if s.to_move == 1
        return s  # already canonical with to_move==1
    else
        # rotate by 6 so positions 7..12 become 1..6
        board = rotate_board(s.board, 6)
        captured = (s.captured[2], s.captured[1])
        cfg = s.config
        return GameState(board, Int8(1), captured, UInt64(0), cfg)
    end
end

# Deterministic serialization: version byte + board (12 bytes) + to_move (1) + captured (2) + config codes (4 bytes)
const CONFIG_ORDER = (:starvation, :grand_slam, :repetition, :forced_feeding)
const STARVATION_MAP = Dict(:allow_capture=>UInt8(0), :prevent_starvation=>UInt8(1))
const GRANDSLAM_MAP = Dict(:allow=>UInt8(0), :forbid=>UInt8(1), :special=>UInt8(2))
const REPETITION_MAP = Dict(:draw_on_repeat=>UInt8(0), :revert=>UInt8(1), :score_diff=>UInt8(2))
const FEEDING_MAP = Dict(:allow_move_feeding=>UInt8(0), :require_feed=>UInt8(1))
# reverse maps
const STARVATION_MAP_REV = Dict(v=>k for (k,v) in STARVATION_MAP)
const GRANDSLAM_MAP_REV = Dict(v=>k for (k,v) in GRANDSLAM_MAP)
const REPETITION_MAP_REV = Dict(v=>k for (k,v) in REPETITION_MAP)
const FEEDING_MAP_REV = Dict(v=>k for (k,v) in FEEDING_MAP)

function serialize_state(s::GameState)::Vector{UInt8}
    buf = Vector{UInt8}()
    push!(buf, UInt8(1))  # version
    for i in 1:12
        push!(buf, s.board[i])
    end
    push!(buf, UInt8(s.to_move))
    push!(buf, s.captured[1])
    push!(buf, s.captured[2])
    # config
    push!(buf, get(STARVATION_MAP, s.config.starvation, UInt8(255)))
    push!(buf, get(GRANDSLAM_MAP, s.config.grand_slam, UInt8(255)))
    push!(buf, get(REPETITION_MAP, s.config.repetition, UInt8(255)))
    push!(buf, get(FEEDING_MAP, s.config.forced_feeding, UInt8(255)))
    return buf
end

function deserialize_state(bytes::Vector{UInt8})::GameState
    if length(bytes) < 1 + 12 + 1 + 2 + 4
        throw(ArgumentError("byte vector too short for GameState"))
    end
    i = 1
    version = bytes[i]; i+=1
    if version != UInt8(1)
        throw(ArgumentError("unsupported serialization version: $version"))
    end
    board = ntuple(k->UInt8(bytes[i + k - 1]), 12)
    i += 12
    to_move = Int8(bytes[i]); i += 1
    captured = (bytes[i], bytes[i+1]); i += 2
    b1 = bytes[i]; i += 1
    b2 = bytes[i]; i += 1
    b3 = bytes[i]; i += 1
    b4 = bytes[i]; i += 1
    starvation = get(STARVATION_MAP_REV, b1, :allow_capture)
    grand_slam = get(GRANDSLAM_MAP_REV, b2, :allow)
    repetition = get(REPETITION_MAP_REV, b3, :draw_on_repeat)
    forced_feeding = get(FEEDING_MAP_REV, b4, :allow_move_feeding)
    cfg = GameConfig(starvation, grand_slam, repetition, forced_feeding)
    s = GameState(board, to_move, captured, UInt64(0), cfg)
    h = hash_state(canonicalize(s))
    return GameState(board, to_move, captured, h, cfg)
end

function hash_state(s::GameState)::UInt64
    bytes = serialize_state(s)
    return fnv1a64(bytes)
end

function validate_invariants(s::GameState)::Bool
    # seed conservation
    total = zero(UInt16)
    for i in 1:12
        total += UInt16(s.board[i])
    end
    total += UInt16(s.captured[1])
    total += UInt16(s.captured[2])
    if total != UInt16(48)
        return false
    end
    # to_move validity
    if !(s.to_move == 1 || s.to_move == 2)
        return false
    end
    return true
end

end # module
||||||| base
=======
module State

using StaticArrays: SVector
using ..Utils: fnv1a64

export GameConfig, GameState, initial_state, canonicalize, serialize_state, deserialize_state, hash_state, validate_invariants

"""
GameConfig: variant flags for Awale rules
"""
struct GameConfig
    starvation::Symbol
    grand_slam::Symbol
    repetition::Symbol
    forced_feeding::Symbol
end

GameConfig(;starvation::Symbol=:allow_capture, grand_slam::Symbol=:allow, repetition::Symbol=:draw_on_repeat, forced_feeding::Symbol=:allow_move_feeding) =
    GameConfig(starvation, grand_slam, repetition, forced_feeding)

"""
GameState: immutable representation of the game
- board: NTuple{12,UInt8}
- to_move: Int8 (1 or 2)
- captured: NTuple{2,UInt8}
- history_hash: UInt64 (computed on canonicalized serialize)
- config: GameConfig
"""
struct GameState
    board::SVector{12,UInt8}
    to_move::Int8
    captured::NTuple{2,UInt8}
    history_hash::UInt64
    config::GameConfig
end

# Create canonical initial state: 4 seeds per pit, player 1 to move
function initial_state(config::GameConfig=GameConfig())::GameState
    board = SVector{12,UInt8}(ntuple(i->UInt8(4), 12))
    to_move = Int8(1)
    captured = (UInt8(0), UInt8(0))
    s = GameState(board, to_move, captured, UInt64(0), config)
    h = hash_state(canonicalize(s))
    return GameState(board, to_move, captured, h, config)
end

# Helper: rotate board by k positions (positive k rotates left)
rotate_board(b::SVector{12,UInt8}, k::Int) = SVector{12,UInt8}(ntuple(i->b[ mod1(i + k, 12) ], 12))

# Canonicalize: rotate so current player's pits are indices 1..6 and to_move set to 1
function canonicalize(s::GameState)::GameState
    if s.to_move == 1
        return s  # already canonical with to_move==1
    else
        # rotate by 6 so positions 7..12 become 1..6
        board = rotate_board(s.board, 6)
        captured = (s.captured[2], s.captured[1])
        cfg = s.config
        return GameState(board, Int8(1), captured, UInt64(0), cfg)
    end
end

# Deterministic serialization: version byte + board (12 bytes) + to_move (1) + captured (2) + config codes (4 bytes)
const CONFIG_ORDER = (:starvation, :grand_slam, :repetition, :forced_feeding)
const STARVATION_MAP = Dict(:allow_capture=>UInt8(0), :prevent_starvation=>UInt8(1))
const GRANDSLAM_MAP = Dict(:allow=>UInt8(0), :forbid=>UInt8(1), :special=>UInt8(2))
const REPETITION_MAP = Dict(:draw_on_repeat=>UInt8(0), :revert=>UInt8(1), :score_diff=>UInt8(2))
const FEEDING_MAP = Dict(:allow_move_feeding=>UInt8(0), :require_feed=>UInt8(1))
# reverse maps
const STARVATION_MAP_REV = Dict(v=>k for (k,v) in STARVATION_MAP)
const GRANDSLAM_MAP_REV = Dict(v=>k for (k,v) in GRANDSLAM_MAP)
const REPETITION_MAP_REV = Dict(v=>k for (k,v) in REPETITION_MAP)
const FEEDING_MAP_REV = Dict(v=>k for (k,v) in FEEDING_MAP)

function serialize_state(s::GameState)::Vector{UInt8}
    buf = Vector{UInt8}()
    push!(buf, UInt8(1))  # version
    for i in 1:12
        push!(buf, s.board[i])
    end
    push!(buf, UInt8(s.to_move))
    push!(buf, s.captured[1])
    push!(buf, s.captured[2])
    # config
    push!(buf, get(STARVATION_MAP, s.config.starvation, UInt8(255)))
    push!(buf, get(GRANDSLAM_MAP, s.config.grand_slam, UInt8(255)))
    push!(buf, get(REPETITION_MAP, s.config.repetition, UInt8(255)))
    push!(buf, get(FEEDING_MAP, s.config.forced_feeding, UInt8(255)))
    return buf
end

function deserialize_state(bytes::Vector{UInt8})::GameState
    if length(bytes) < 1 + 12 + 1 + 2 + 4
        throw(ArgumentError("byte vector too short for GameState"))
    end
    i = 1
    version = bytes[i]; i+=1
    if version != UInt8(1)
        throw(ArgumentError("unsupported serialization version: $version"))
    end
    board = ntuple(k->UInt8(bytes[i + k - 1]), 12)
    i += 12
    to_move = Int8(bytes[i]); i += 1
    captured = (bytes[i], bytes[i+1]); i += 2
    b1 = bytes[i]; i += 1
    b2 = bytes[i]; i += 1
    b3 = bytes[i]; i += 1
    b4 = bytes[i]; i += 1
    starvation = get(STARVATION_MAP_REV, b1, :allow_capture)
    grand_slam = get(GRANDSLAM_MAP_REV, b2, :allow)
    repetition = get(REPETITION_MAP_REV, b3, :draw_on_repeat)
    forced_feeding = get(FEEDING_MAP_REV, b4, :allow_move_feeding)
    cfg = GameConfig(starvation, grand_slam, repetition, forced_feeding)
    s = GameState(board, to_move, captured, UInt64(0), cfg)
    h = hash_state(canonicalize(s))
    return GameState(board, to_move, captured, h, cfg)
end

function hash_state(s::GameState)::UInt64
    bytes = serialize_state(s)
    return fnv1a64(bytes)
end

function validate_invariants(s::GameState)::Bool
    # seed conservation
    total = zero(UInt16)
    for i in 1:12
        total += UInt16(s.board[i])
    end
    total += UInt16(s.captured[1])
    total += UInt16(s.captured[2])
    if total != UInt16(48)
        return false
    end
    # to_move validity
    if !(s.to_move == 1 || s.to_move == 2)
        return false
    end
    return true
end

end # module
>>>>>>> theirs
