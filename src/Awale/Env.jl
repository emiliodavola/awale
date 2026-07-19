module Env

using StaticArrays: SVector
using ..State: GameState, GameConfig, canonicalize, hash_state
using ..Utils: fnv1a64

export legal_actions, transition, is_terminal, reward, local_to_global

opponent_player(p::Int8) = p == Int8(1) ? Int8(2) : Int8(1)

same_side(i::Int, player::Int)::Bool =
    (player == 1 && 1 <= i <= 6) || (player == 2 && 7 <= i <= 12)

function side_seed_totals(s::GameState)::Tuple{Int,Int}
    p1_total = Int(s.captured[1])
    p2_total = Int(s.captured[2])

    for i in 1:6
        p1_total += Int(s.board[i])
    end
    for i in 7:12
        p2_total += Int(s.board[i])
    end

    return p1_total, p2_total
end

function terminal_score_totals(s::GameState)::Tuple{Int,Int}
    if isempty(legal_actions(s))
        return side_seed_totals(s)
    end
    return Int(s.captured[1]), Int(s.captured[2])
end

function score_to_perspective_reward(s::GameState, p1_score::Int, p2_score::Int)::Float32
    if p1_score > p2_score
        return (s.to_move == 1) ? 1.0f0 : -1.0f0
    elseif p2_score > p1_score
        return (s.to_move == 2) ? 1.0f0 : -1.0f0
    else
        return 0.0f0
    end
end

function is_starved(s::GameState, p::Integer)::Bool
    start = (p == 1) ? 1 : 7
    return all(i -> s.board[i] == 0, start:(start+5))
end

function local_to_global(action::Int, s::GameState)::Int
    return (s.to_move == 1) ? action : action + 6
end


# Helper to simulate a move without creating a full GameState object
function simulate_move(s::GameState, action::Int)
    idx = local_to_global(action, s)
    seeds = s.board[idx]
    
    # Use a local mutable array for calculation to minimize allocations in the hot-path
    board_vec = zeros(UInt8, 12)
    for i in 1:12
        board_vec[i] = s.board[i]
    end
    
    board_vec[idx] = 0
    pos = idx
    while seeds > 0
        pos = (pos % 12) + 1
        board_vec[pos] += 1
        seeds -= 1
    end
    captured = 0
    current_p = s.to_move
    is_opp = (current_p == 1) ? (7 <= pos <= 12) : (1 <= pos <= 6)
    if is_opp && (board_vec[pos] == 2 || board_vec[pos] == 3)
        temp_pos = pos
        while is_opp && (board_vec[temp_pos] == 2 || board_vec[temp_pos] == 3)
            captured += board_vec[temp_pos]
            board_vec[temp_pos] = 0
            temp_pos = (temp_pos - 2 + 12) % 12 + 1
            is_opp = (current_p == 1) ? (7 <= temp_pos <= 12) : (1 <= temp_pos <= 6)
        end
    end
    return SVector{12,UInt8}(board_vec), captured
end

function legal_actions(s::GameState)::Vector{Int}
    slice = (s.to_move == 1) ? s.board[1:6] : s.board[7:12]
    base_actions = [i for i in 1:6 if slice[i] > 0]
    if isempty(base_actions)
        return Int[]
    end
    opp = opponent_player(s.to_move)
    opp_starved = is_starved(s, opp)
    if s.config.forced_feeding == :require_feed && opp_starved
        feeding_actions = Int[]
        for a in base_actions
            board_after, _ = simulate_move(s, a)
            opp_start = (opp == 2) ? 7 : 1
            if any(i -> board_after[i] > 0, opp_start:(opp_start+5))
                push!(feeding_actions, a)
            end
        end
        if !isempty(feeding_actions)
            return feeding_actions
        end
    end
    if s.config.starvation == :prevent_starvation
        non_starving_actions = Int[]
        for a in base_actions
            board_after, _ = simulate_move(s, a)
            opp_start = (opp == 2) ? 7 : 1
            if any(i -> board_after[i] > 0, opp_start:(opp_start+5))
                push!(non_starving_actions, a)
            end
        end
        if !isempty(non_starving_actions)
            return non_starving_actions
        end
    end
    return base_actions
end


function transition(s::GameState, action::Int)::GameState
    actions = legal_actions(s)
    if !(action in actions)
        throw(ErrorException("IllegalAction: action $action is not legal under current config"))
    end
    board_vec, captured_this_turn = simulate_move(s, action)
    current_player = s.to_move
    opp_player = 3 - current_player
    new_captured = s.captured
    if current_player == 1
        new_captured = (s.captured[1] + captured_this_turn, s.captured[2])
    else
        new_captured = (s.captured[1], s.captured[2] + captured_this_turn)
    end
    new_board = board_vec # SVector{12,UInt8}(board_vec) is implicitly converted if types match
    new_to_move = Int8(opp_player)
    
    # Construct new state and update history to include the hash of the state we just left
    temp_s = GameState(new_board, new_to_move, new_captured, UInt64(0), s.config, s.history_hashes)
    h = hash_state(canonicalize(temp_s))
    new_history = copy(s.history_hashes)
    push!(new_history, s.history_hash)
    
    return GameState(new_board, new_to_move, new_captured, h, s.config, new_history)
end

function is_terminal(s::GameState)::Bool
    # 1. Capture threshold / draw.
    if Int(s.captured[1]) >= 25 || Int(s.captured[2]) >= 25
        return true
    end
    if Int(s.captured[1]) == 24 && Int(s.captured[2]) == 24
        return true
    end

    # 2. Repetition handling depends on the configured policy.
    if s.history_hash in s.history_hashes
        return s.config.repetition != :revert
    end

    # 3. End of game (no legal moves / no-feed resolution).
    if isempty(legal_actions(s))
        return true
    end

    return false
end

function reward(s::GameState)::Float32
    if s.history_hash in s.history_hashes
        if s.config.repetition == :draw_on_repeat
            return 0.0f0
        elseif s.config.repetition == :score_diff
            return score_to_perspective_reward(s, Int(s.captured[1]), Int(s.captured[2]))
        end
    end

    p1_score, p2_score = terminal_score_totals(s)
    return score_to_perspective_reward(s, p1_score, p2_score)
end

end # module
