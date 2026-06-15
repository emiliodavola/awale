module Env

using ..State: GameState, GameConfig, canonicalize, hash_state

export legal_actions, transition, is_terminal, reward, local_to_global

# local actions are 1..6 relative to current player (Julia 1-based indices)
function local_to_global(action::Int, s::GameState)::Int
    @assert 1 <= action <= 6
    if s.to_move == 1
        return action
    else
        return action + 6
    end
end

# legal_actions: return vector of local indices 1..6 with seeds > 0 on current player\'s side
function legal_actions(s::GameState)::Vector{Int}
    if s.to_move == 1
        slice = s.board[1:6]
    else
        slice = s.board[7:12]
    end
    actions = Int[]
    for i in 1:6
        if slice[i] > 0
            push!(actions, i)
        end
    end
    return actions
end

# transition: implements sowing and capture rules
function transition(s::GameState, action::Int)::GameState
    actions = legal_actions(s)
    if !(action in actions)
        throw(ErrorException("IllegalAction: action $action is not legal"))
    end

    idx = local_to_global(action, s)
    seeds = s.board[idx]
    
    # Use a mutable array for the board during transition
    new_board_vec = collect(s.board)
    new_board_vec[idx] = 0
    
    pos = idx
    while seeds > 0
        pos = (pos % 12) + 1
        new_board_vec[pos] += 1
        seeds -= 1
    end

    # Capture logic
    captured_this_turn = 0
    current_player = s.to_move
    opponent_player = 3 - current_player
    
    # Opponent pits: P1 (1..6), P2 (7..12)
    is_opponent_pit(p) = (current_player == 1) ? (7 <= p <= 12) : (1 <= p <= 6)
    
    if is_opponent_pit(pos) && (new_board_vec[pos] == 2 || new_board_vec[pos] == 3)
        # Capture consecutively backwards
        temp_pos = pos
        while is_opponent_pit(temp_pos) && (new_board_vec[temp_pos] == 2 || new_board_vec[temp_pos] == 3)
            captured_this_turn += new_board_vec[temp_pos]
            new_board_vec[temp_pos] = 0
            temp_pos = (temp_pos - 2 + 12) % 12 + 1 # go back one pit in 1..12 range
        end
    end

    # Update captured counts
    new_captured = s.captured
    if current_player == 1
        new_captured = (s.captured[1] + captured_this_turn, s.captured[2])
    else
        new_captured = (s.captured[1], s.captured[2] + captured_this_turn)
    end

    # New state attributes
    new_board = NTuple{12,UInt8}(new_board_vec)
    new_to_move = Int8(opponent_player)
    
    # Construct temporary state to calculate hash (since hash_state requires GameState)
    temp_s = GameState(new_board, new_to_move, new_captured, UInt64(0), s.config)
    h = hash_state(canonicalize(temp_s))

    return GameState(new_board, new_to_move, new_captured, h, s.config)
end

function is_terminal(s::GameState)::Bool
    # Simple terminal condition: current player has no legal moves
    if length(legal_actions(s)) == 0
        return true
    end
    # Or if one player has captured all seeds (rare but possible)
    # Total seeds = 48. If someone has > 24 they might have won, but the rule is usually "last to move" or "most seeds".
    # For now, we follow "no legal moves".
    return false
end

function reward(s::GameState)::Float32
    # Reward from current player\'s perspective.
    # If it\'s terminal because they have no moves, they probably lost or drew.
    # A better way is to compare captured counts.
    p1_score = s.captured[1]
    p2_score = s.captured[2]
    
    if p1_score > p2_score
        return (s.to_move == 1) ? 1.0f0 : -1.0f0
    elseif p2_score > p1_score
        return (s.to_move == 2) ? 1.0f0 : -1.0f0
    else
        return 0.0f0
    end
end

end # module
