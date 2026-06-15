module Evaluation

using ..State: GameState, initial_state, GameConfig
using ..Env: is_terminal, transition, reward, legal_actions
using ..MCTS: MCTSSearch, search
using Random

export RandomAgent, HeuristicAgent, ModelAgent, evaluate_agents, play_match

# Agent interfaces
struct RandomAgent end
function select_action(agent::RandomAgent, s::GameState)
    actions = legal_actions(s)
    return rand(actions)
end

struct HeuristicAgent end
function select_action(agent::HeuristicAgent, s::GameState)
    actions = legal_actions(s)
    if isempty(actions) return 0 end
    
    current_player = s.to_move
    best_action = actions[1]
    max_gain = -1
    
    for action in actions
        s_next = transition(s, action)
        gain = (current_player == 1) ? 
            (Int(s_next.captured[1]) - Int(s.captured[1])) : 
            (Int(s_next.captured[2]) - Int(s.captured[2]))
        
        if gain > max_gain
            max_gain = gain
            best_action = action
        end
    end
    return best_action
end

struct ModelAgent
    mcts::MCTSSearch
    sims::Int
end
function select_action(agent::ModelAgent, s::GameState)
    return search(agent.mcts, s, agent.sims)
end

# Plays a single game between two agents and returns the winner from P1 perspective and duration.
function play_match(agent_p1, agent_p2, config::GameConfig=GameConfig())
    s = initial_state(config)
    turn = 1 # Assume player 1 starts
    turns_played = 0
    max_turns = 1000
    
    while !is_terminal(s) && turns_played < max_turns
        current_agent = (turn == 1) ? agent_p1 : agent_p2
        action = select_action(current_agent, s)
        s = transition(s, action)
        turn = (turn == 1) ? 2 : 1
        turns_played += 1
    end
    
    p1_score, p2_score = s.captured
    res = 0
    if p1_score > p2_score res = 1 end
    if p2_score > p1_score res = -1 end
    
    return res, turns_played
end

function evaluate_agents(agent1, agent2, n_games::Int, config::GameConfig=GameConfig())
    wins = 0
    losses = 0
    draws = 0
    total_turns = 0
    
    for i in 1:n_games
        # Swap roles to eliminate color bias
        if i % 2 == 0
            res, turns = play_match(agent1, agent2, config) # agent1 is P1
        else
            res, turns = play_match(agent2, agent1, config) # agent1 is P2
            res = -res
        end
        
        total_turns += turns
        if res == 1 wins += 1
        elseif res == -1 losses += 1
        else draws += 1
        end
    end
    
    return (wins=wins, losses=losses, draws=draws, avg_turns=total_turns / n_games)
end

end # module
