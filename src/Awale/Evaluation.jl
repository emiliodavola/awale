module Evaluation

using ..State: GameState, initial_state, GameConfig
using ..Env: is_terminal, transition, legal_actions, reward
using ..MCTS: MCTSSearch, search
using Random

export RandomAgent, HeuristicAgent, ModelAgent, evaluate_agents, play_match

function result_from_terminal_state(state::GameState)::Int
    terminal_reward = reward(state)
    p1_perspective_reward = state.to_move == 1 ? terminal_reward : -terminal_reward
    return p1_perspective_reward > 0 ? 1 : p1_perspective_reward < 0 ? -1 : 0
end

struct RandomAgent end

function select_action(::RandomAgent, s::GameState, rng=Random.default_rng())
    actions = legal_actions(s)
    return actions[rand(rng, 1:length(actions))]
end

struct HeuristicAgent end

function select_action(::HeuristicAgent, s::GameState, rng=Random.default_rng())
    actions = legal_actions(s)
    isempty(actions) && return 0

    current_player = s.to_move
    best_action = actions[1]
    max_gain = -1

    for action in actions
        next_state = transition(s, action)
        gain = current_player == 1 ?
            (Int(next_state.captured[1]) - Int(s.captured[1])) :
            (Int(next_state.captured[2]) - Int(s.captured[2]))

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

function select_action(agent::ModelAgent, s::GameState, rng=Random.default_rng())
    return search(agent.mcts, s, agent.sims, rng; add_root_noise=false)
end

function play_match(agent_p1, agent_p2, config::GameConfig=GameConfig(), rng=Random.default_rng())
    state = initial_state(config)
    turn = 1
    turns_played = 0
    max_turns = 1000

    while !is_terminal(state) && turns_played < max_turns
        current_agent = turn == 1 ? agent_p1 : agent_p2
        action = select_action(current_agent, state, rng)
        state = transition(state, action)
        turn = turn == 1 ? 2 : 1
        turns_played += 1
    end

    return result_from_terminal_state(state), turns_played
end

function evaluate_agents(agent1, agent2, n_games::Int, config::GameConfig=GameConfig(), rng=Random.default_rng())
    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    for game_idx in 1:n_games
        if game_idx % 2 == 0
            result, turns = play_match(agent1, agent2, config, rng)
        else
            result, turns = play_match(agent2, agent1, config, rng)
            result = -result
        end

        total_turns += turns
        if result == 1
            wins += 1
        elseif result == -1
            losses += 1
        else
            draws += 1
        end
    end

    return (wins=wins, losses=losses, draws=draws, avg_turns=total_turns / n_games)
end

end # module
