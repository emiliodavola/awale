"""
    Evaluation

Agent definitions and match evaluation: agent types (random, heuristic, model),
match play, opening suite generation, and batched agent evaluation.
"""
module Evaluation

using ..State: GameState, initial_state, GameConfig
using ..Env: is_terminal, transition, legal_actions, reward
using ..MCTS: MCTSSearch, search
using Random

export RandomAgent, HeuristicAgent, ModelAgent, evaluate_agents, evaluate_agents_on_openings, generate_opening_suite, play_match

"""
    result_from_terminal_state(state::GameState) -> Int

Convert a terminal `reward` value to a match result from player 1's perspective:
1 (P1 wins), -1 (P2 wins), or 0 (draw).
"""
function result_from_terminal_state(state::GameState)::Int
    terminal_reward = reward(state)
    p1_perspective_reward = state.to_move == 1 ? terminal_reward : -terminal_reward
    return p1_perspective_reward > 0 ? 1 : p1_perspective_reward < 0 ? -1 : 0
end

"""
    result_from_cutoff_state(state::GameState) -> Int

Return a match result from captures when the game was cut off (max turns reached):
1 if player 1 has more captures, -1 if player 2 has more, 0 for a draw.
"""
function result_from_cutoff_state(state::GameState)::Int
    if state.captured[1] > state.captured[2]
        return 1
    elseif state.captured[2] > state.captured[1]
        return -1
    end

    return 0
end

"""
    MatchOutcome

Result of a single match: winner, turns played, and whether it ended by cutoff.

# Fields
- result::Int — 1 (P1 wins), -1 (P2 wins), 0 (draw)
- turns_played::Int — number of turns before termination
- cutoff::Bool — true if the match hit max_turns without a terminal state
"""
struct MatchOutcome
    result::Int
    turns_played::Int
    cutoff::Bool
end

"""
    Base.iterate(outcome::MatchOutcome)

Destructure a `MatchOutcome` into `(result, turns_played)` for convenient unpacking.
"""
Base.iterate(outcome::MatchOutcome) = (outcome.result, 1)
Base.iterate(outcome::MatchOutcome, state::Int) = state == 1 ? (outcome.turns_played, 2) : nothing

"""
    RandomAgent

Agent that selects uniformly among legal actions. Baseline for comparison.
"""
struct RandomAgent end

"""
    select_action(::RandomAgent, s::GameState, rng=Random.default_rng()) -> Int

Select a legal action uniformly at random. Baseline agent for comparison.
"""
function select_action(::RandomAgent, s::GameState, rng=Random.default_rng())
    actions = legal_actions(s)
    return actions[rand(rng, 1:length(actions))]
end

"""
    HeuristicAgent

Greedy agent that picks the action maximizing immediate captures.
Simple heuristic baseline for comparison.
"""
struct HeuristicAgent end

"""
    select_action(::HeuristicAgent, s::GameState, rng=Random.default_rng()) -> Int

Select the legal action that maximises immediate captures (greedy heuristic).
"""
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

"""
    ModelAgent

Neural network agent that uses MCTS to select actions.
Wraps an `MCTSSearch` instance with a fixed simulation budget.
"""
struct ModelAgent
    mcts::MCTSSearch
    sims::Int
end

"""
    select_action(agent::ModelAgent, s::GameState, rng=Random.default_rng()) -> Int

Select an action using MCTS search with the agent's neural model.
"""
function select_action(agent::ModelAgent, s::GameState, rng=Random.default_rng())
    return search(agent.mcts, s, agent.sims, rng; add_root_noise=false)
end

"""
    play_match_from_state(initial_state, agent_p1, agent_p2, rng; max_turns) -> MatchOutcome

Play a match starting from a given state, alternating agents.
Returns a `MatchOutcome` with result, turn count, and cutoff flag.
"""
function play_match_from_state(initial_state::GameState, agent_p1, agent_p2, rng; max_turns::Int)
    state = initial_state
    turn = 1
    turns_played = 0

    while !is_terminal(state) && turns_played < max_turns
        current_agent = turn == 1 ? agent_p1 : agent_p2
        action = select_action(current_agent, state, rng)
        state = transition(state, action)
        turn = turn == 1 ? 2 : 1
        turns_played += 1
    end

    cutoff = !is_terminal(state)
    result = cutoff ? result_from_cutoff_state(state) : result_from_terminal_state(state)
    return MatchOutcome(result, turns_played, cutoff)
end

"""
    play_match(agent_p1, agent_p2, config, rng; max_turns) -> MatchOutcome

Play a match from the initial position with the given game config.
"""
function play_match(agent_p1, agent_p2, config, rng; max_turns::Int)
    return play_match_from_state(initial_state(config), agent_p1, agent_p2, rng; max_turns=max_turns)
end

"""
    generate_opening_suite(; plies, openings_per_ply, seed, config) -> Vector{GameState}

Generate a reproducible set of opening positions for agent evaluation.
For each ply depth, creates `openings_per_ply` random positions.
"""
function generate_opening_suite(; plies::Vector{Int}, openings_per_ply::Int, seed::Int, config::GameConfig=GameConfig())
    rng = MersenneTwister(seed)
    openings = GameState[]

    for ply_count in plies
        for _ in 1:openings_per_ply
            state = initial_state(config)
            for _ in 1:ply_count
                actions = legal_actions(state)
                isempty(actions) && break
                action = actions[rand(rng, 1:length(actions))]
                state = transition(state, action)
                is_terminal(state) && break
            end
            push!(openings, state)
        end
    end

    return openings
end

"""
    evaluate_agents_on_openings(agent1, agent2, openings, n_games, rng; max_turns) -> NamedTuple

Evaluate two agents across a suite of opening positions, swapping sides.
Returns a named tuple with wins, losses, draws, and average turns.
"""
function evaluate_agents_on_openings(agent1, agent2, openings, n_games::Int, rng; max_turns::Int)
    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    for game_idx in 1:n_games
        opening = openings[mod1(game_idx, length(openings))]
        if game_idx % 2 == 0
            result, turns = play_match_from_state(opening, agent1, agent2, rng; max_turns=max_turns)
        else
            result, turns = play_match_from_state(opening, agent2, agent1, rng; max_turns=max_turns)
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

"""
    evaluate_agents(agent1, agent2, n_games, config, rng; max_turns) -> NamedTuple

Evaluate two agents across `n_games` from the initial position, swapping sides.
Returns a named tuple with wins, losses, draws, and average turns.
"""
function evaluate_agents(agent1, agent2, n_games::Int, config, rng; max_turns::Int)
    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    for game_idx in 1:n_games
        if game_idx % 2 == 0
            result, turns = play_match(agent1, agent2, config, rng; max_turns=max_turns)
        else
            result, turns = play_match(agent2, agent1, config, rng; max_turns=max_turns)
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