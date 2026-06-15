module Training

using ..State: GameState, initial_state, GameConfig, canonicalize
using ..Env: is_terminal, transition, reward, legal_actions
using ..Model: create_model, predict, predict_raw, encode_state
using ..MCTS: MCTSSearch, search_with_stats
using Flux
using Random

export play_game, collect_selfplay_data, train_step, run_training_iteration

# plays a single game using MCTS and returns the history of (state, pi_target, v_target)
function collect_selfplay_data(mcts::MCTSSearch, config::GameConfig=GameConfig(), sims_per_move=100, rng=Random.default_rng())
    s = initial_state(config)
    data = Tuple{GameState, Vector{Float32}, Float32}[]
    turns_played = 0
    max_turns = 1000
    
    while !is_terminal(s) && turns_played < max_turns
        # Search for the best action and get visit proportions as targets
        action, pi_target = search_with_stats(mcts, s, sims_per_move, rng)
        push!(data, (s, pi_target, 0.0f0)) # v_target will be filled after game ends
        s = transition(s, action)
        turns_played += 1
    end
    
    # Final reward from the perspective of the first player in each state
    final_reward = reward(s) 
    
    # Backfill v_targets: since it\'s a zero-sum game, value alternates sign
    curr_val = final_reward
    for i in length(data):-1:1
        st, pi, _ = data[i]
        data[i] = (st, pi, curr_val)
        curr_val = -curr_val
    end
    
    return data
end

function train_step(model, optimizer, states, target_pis, target_vs)

    X = hcat([encode_state(s) for s in states]...)

    Y_pi = hcat(target_pis...)  # 6×N fijo
    Y_v = reshape(target_vs, 1, :)

    function loss_fn(m)
        logits, v = predict_raw(m, X)

        log_p = Flux.logsoftmax(logits, dims=1)

        loss_pi = -sum(Y_pi .* log_p) / size(X, 2)
        loss_v = Flux.mse(v, Y_v)

        return loss_pi + loss_v
    end

    grads = Flux.gradient(loss_fn, model)[1]
    Flux.update!(optimizer, model, grads)

    return loss_fn(model)
end

# Wrapper for the high-level loop to collect and train
function run_training_iteration(mcts, optimizer, model, n_games=5, sims=100, rng=Random.default_rng())
    all_states = GameState[]
    all_pi = Vector{Vector{Float32}}()
    all_v = Float32[]
    
    for g in 1:n_games
        print("  Game $g/$n_games... ")
        game_data = collect_selfplay_data(mcts, GameConfig(), sims, rng)
        for (s, pi, v) in game_data
            push!(all_states, s)
            push!(all_pi, pi)
            push!(all_v, v)
        end
        println(" Done.")
    end
    
    loss = train_step(model, optimizer, all_states, all_pi, all_v)
    return loss
end

# Added for completeness if needed by other parts of the system
function play_game(mcts, config=GameConfig(), sims=100, rng=Random.default_rng())
    return collect_selfplay_data(mcts, config, sims, rng)
end

end # module
