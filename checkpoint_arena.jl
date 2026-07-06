const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))
using .Awale
using .Awale.Evaluation: ModelAgent, result_from_terminal_state
using .Awale.MCTS: MCTSSearch
using .Awale.Model: load_model
using Random
using TOML

config = TOML.parsefile(joinpath(ROOT_DIR, "config.toml"))
training_cfg = config["training"]
mcts_cfg = config["mcts"]

CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
C_PUCT = Float32(mcts_cfg["c_puct"])
DEFAULT_GAMES = 200
DEFAULT_SIMS = [0, 50, 200]
DEFAULT_OPENING_PLIES = [0, 2, 4, 6]
OPENINGS_PER_PLY = 4
OPENING_SEED = 20260705

function checkpoint_path(label)
    if label isa Int
        return joinpath(CHECKPOINT_DIR, "model_iter_$(label).bin")
    end

    mapping = Dict(
        "last" => String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))),
        "best" => String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))),
        "final" => String(config["evaluation"]["checkpoint_path"]),
    )

    return get(mapping, String(label), joinpath(CHECKPOINT_DIR, String(label)))
end

function checkpoint_label(label)
    return label isa Int ? "iter_$(label)" : String(label)
end

function existing_checkpoint_labels()
    labels = Any[]

    if !isdir(CHECKPOINT_DIR)
        return labels
    end

    for entry in readdir(CHECKPOINT_DIR)
        match_result = match(r"model_iter_(\d+)\.bin", entry)
        if match_result !== nothing
            push!(labels, parse(Int, match_result.captures[1]))
        end
    end

    for label in ("last", "best", "final")
        if isfile(checkpoint_path(label))
            push!(labels, label)
        end
    end

    return labels
end

function available_matchups()
    numeric_labels = sort([label for label in existing_checkpoint_labels() if label isa Int])
    matchups = Tuple{Int, Int}[]

    for idx in 1:(length(numeric_labels) - 1)
        push!(matchups, (numeric_labels[idx], numeric_labels[idx + 1]))
    end

    return matchups
end

function generate_opening_suite(; plies=DEFAULT_OPENING_PLIES, openings_per_ply::Int=OPENINGS_PER_PLY, seed::Int=OPENING_SEED)
    rng = MersenneTwister(seed)
    openings = Awale.GameState[]

    for ply_count in plies
        for _ in 1:openings_per_ply
            state = Awale.initial_state()
            for _ in 1:ply_count
                actions = Awale.legal_actions(state)
                isempty(actions) && break
                action = actions[rand(rng, 1:length(actions))]
                state = Awale.transition(state, action)
                Awale.is_terminal(state) && break
            end
            push!(openings, state)
        end
    end

    return openings
end

function play_match_from_state(initial_state, agent_p1, agent_p2)
    state = initial_state
    turn = 1
    turns_played = 0
    max_turns = 1000

    while !Awale.is_terminal(state) && turns_played < max_turns
        current_agent = turn == 1 ? agent_p1 : agent_p2
        action = Awale.Evaluation.select_action(current_agent, state)
        state = Awale.transition(state, action)
        turn = turn == 1 ? 2 : 1
        turns_played += 1
    end

    return result_from_terminal_state(state), turns_played
end

function evaluate_agents_on_openings(agent1, agent2, openings, games::Int)
    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    for game_idx in 1:games
        opening = openings[mod1(game_idx, length(openings))]
        if game_idx % 2 == 0
            result, turns = play_match_from_state(opening, agent1, agent2)
        else
            result, turns = play_match_from_state(opening, agent2, agent1)
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

    return (wins=wins, losses=losses, draws=draws, avg_turns=total_turns / games)
end

function run_duel(label_a, label_b; sims::Int, games::Int, openings=generate_opening_suite())
    path_a = checkpoint_path(label_a)
    path_b = checkpoint_path(label_b)

    if !isfile(path_a) || !isfile(path_b)
        return nothing
    end

    model_a = load_model(path_a)
    model_b = load_model(path_b)
    agent_a = ModelAgent(MCTSSearch(model_a, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    agent_b = ModelAgent(MCTSSearch(model_b, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    return evaluate_agents_on_openings(agent_a, agent_b, openings, games)
end

function main()
    println("--- Awale checkpoint arena ---")
    matchups = available_matchups()

    if isempty(matchups)
        println("No hay suficientes checkpoints compatibles para correr el arena.")
        return
    end

    openings = generate_opening_suite()
    println("Checkpoints detectados: $(join(checkpoint_label.(existing_checkpoint_labels()), ", "))")
    println("Opening suite: $(length(openings)) posiciones reproducibles")

    for sims in DEFAULT_SIMS
        println("\nSims per side: $sims")
        for (label_a, label_b) in matchups
            results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES, openings=openings)
            if results === nothing
                println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => skipped (missing checkpoint)")
            else
                println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => W:$(results.wins) L:$(results.losses) D:$(results.draws) AvgTurns:$(round(results.avg_turns, digits=2))")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
