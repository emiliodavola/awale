include("src/Awale.jl")
using .Awale
using .Awale.Evaluation: ModelAgent, evaluate_agents
using .Awale.MCTS: MCTSSearch
using .Awale.Model: load_model
using TOML

config = TOML.parsefile("config.toml")
training_cfg = config["training"]
mcts_cfg = config["mcts"]

CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
C_PUCT = Float32(mcts_cfg["c_puct"])
DEFAULT_GAMES = 200
DEFAULT_SIMS = [0, 50, 200]
DEFAULT_MATCHUPS = [(1, 5), (5, 10), (10, 25), (25, 5)]

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
    existing = Set(existing_checkpoint_labels())
    matchups = Tuple[]

    for matchup in DEFAULT_MATCHUPS
        if matchup[1] in existing && matchup[2] in existing
            push!(matchups, matchup)
        end
    end

    numeric_labels = sort([label for label in existing if label isa Int])
    if isempty(matchups) && length(numeric_labels) >= 2
        for idx in 1:(length(numeric_labels) - 1)
            push!(matchups, (numeric_labels[idx], numeric_labels[idx + 1]))
        end
        if length(numeric_labels) >= 3
            push!(matchups, (numeric_labels[end], numeric_labels[1]))
        end
    end

    return matchups
end

function run_duel(label_a, label_b; sims::Int, games::Int)
    path_a = checkpoint_path(label_a)
    path_b = checkpoint_path(label_b)

    if !isfile(path_a) || !isfile(path_b)
        return nothing
    end

    model_a = load_model(path_a)
    model_b = load_model(path_b)
    agent_a = ModelAgent(MCTSSearch(model_a, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    agent_b = ModelAgent(MCTSSearch(model_b, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    return evaluate_agents(agent_a, agent_b, games)
end

function main()
    println("--- Awale checkpoint arena ---")
    matchups = available_matchups()

    if isempty(matchups)
        println("No hay suficientes checkpoints compatibles para correr el arena.")
        return
    end

    println("Checkpoints detectados: $(join(checkpoint_label.(existing_checkpoint_labels()), ", "))")

    for sims in DEFAULT_SIMS
        println("\nSims per side: $sims")
        for (label_a, label_b) in matchups
            results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES)
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
