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

function run_duel(label_a, label_b; sims::Int, games::Int)
    path_a = checkpoint_path(label_a)
    path_b = checkpoint_path(label_b)

    if !isfile(path_a)
        error("Missing checkpoint: $path_a")
    end
    if !isfile(path_b)
        error("Missing checkpoint: $path_b")
    end

    model_a = load_model(path_a)
    model_b = load_model(path_b)
    agent_a = ModelAgent(MCTSSearch(model_a, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    agent_b = ModelAgent(MCTSSearch(model_b, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    return evaluate_agents(agent_a, agent_b, games)
end

function main()
    println("--- Awale checkpoint arena ---")
    for sims in DEFAULT_SIMS
        println("\nSims per side: $sims")
        for (label_a, label_b) in DEFAULT_MATCHUPS
            results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES)
            println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => W:$(results.wins) L:$(results.losses) D:$(results.draws) AvgTurns:$(round(results.avg_turns, digits=2))")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
