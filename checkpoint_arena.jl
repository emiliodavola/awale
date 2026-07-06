const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))
using .Awale
using .Awale.Evaluation: ModelAgent, evaluate_agents_on_openings, generate_opening_suite
using .Awale.MCTS: MCTSSearch
using .Awale.Model: load_model
using Random
using TOML

config = TOML.parsefile(joinpath(ROOT_DIR, "config.toml"))
training_cfg = config["training"]
selection_cfg = get(config, "selection", Dict{String, Any}())
mcts_cfg = config["mcts"]

CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
C_PUCT = Float32(mcts_cfg["c_puct"])
DEFAULT_GAMES = Int(get(selection_cfg, "promotion_games", 200))
DEFAULT_SIMS = [0, 50, 200]
DEFAULT_OPENING_PLIES = Int[get(selection_cfg, "opening_plies", [0, 2, 4, 6, 8, 10])...]
OPENINGS_PER_PLY = Int(get(selection_cfg, "openings_per_ply", 6))
OPENING_SEED = Int(get(selection_cfg, "opening_seed", 20260705))

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

winner_label(label_a, label_b, results) = ifelse(results.wins > results.losses, checkpoint_label(label_a), ifelse(results.losses > results.wins, checkpoint_label(label_b), "tie"))

function winner_percentage(results)
    decided = results.wins + results.losses
    decided == 0 && return 0.0
    return max(results.wins, results.losses) / decided * 100.0
end

const TABLE_WIDTHS = (checkpoint = 14, wins = 5, losses = 5, draws = 5, avg_turns = 9, who_wins = 32)

pad_cell(value, width::Int) = rpad(string(value), width)

function format_header()
    return join([
        pad_cell("Checkpoint A", TABLE_WIDTHS.checkpoint),
        pad_cell("Checkpoint B", TABLE_WIDTHS.checkpoint),
        pad_cell("W", TABLE_WIDTHS.wins),
        pad_cell("L", TABLE_WIDTHS.losses),
        pad_cell("D", TABLE_WIDTHS.draws),
        pad_cell("AvgTurns", TABLE_WIDTHS.avg_turns),
        pad_cell("Who wins", TABLE_WIDTHS.who_wins),
    ], " | ")
end

function format_duel_result(label_a, label_b, results)
    winner = winner_label(label_a, label_b, results)
    winner_text = winner == "tie" ? "tie" : "$(winner) ($(round(winner_percentage(results), digits=1))% of decided games)"
    return join([
        pad_cell(checkpoint_label(label_a), TABLE_WIDTHS.checkpoint),
        pad_cell(checkpoint_label(label_b), TABLE_WIDTHS.checkpoint),
        pad_cell(results.wins, TABLE_WIDTHS.wins),
        pad_cell(results.losses, TABLE_WIDTHS.losses),
        pad_cell(results.draws, TABLE_WIDTHS.draws),
        pad_cell(round(results.avg_turns, digits=2), TABLE_WIDTHS.avg_turns),
        pad_cell(winner_text, TABLE_WIDTHS.who_wins),
    ], " | ")
end

stable_label_seed(label) = label isa Int ? label : sum(codeunits(String(label)))

function run_duel(label_a, label_b; sims::Int, games::Int, openings=generate_opening_suite(plies=DEFAULT_OPENING_PLIES, openings_per_ply=OPENINGS_PER_PLY, seed=OPENING_SEED))
    path_a = checkpoint_path(label_a)
    path_b = checkpoint_path(label_b)

    if !isfile(path_a) || !isfile(path_b)
        return nothing
    end

    model_a = load_model(path_a)
    model_b = load_model(path_b)
    agent_a = ModelAgent(MCTSSearch(model_a, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    agent_b = ModelAgent(MCTSSearch(model_b, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), sims)
    duel_rng = Random.MersenneTwister(OPENING_SEED + 1000 * sims + 31 * stable_label_seed(label_a) + stable_label_seed(label_b))
    return evaluate_agents_on_openings(agent_a, agent_b, openings, games, duel_rng)
end

function main()
    println("--- Awale checkpoint arena ---")
    matchups = available_matchups()

    if isempty(matchups)
        println("No hay suficientes checkpoints compatibles para correr el arena.")
        return
    end

    openings = generate_opening_suite(plies=DEFAULT_OPENING_PLIES, openings_per_ply=OPENINGS_PER_PLY, seed=OPENING_SEED)
    println("Checkpoints detectados: $(join(checkpoint_label.(existing_checkpoint_labels()), ", "))")
    println("Opening suite: $(length(openings)) posiciones reproducibles (plies=$(DEFAULT_OPENING_PLIES), openings_per_ply=$(OPENINGS_PER_PLY))")

    for sims in DEFAULT_SIMS
        println("\nSims per side: $sims")
        println(format_header())
        println(repeat("-", sum(values(TABLE_WIDTHS)) + 3 * 6))
        for (label_a, label_b) in matchups
            results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES, openings=openings)
            if results === nothing
                println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => skipped (missing checkpoint)")
            else
                println(format_duel_result(label_a, label_b, results))
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
