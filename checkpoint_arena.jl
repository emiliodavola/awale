const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))
using .Awale
using .Awale.Evaluation: ModelAgent, evaluate_agents_on_openings, generate_opening_suite
using .Awale.MCTS: MCTSSearch
using .Awale.Model: load_model
using .Awale.Utils: architecture_slug, architecture_scoped_candidates, first_existing_path
using Random
using TOML

config = TOML.parsefile(joinpath(ROOT_DIR, "config.toml"))
training_cfg = config["training"]
selection_cfg = get(config, "selection", Dict{String, Any}())
mcts_cfg = config["mcts"]
arena_cfg = get(config, "arena", Dict{String, Any}())

CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
MODEL_CONFIG_PATH = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))
MAX_TURNS = Int(training_cfg["max_turns"])
C_PUCT = Float32(mcts_cfg["c_puct"])
DIRICHLET_ALPHA = Float32(get(mcts_cfg, "dirichlet_alpha", 0.3))
DIRICHLET_EPSILON = Float32(get(mcts_cfg, "dirichlet_epsilon", 0.25))
DEFAULT_GAMES = Int(get(selection_cfg, "promotion_games", 200))
DEFAULT_SIMS = Int[get(arena_cfg, "sim_candidates", [0, 50, 200])...]
DEFAULT_OPENING_PLIES = Int[get(selection_cfg, "opening_plies", [0, 2, 4, 6, 8, 10])...]
OPENINGS_PER_PLY = Int(get(selection_cfg, "openings_per_ply", 6))
OPENING_SEED = Int(get(selection_cfg, "opening_seed", 20260705))

function model_architecture_name()
    return Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"])
end

function checkpoint_architecture(architecture=nothing)
    return architecture === nothing ? model_architecture_name() : String(architecture)
end

function checkpoint_namespace_dir(architecture=nothing)
    return joinpath(CHECKPOINT_DIR, architecture_slug(checkpoint_architecture(architecture)))
end

function checkpoint_path_candidates(configured_path::AbstractString, default_filename::AbstractString; architecture=nothing)
    return architecture_scoped_candidates(CHECKPOINT_DIR, checkpoint_architecture(architecture), configured_path, default_filename)
end

function checkpoint_path_from_config(configured_path::AbstractString, default_filename::AbstractString; architecture=nothing)
    candidates = checkpoint_path_candidates(configured_path, default_filename; architecture=architecture)
    found = first_existing_path(candidates)
    return found === nothing ? first(candidates) : found
end

function checkpoint_path(label; architecture=nothing)
    arch = checkpoint_architecture(architecture)
    if label isa Int
        candidates = [joinpath(checkpoint_namespace_dir(arch), "model_iter_$(label).bin"), joinpath(CHECKPOINT_DIR, "model_iter_$(label).bin")]
        found = first_existing_path(candidates)
        return found === nothing ? first(candidates) : found
    end

    mapping = Dict(
        "last" => checkpoint_path_from_config(String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin"; architecture=arch),
        "best" => checkpoint_path_from_config(String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin"; architecture=arch),
        "final" => checkpoint_path_from_config(String(config["evaluation"]["checkpoint_path"]), "model_final.bin"; architecture=arch),
    )

    return get(mapping, String(label), joinpath(checkpoint_namespace_dir(arch), String(label)))
end

function checkpoint_label(label)
    return label isa Int ? "iter_$(label)" : String(label)
end

function existing_checkpoint_labels()
    labels = Any[]
    seen_iters = Set{Int}()

    for dir in (checkpoint_namespace_dir(), CHECKPOINT_DIR)
        isdir(dir) || continue
        for entry in readdir(dir)
            match_result = match(r"model_iter_(\d+)\.bin", entry)
            if match_result !== nothing
                iter = parse(Int, match_result.captures[1])
                if !(iter in seen_iters)
                    push!(labels, iter)
                    push!(seen_iters, iter)
                end
            end
        end
    end

    for label in ("last", "best", "final")
        if isfile(checkpoint_path(label))
            push!(labels, label)
        end
    end

    return labels
end

function numeric_checkpoint_labels(labels=existing_checkpoint_labels())
    return sort([label for label in labels if label isa Int])
end

function available_matchups(numeric_labels=numeric_checkpoint_labels())
    matchups = Tuple{Int, Int}[]

    for idx in 1:(length(numeric_labels) - 1)
        push!(matchups, (numeric_labels[idx], numeric_labels[idx + 1]))
    end

    return matchups
end

LATEST_ANCHOR_COUNT = Int(get(arena_cfg, "latest_anchor_count", 3))

function latest_anchor_matchups(numeric_labels=numeric_checkpoint_labels(), anchor_count::Int=LATEST_ANCHOR_COUNT)
    length(numeric_labels) <= 1 && return Tuple{Int, Int}[]

    latest = numeric_labels[end]
    anchor_start = max(1, length(numeric_labels) - anchor_count)
    anchors = numeric_labels[anchor_start:(end - 1)]
    return [(latest, anchor) for anchor in reverse(anchors)]
end

function operational_alias_matchups(labels=existing_checkpoint_labels(), numeric_labels=numeric_checkpoint_labels(labels))
    isempty(numeric_labels) && return Tuple{Any, Any}[]

    latest = numeric_labels[end]
    label_set = Set(labels)
    matchups = Tuple{Any, Any}[]

    if "best" in label_set
        push!(matchups, ("best", latest))
    end
    if "last" in label_set
        push!(matchups, ("last", latest))
    end
    if "final" in label_set
        push!(matchups, ("final", latest))
    end
    if "best" in label_set && "last" in label_set
        push!(matchups, ("best", "last"))
    end
    if "best" in label_set && "final" in label_set
        push!(matchups, ("best", "final"))
    end
    if "final" in label_set && "last" in label_set
        push!(matchups, ("final", "last"))
    end

    return unique(matchups)
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

function collect_duel_labels(matchups)::Vector{Any}
    labels = Any[]
    for (label_a, label_b) in matchups
        push!(labels, label_a, label_b)
    end
    return unique(labels)
end

function cache_key(label; architecture=nothing)
    return architecture === nothing ? label : (architecture_slug(String(architecture)), label)
end

function build_model_cache(labels; architecture=nothing)
    cache = Dict{Any, Any}()
    for label in labels
        path = checkpoint_path(label; architecture=architecture)
        isfile(path) || continue
        model = load_model(path)
        cache[label] = model
        cache[path] = model
        architecture === nothing || (cache[cache_key(label; architecture=architecture)] = model)
    end
    return cache
end

function resolve_model(label, path::AbstractString, model_cache; architecture=nothing)
    if model_cache !== nothing
        if architecture !== nothing
            arch_key = cache_key(label; architecture=architecture)
            haskey(model_cache, arch_key) && return model_cache[arch_key]
        end
        haskey(model_cache, path) && return model_cache[path]
        architecture === nothing && haskey(model_cache, label) && return model_cache[label]
        return nothing
    end
    return isfile(path) ? load_model(path) : nothing
end

function run_duel(label_a, label_b; sims::Int, games::Int, openings=generate_opening_suite(plies=DEFAULT_OPENING_PLIES, openings_per_ply=OPENINGS_PER_PLY, seed=OPENING_SEED), model_cache=nothing, max_turns::Int=MAX_TURNS, architecture_a=nothing, architecture_b=nothing)
    path_a = checkpoint_path(label_a; architecture=architecture_a)
    path_b = checkpoint_path(label_b; architecture=architecture_b)

    model_a = resolve_model(label_a, path_a, model_cache; architecture=architecture_a)
    model_b = resolve_model(label_b, path_b, model_cache; architecture=architecture_b)
    if model_a === nothing || model_b === nothing
        return nothing
    end

    agent_a = ModelAgent(MCTSSearch(model_a, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}()), sims)
    agent_b = ModelAgent(MCTSSearch(model_b, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}()), sims)
    duel_rng = Random.MersenneTwister(OPENING_SEED + 1000 * sims + 31 * stable_label_seed(label_a) + stable_label_seed(label_b))
    return evaluate_agents_on_openings(agent_a, agent_b, openings, games, duel_rng; max_turns=max_turns)
end

function main(; post_freeze_callback=nothing)
    println("--- Awale checkpoint arena ---")
    detected_labels = existing_checkpoint_labels()
    numeric_labels = numeric_checkpoint_labels(detected_labels)
    matchups = available_matchups(numeric_labels)
    anchor_matchups = latest_anchor_matchups(numeric_labels, LATEST_ANCHOR_COUNT)
    alias_matchups = operational_alias_matchups(detected_labels, numeric_labels)

    if isempty(matchups) && isempty(anchor_matchups) && isempty(alias_matchups)
        println("Not enough compatible checkpoints are available to run the arena.")
        return
    end

    all_matchups = vcat(matchups, anchor_matchups, alias_matchups)
    planned_labels = collect_duel_labels(all_matchups)
    model_cache = build_model_cache(planned_labels)
    post_freeze_callback === nothing || post_freeze_callback((;
        detected_labels=copy(detected_labels),
        numeric_labels=copy(numeric_labels),
        matchups=copy(matchups),
        anchor_matchups=copy(anchor_matchups),
        alias_matchups=copy(alias_matchups),
        planned_labels=copy(planned_labels),
        model_cache=model_cache,
    ))
    openings = generate_opening_suite(plies=DEFAULT_OPENING_PLIES, openings_per_ply=OPENINGS_PER_PLY, seed=OPENING_SEED)
    println("Detected checkpoints: $(join(checkpoint_label.(detected_labels), ", "))")
    println("Opening suite: $(length(openings)) reproducible positions (plies=$(DEFAULT_OPENING_PLIES), openings_per_ply=$(OPENINGS_PER_PLY))")
    println("Frozen labels for this run: $(join(sort!(string.(keys(model_cache))), ", "))")
    println("Planned labels for this run: $(join(sort!(string.(planned_labels)), ", "))")

    for sims in DEFAULT_SIMS
        println("\nSims per side: $sims")
        println(format_header())
        println(repeat("-", sum(values(TABLE_WIDTHS)) + 3 * 6))
        for (label_a, label_b) in matchups
            results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES, openings=openings, model_cache=model_cache)
            if results === nothing
                println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => skipped (missing checkpoint)")
            else
                println(format_duel_result(label_a, label_b, results))
            end
        end

        if !isempty(anchor_matchups)
            println("\nLatest checkpoint vs prior anchors (last $(LATEST_ANCHOR_COUNT))")
            println(format_header())
            println(repeat("-", sum(values(TABLE_WIDTHS)) + 3 * 6))
            for (label_a, label_b) in anchor_matchups
                results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES, openings=openings, model_cache=model_cache)
                if results === nothing
                    println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => skipped (missing checkpoint)")
                else
                    println(format_duel_result(label_a, label_b, results))
                end
            end
        end

        if !isempty(alias_matchups)
            println("\nOperational aliases")
            println(format_header())
            println(repeat("-", sum(values(TABLE_WIDTHS)) + 3 * 6))
            for (label_a, label_b) in alias_matchups
                results = run_duel(label_a, label_b; sims=sims, games=DEFAULT_GAMES, openings=openings, model_cache=model_cache)
                if results === nothing
                    println("$(checkpoint_label(label_a)) vs $(checkpoint_label(label_b)) => skipped (missing checkpoint)")
                else
                    println(format_duel_result(label_a, label_b, results))
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
