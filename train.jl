using Flux
const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))
using .Awale
using .Awale.Training: run_training_iteration
using .Awale.Model: save_model, load_model
using .Awale.Evaluation: HeuristicAgent, RandomAgent, ModelAgent, evaluate_agents, evaluate_agents_on_openings, generate_opening_suite
using .Awale.MCTS: MCTSSearch
using .Awale.ReplayBuffers: ReplayBuffer
using Random
using TOML
using Dates

config = TOML.parsefile(joinpath(ROOT_DIR, "config.toml"))
training_cfg = config["training"]
eval_cfg = config["evaluation"]
selection_cfg = get(config, "selection", Dict{String, Any}())
mcts_cfg = config["mcts"]

NUM_ITERATIONS = Int(training_cfg["num_iterations"])
GAMES_PER_ITERATION = Int(training_cfg["games_per_iteration"])
SIMS_PER_MOVE = Int(training_cfg["sims_per_move"])
LEARNING_RATE = Float32(training_cfg["learning_rate"])
CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
REPLAY_BUFFER_CAPACITY = Int(get(training_cfg, "replay_buffer_capacity", 50_000))
BATCH_SIZE = Int(get(training_cfg, "batch_size", 128))
UPDATES_PER_ITERATION = Int(get(training_cfg, "updates_per_iteration", 16))
REPLAY_RECENT_FRACTION = Float64(get(training_cfg, "replay_recent_fraction", 0.5))
REPLAY_RECENT_WINDOW = Int(get(training_cfg, "replay_recent_window", 4096))
TEMPERATURE_MOVES = Int(get(training_cfg, "temperature_moves", 20))
CHECKPOINT_EVERY = Int(get(training_cfg, "checkpoint_every", 25))
LAST_CHECKPOINT_PATH = String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin")))
BEST_CHECKPOINT_PATH = String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin")))
STATE_PATH = String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml")))
MODEL_CONFIG_PATH = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))
EVAL_GAMES = Int(eval_cfg["eval_games"])
SIMS_PER_EVAL = Int(eval_cfg["sims_per_eval"])
CHECKPOINT_PATH = String(eval_cfg["checkpoint_path"])
BEST_TARGET_SIMS = Int(get(selection_cfg, "target_sims", SIMS_PER_EVAL))
BEST_PROMOTION_GAMES = Int(get(selection_cfg, "promotion_games", EVAL_GAMES))
BEST_PROMOTION_THRESHOLD = Float64(get(selection_cfg, "promotion_threshold", 55.0))
BEST_OPENING_PLIES = Int[get(selection_cfg, "opening_plies", [0, 2, 4, 6, 8, 10])...]
BEST_OPENINGS_PER_PLY = Int(get(selection_cfg, "openings_per_ply", 6))
BEST_OPENING_SEED = Int(get(selection_cfg, "opening_seed", 20260705))
BEST_SELECTION_SEED = Int(get(selection_cfg, "selection_seed", 20260706))
USE_RANDOM_ANCHOR = Bool(get(selection_cfg, "use_random_anchor", true))
USE_HEURISTIC_ANCHOR = Bool(get(selection_cfg, "use_heuristic_anchor", false))
ANCHOR_MIN_DECIDED_WIN_RATE = Float64(get(selection_cfg, "anchor_min_decided_win_rate", 50.0))
C_PUCT = Float32(mcts_cfg["c_puct"])

function write_training_state(path::String, last_iter::Int, best_selection_score::Float64)
    open(path, "w") do io
        println(io, "last_iter = $last_iter")
        println(io, "best_selection_score = $best_selection_score")
    end
end

function read_training_state(path::String)
    if !isfile(path)
        return 0, -1.0
    end

    state = TOML.parsefile(path)
    last_iter = Int(get(state, "last_iter", 0))
    best_selection_score = Float64(get(state, "best_selection_score", get(state, "best_win_rate", -1.0)))
    return last_iter, best_selection_score
end

function decided_win_rate(results)::Float64
    decided = results.wins + results.losses
    return decided == 0 ? 50.0 : (results.wins / decided) * 100.0
end

function validate_training_config()
    UPDATES_PER_ITERATION > 0 || throw(ArgumentError("training.updates_per_iteration must be > 0"))
    0.0 <= REPLAY_RECENT_FRACTION <= 1.0 || throw(ArgumentError("training.replay_recent_fraction must be between 0 and 1"))
    REPLAY_RECENT_WINDOW >= 0 || throw(ArgumentError("training.replay_recent_window must be >= 0"))
    REPLAY_RECENT_FRACTION == 0.0 || REPLAY_RECENT_WINDOW > 0 || throw(ArgumentError("training.replay_recent_window must be > 0 when replay_recent_fraction > 0"))
    return nothing
end

function validate_selection_config(games::Int, opening_plies, openings_per_ply::Int)
    BEST_TARGET_SIMS >= 0 || throw(ArgumentError("selection.target_sims must be >= 0"))
    games > 0 || throw(ArgumentError("selection.promotion_games must be > 0"))
    0.0 <= BEST_PROMOTION_THRESHOLD <= 100.0 || throw(ArgumentError("selection.promotion_threshold must be between 0 and 100"))
    0.0 <= ANCHOR_MIN_DECIDED_WIN_RATE <= 100.0 || throw(ArgumentError("selection.anchor_min_decided_win_rate must be between 0 and 100"))
    !isempty(opening_plies) || throw(ArgumentError("selection.opening_plies must not be empty"))
    openings_per_ply > 0 || throw(ArgumentError("selection.openings_per_ply must be > 0"))
end

function build_selection_openings()
    validate_selection_config(BEST_PROMOTION_GAMES, BEST_OPENING_PLIES, BEST_OPENINGS_PER_PLY)
    return generate_opening_suite(
        plies=BEST_OPENING_PLIES,
        openings_per_ply=BEST_OPENINGS_PER_PLY,
        seed=BEST_OPENING_SEED,
    )
end

selection_rng(offset::Int) = Random.MersenneTwister(BEST_SELECTION_SEED + offset)

function selection_gate_status(current_best_rate, anchor_reports)
    passes_best = current_best_rate === nothing || current_best_rate >= BEST_PROMOTION_THRESHOLD
    passes_anchors = all(report.decided_win_rate >= ANCHOR_MIN_DECIDED_WIN_RATE for report in anchor_reports)
    promoted = passes_best && passes_anchors
    reasons = String[]
    !passes_best && push!(reasons, "current-best threshold")
    !passes_anchors && push!(reasons, "anchor floor")
    return (passes_best=passes_best, passes_anchors=passes_anchors, promoted=promoted, reasons=reasons)
end

function evaluate_best_promotion(candidate_model)
    selection_openings = build_selection_openings()
    candidate_agent = ModelAgent(MCTSSearch(candidate_model, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), BEST_TARGET_SIMS)
    current_best_results = nothing
    current_best_rate = nothing

    if isfile(BEST_CHECKPOINT_PATH)
        best_model = load_model(BEST_CHECKPOINT_PATH)
        best_agent = ModelAgent(MCTSSearch(best_model, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}()), BEST_TARGET_SIMS)
        current_best_results = evaluate_agents_on_openings(candidate_agent, best_agent, selection_openings, BEST_PROMOTION_GAMES, selection_rng(1))
        current_best_rate = decided_win_rate(current_best_results)
    end

    anchor_reports = NamedTuple[]
    if USE_RANDOM_ANCHOR
        random_results = evaluate_agents_on_openings(candidate_agent, RandomAgent(), selection_openings, BEST_PROMOTION_GAMES, selection_rng(2))
        push!(anchor_reports, (name="random", results=random_results, decided_win_rate=decided_win_rate(random_results)))
    end
    if USE_HEURISTIC_ANCHOR
        heuristic_results = evaluate_agents_on_openings(candidate_agent, HeuristicAgent(), selection_openings, BEST_PROMOTION_GAMES, selection_rng(3))
        push!(anchor_reports, (name="heuristic", results=heuristic_results, decided_win_rate=decided_win_rate(heuristic_results)))
    end

    gates = selection_gate_status(current_best_rate, anchor_reports)
    promotion_score = current_best_rate === nothing ? (isempty(anchor_reports) ? 100.0 : minimum(report.decided_win_rate for report in anchor_reports)) : current_best_rate

    return (
        promoted=gates.promoted,
        promotion_score=promotion_score,
        current_best_results=current_best_results,
        current_best_rate=current_best_rate,
        anchor_reports=anchor_reports,
        openings=length(selection_openings),
        gate_reasons=gates.reasons,
    )
end

function maybe_promote_best!(model, best_selection_score_ref, selection)
    if !selection.promoted
        return false
    end

    best_selection_score_ref[] = selection.promotion_score
    save_model(model, BEST_CHECKPOINT_PATH)
    return true
end

function save_run_config(log_dir::String)
    timestamp = Dates.format(Dates.now(), "yyyy_mm_dd_HH_mm")
    log_file = joinpath(log_dir, "training_config_$timestamp.toml")
    println("Registrando configuración en: $log_file")

    try
        data = read(joinpath(ROOT_DIR, "config.toml"), String)
        write(log_file, data)
    catch err
        println("Error al copiar configuración: $err")
    end
end

function maybe_resume_from_legacy_checkpoint!(model_ref, start_iter_ref)
    files = readdir(CHECKPOINT_DIR)
    found_iters = Int[]

    for file in files
        match_result = match(r"model_iter_(\d+)\.bin", file)
        if match_result !== nothing
            push!(found_iters, parse(Int, match_result.captures[1]))
        end
    end

    if isempty(found_iters)
        return
    end

    last_iter = maximum(found_iters)
    if last_iter < NUM_ITERATIONS
        start_iter_ref[] = last_iter + 1
        checkpoint_file = joinpath(CHECKPOINT_DIR, "model_iter_$last_iter.bin")
        println("¡Checkpoint legacy detectado! Reanudando desde la iteración $last_iter...")
        println("Cargando modelo: $checkpoint_file")
        model_ref[] = load_model(checkpoint_file)
    else
        println("El entrenamiento legacy alcanzó la iteración máxima ($last_iter).")
        start_iter_ref[] = NUM_ITERATIONS + 1
    end
end

is_power_of_two(value::Int) = value > 0 && (value & (value - 1)) == 0

function should_save_snapshot(iter::Int, num_iterations::Int, checkpoint_every::Int)
    return iter == 1 ||
        is_power_of_two(iter) ||
        (checkpoint_every > 0 && iter % checkpoint_every == 0)
end

function main(args::Vector{String}=Base.ARGS)
    println("--- Iniciando Entrenamiento y Evaluación de Awale ---")

    validate_training_config()
    validate_selection_config(BEST_PROMOTION_GAMES, BEST_OPENING_PLIES, BEST_OPENINGS_PER_PLY)

    if !isdir(CHECKPOINT_DIR)
        mkdir(CHECKPOINT_DIR)
    end

    log_dir = joinpath(CHECKPOINT_DIR, "log")
    if !isdir(log_dir)
        mkdir(log_dir)
    end

    save_run_config(log_dir)

    rng = Random.MersenneTwister(42)

    start_iter = Ref(1)
    best_selection_score = Ref(-1.0)
    model = Ref(Awale.create_model(MODEL_CONFIG_PATH))

    if "--reset" in args
        println("⚠️ [RESTART] Modo reinicio activado. Ignorando checkpoints.")
    elseif isfile(LAST_CHECKPOINT_PATH) && isfile(STATE_PATH)
        last_iter, saved_best_selection_score = read_training_state(STATE_PATH)
        best_selection_score[] = saved_best_selection_score

        if last_iter < NUM_ITERATIONS
            start_iter[] = last_iter + 1
            println("¡Checkpoint detectado! Reanudando desde la iteración $last_iter...")
            println("Cargando modelo: $LAST_CHECKPOINT_PATH")
            model[] = load_model(LAST_CHECKPOINT_PATH)
        else
            println("El entrenamiento alcanzó la iteración máxima ($last_iter).")
            if isfile(CHECKPOINT_PATH)
                model[] = load_model(CHECKPOINT_PATH)
            else
                model[] = load_model(LAST_CHECKPOINT_PATH)
            end
            start_iter[] = NUM_ITERATIONS + 1
        end
    elseif isfile(CHECKPOINT_PATH)
        println("¡Modelo final detectado! El entrenamiento ya fue completado.")
        model[] = load_model(CHECKPOINT_PATH)
        start_iter[] = NUM_ITERATIONS + 1
    else
        maybe_resume_from_legacy_checkpoint!(model, start_iter)
    end

    optimizer = Flux.setup(Flux.Adam(LEARNING_RATE), model[])
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    training_mcts = MCTSSearch(model[], C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}())
    evaluation_mcts = MCTSSearch(model[], C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}())
    agent_random = RandomAgent()

    if start_iter[] <= NUM_ITERATIONS
        for iter in start_iter[]:NUM_ITERATIONS
            println("\nIteración $iter / $NUM_ITERATIONS")

            loss = run_training_iteration(
                training_mcts,
                optimizer,
                model[],
                replay_buffer;
                n_games=GAMES_PER_ITERATION,
                sims=SIMS_PER_MOVE,
                batch_size=BATCH_SIZE,
                updates_per_iteration=UPDATES_PER_ITERATION,
                replay_recent_fraction=REPLAY_RECENT_FRACTION,
                replay_recent_window=REPLAY_RECENT_WINDOW,
                temperature_moves=TEMPERATURE_MOVES,
                rng=rng,
            )
            println("  Loss promedio: $(round(loss, digits=4))")
            println("  Replay buffer: $(length(replay_buffer)) muestras")

            agent_model = ModelAgent(evaluation_mcts, SIMS_PER_EVAL)
            results = evaluate_agents(agent_model, agent_random, EVAL_GAMES, Awale.GameConfig(), rng)
            win_rate = (results.wins / EVAL_GAMES) * 100
            println("  Baseline vs Random @ $(SIMS_PER_EVAL) sims: $(round(win_rate, digits=2))% (W:$(results.wins) L:$(results.losses) D:$(results.draws))")

            save_model(model[], LAST_CHECKPOINT_PATH)

            selection = evaluate_best_promotion(model[])
            println("  Best-selection target: $(BEST_TARGET_SIMS) sims, $(BEST_PROMOTION_GAMES) games, $(selection.openings) openings, threshold $(BEST_PROMOTION_THRESHOLD)%")
            if selection.current_best_results === nothing
                println("  Candidate vs current best: bootstrap (no current best checkpoint)")
            else
                best_results = selection.current_best_results
                println("  Candidate vs current best: $(round(selection.current_best_rate, digits=2))% decided wins (W:$(best_results.wins) L:$(best_results.losses) D:$(best_results.draws))")
            end
            for report in selection.anchor_reports
                anchor_results = report.results
                println("  Candidate vs $(report.name) anchor: $(round(report.decided_win_rate, digits=2))% decided wins (W:$(anchor_results.wins) L:$(anchor_results.losses) D:$(anchor_results.draws))")
            end

            if maybe_promote_best!(model[], best_selection_score, selection)
                println("  ✅ Nuevo mejor modelo guardado en: $BEST_CHECKPOINT_PATH")
            else
                println("  ↳ Best no promovido: falló $(join(selection.gate_reasons, ", ")).")
            end

            if should_save_snapshot(iter, NUM_ITERATIONS, CHECKPOINT_EVERY)
                snapshot_path = joinpath(CHECKPOINT_DIR, "model_iter_$iter.bin")
                save_model(model[], snapshot_path)
                println("  📦 Snapshot guardado en: $snapshot_path")
            end

            write_training_state(STATE_PATH, iter, best_selection_score[])
        end

        save_model(model[], CHECKPOINT_PATH)
        println("\n--- Entrenamiento Finalizado ---")
        println(" Modelo final guardado en: $CHECKPOINT_PATH")
    else
        println("--- Entrenamiento ya completado. ---")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
