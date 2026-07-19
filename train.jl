using Flux
const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))
using .Awale
using .Awale.Training: run_training_iteration
using .Awale.Model: save_model, load_model, atomic_write
using .Awale.Evaluation: HeuristicAgent, RandomAgent, ModelAgent, evaluate_agents, evaluate_agents_on_openings, generate_opening_suite
using .Awale.MCTS: MCTSSearch
using .Awale.ReplayBuffers: ReplayBuffer
using .Awale.Publication: release_summary_path, release_id_slug, release_timestamp, runtime_config_snapshot_path, model_config_snapshot_path, write_release_summary
using .Awale.Utils: architecture_slug, architecture_scoped_path, architecture_scoped_candidates, first_existing_path
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
MODEL_CONFIG_PATH = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))
EVAL_GAMES = Int(eval_cfg["eval_games"])
SIMS_PER_EVAL = Int(eval_cfg["sims_per_eval"])
LAST_CHECKPOINT_PATH = architecture_scoped_path(CHECKPOINT_DIR, Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"]), String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin")
BEST_CHECKPOINT_PATH = architecture_scoped_path(CHECKPOINT_DIR, Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"]), String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin")
STATE_PATH = architecture_scoped_path(CHECKPOINT_DIR, Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"]), String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml"))), "training_state.toml")
CHECKPOINT_PATH = architecture_scoped_path(CHECKPOINT_DIR, Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"]), String(eval_cfg["checkpoint_path"]), "model_final.bin")
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
const INITIAL_MODEL_SEED = Int(training_cfg["initial_model_seed"])
const BOOTSTRAP_RNG_SEED = Int(training_cfg["bootstrap_rng_seed"])
const MAX_TURNS = Int(training_cfg["max_turns"])
const TRAINING_STATE_RESUME_CONTRACT = "weights-only"

function model_architecture_name()
    return Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"])
end

function checkpoint_namespace_dir()
    return joinpath(CHECKPOINT_DIR, architecture_slug(model_architecture_name()))
end

function checkpoint_write_path(configured_path::AbstractString, default_filename::AbstractString)
    return architecture_scoped_path(CHECKPOINT_DIR, model_architecture_name(), configured_path, default_filename)
end

function checkpoint_candidates(configured_path::AbstractString, default_filename::AbstractString)
    return architecture_scoped_candidates(CHECKPOINT_DIR, model_architecture_name(), configured_path, default_filename)
end

function checkpoint_existing_path(configured_path::AbstractString, default_filename::AbstractString)
    return first_existing_path(checkpoint_candidates(configured_path, default_filename))
end

function training_log_dir()
    return joinpath(checkpoint_namespace_dir(), "log")
end

function training_log_file_path()
    timestamp = Dates.format(Dates.now(), "yyyy_mm_dd_HH_mm")
    architecture = architecture_slug(model_architecture_name())
    return joinpath(training_log_dir(), "training_config_$(architecture)_$timestamp.toml")
end

function current_commit_sha()
    return readchomp(`git -C $ROOT_DIR rev-parse HEAD`)
end

function training_snapshot_path(iter::Int)
    return joinpath(checkpoint_namespace_dir(), "model_iter_$(iter).bin")
end

function training_last_checkpoint_path()
    return checkpoint_write_path(String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin")
end

function training_last_checkpoint_existing_path()
    return checkpoint_existing_path(String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin")
end

function training_best_checkpoint_path()
    return checkpoint_write_path(String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin")
end

function training_best_checkpoint_existing_path()
    return checkpoint_existing_path(String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin")
end

function training_state_file_path()
    return checkpoint_write_path(String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml"))), "training_state.toml")
end

function training_state_existing_path()
    return checkpoint_existing_path(String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml"))), "training_state.toml")
end

function evaluation_checkpoint_path()
    return checkpoint_write_path(String(get(eval_cfg, "checkpoint_path", joinpath(CHECKPOINT_DIR, "model_final.bin"))), "model_final.bin")
end

function evaluation_checkpoint_existing_path()
    return checkpoint_existing_path(String(get(eval_cfg, "checkpoint_path", joinpath(CHECKPOINT_DIR, "model_final.bin"))), "model_final.bin")
end

function write_training_state(path::String, last_iter::Int, best_selection_score::Float64)
    atomic_write(path) do io
        println(io, "resume_contract = \"$(TRAINING_STATE_RESUME_CONTRACT)\"")
        println(io, "last_iter = $last_iter")
        println(io, "best_selection_score = $best_selection_score")
    end
end

function read_training_state(path::String)
    if !isfile(path)
        return 0, -1.0, TRAINING_STATE_RESUME_CONTRACT
    end

    state = TOML.parsefile(path)
    last_iter = Int(get(state, "last_iter", 0))
    best_selection_score = Float64(get(state, "best_selection_score", get(state, "best_win_rate", -1.0)))
    resume_contract = String(get(state, "resume_contract", TRAINING_STATE_RESUME_CONTRACT))
    return last_iter, best_selection_score, resume_contract
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

function create_initial_model()
    println("Inicializando modelo base con seed fija: $INITIAL_MODEL_SEED")
    Random.seed!(INITIAL_MODEL_SEED)
    return Awale.create_model(MODEL_CONFIG_PATH)
end

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
    best_checkpoint_path = training_best_checkpoint_existing_path()

    if best_checkpoint_path !== nothing
        best_model = load_model(best_checkpoint_path)
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
    save_model(model, training_best_checkpoint_path())
    return true
end

function write_release_summary_file(
    release_summary_file::AbstractString;
    commit_sha::AbstractString,
    architecture::AbstractString,
    release_id::AbstractString,
    checkpoint_dir::AbstractString,
    runtime_config_snapshot::AbstractString,
    model_config_snapshot::AbstractString,
    training_state_path::AbstractString,
    last_checkpoint_path::AbstractString,
    best_checkpoint_path::AbstractString,
    final_checkpoint_path::AbstractString,
    last_iter::Int,
    best_selection_score::Real,
    baseline_win_rate::Real,
    final_loss::Real,
    selection_current_best_rate::Union{Nothing, Real}=nothing,
    selection_promoted::Union{Nothing, Bool}=nothing,
)
    write_release_summary(
        release_summary_file;
        commit_sha=commit_sha,
        architecture=architecture,
        release_id=release_id,
        timestamp=release_timestamp(),
        checkpoint_dir=checkpoint_dir,
        runtime_config_snapshot=runtime_config_snapshot,
        model_config_snapshot=model_config_snapshot,
        training_state_path=training_state_path,
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        last_iter=last_iter,
        best_selection_score=best_selection_score,
        baseline_win_rate=baseline_win_rate,
        final_loss=final_loss,
        selection_current_best_rate=selection_current_best_rate,
        selection_promoted=selection_promoted,
    )
    println(" 📄 Release summary guardada en: $release_summary_file")
    return release_summary_file
end

function snapshot_run_configs(log_dir::String, architecture::AbstractString, release_id::AbstractString)
    arch = architecture_slug(architecture)
    runtime_config_path = runtime_config_snapshot_path(log_dir, arch, release_id)
    model_config_path = model_config_snapshot_path(log_dir, arch, release_id)
    model_config_source = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))

    println("Registrando configuración para arquitectura $arch en: $runtime_config_path")
    println("Registrando configuración de modelo para arquitectura $arch en: $model_config_path")

    runtime_data = read(joinpath(ROOT_DIR, "config.toml"), String)
    runtime_header = "# training_architecture = $arch\n# checkpoint_namespace = $(checkpoint_namespace_dir())\n"
    write(runtime_config_path, runtime_header * runtime_data)
    write(model_config_path, read(model_config_source, String))

    return runtime_config_path, model_config_path
end

function maybe_resume_from_legacy_checkpoint!(model_ref, start_iter_ref)
    found_iters = Int[]

    for dir in (checkpoint_namespace_dir(), CHECKPOINT_DIR)
        isdir(dir) || continue
        for file in readdir(dir)
            match_result = match(r"model_iter_(\d+)\.bin", file)
            if match_result !== nothing
                push!(found_iters, parse(Int, match_result.captures[1]))
            end
        end
    end

    if isempty(found_iters)
        return
    end

    last_iter = maximum(found_iters)
    if last_iter < NUM_ITERATIONS
        start_iter_ref[] = last_iter + 1
        namespaced_checkpoint = training_snapshot_path(last_iter)
        legacy_checkpoint = joinpath(CHECKPOINT_DIR, "model_iter_$last_iter.bin")
        checkpoint_file = isfile(namespaced_checkpoint) ? namespaced_checkpoint : legacy_checkpoint
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

    checkpoint_root = checkpoint_namespace_dir()
    mkpath(checkpoint_root)
    mkpath(training_log_dir())

    println("Arquitectura activa: $(architecture_slug(model_architecture_name()))")
    println("Checkpoint namespace: $checkpoint_root")

    release_id = release_id_slug()
    commit_sha = current_commit_sha()
    runtime_config_snapshot, model_config_snapshot = snapshot_run_configs(training_log_dir(), model_architecture_name(), release_id)
    release_summary_file = release_summary_path(CHECKPOINT_DIR, model_architecture_name())

    rng = Random.MersenneTwister(BOOTSTRAP_RNG_SEED)
println("Bootstrap RNG seed: $BOOTSTRAP_RNG_SEED | max_turns: $MAX_TURNS")

    start_iter = Ref(1)
    best_selection_score = Ref(-1.0)
    model = Ref(create_initial_model())
    last_loss = Ref(Float64(NaN))
    last_baseline_win_rate = Ref(Float64(NaN))
    last_selection_current_best_rate = Ref{Union{Nothing, Float64}}(nothing)
    last_selection_promoted = Ref{Union{Nothing, Bool}}(nothing)
    last_completed_iter = Ref(0)

    last_checkpoint_path = training_last_checkpoint_existing_path()
    training_state_path = training_state_existing_path()
    checkpoint_path = evaluation_checkpoint_existing_path()

    if "--reset" in args
        println("⚠️ [RESTART] Modo reinicio activado. Ignorando checkpoints.")
    elseif last_checkpoint_path !== nothing && training_state_path !== nothing
        last_iter, saved_best_selection_score, resume_contract = read_training_state(training_state_path)
        best_selection_score[] = saved_best_selection_score

        if last_iter < NUM_ITERATIONS
            start_iter[] = last_iter + 1
            println("¡Checkpoint detectado! Reanudando desde la iteración $last_iter...")
            println("Contrato de reanudación: $resume_contract (solo pesos; optimizer/replay/RNG no se persisten).")
            println("Cargando modelo: $last_checkpoint_path")
            model[] = load_model(last_checkpoint_path)
        else
            println("El entrenamiento alcanzó la iteración máxima ($last_iter).")
            if checkpoint_path !== nothing
                model[] = load_model(checkpoint_path)
            else
                model[] = load_model(last_checkpoint_path)
            end
            start_iter[] = NUM_ITERATIONS + 1
            last_completed_iter[] = last_iter
        end
    elseif checkpoint_path !== nothing
        println("¡Modelo final detectado! El entrenamiento ya fue completado.")
        model[] = load_model(checkpoint_path)
        last_completed_iter[] = NUM_ITERATIONS
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
                max_turns=MAX_TURNS,
            )
            println("  Loss promedio: $(round(loss, digits=4))")
            println("  Replay buffer: $(length(replay_buffer)) muestras")
            last_loss[] = Float64(loss)

            agent_model = ModelAgent(evaluation_mcts, SIMS_PER_EVAL)
            results = evaluate_agents(agent_model, agent_random, EVAL_GAMES, Awale.GameConfig(), rng)
            win_rate = (results.wins / EVAL_GAMES) * 100
            println("  Baseline vs Random @ $(SIMS_PER_EVAL) sims: $(round(win_rate, digits=2))% (W:$(results.wins) L:$(results.losses) D:$(results.draws))")
            last_baseline_win_rate[] = Float64(win_rate)

            save_model(model[], training_last_checkpoint_path())

            selection = evaluate_best_promotion(model[])
            last_selection_current_best_rate[] = selection.current_best_rate === nothing ? nothing : Float64(selection.current_best_rate)
            last_selection_promoted[] = selection.promoted
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
                println("  ✅ Nuevo mejor modelo guardado en: $(training_best_checkpoint_path())")
            else
                println("  ↳ Best no promovido: falló $(join(selection.gate_reasons, ", ")).")
            end

            if should_save_snapshot(iter, NUM_ITERATIONS, CHECKPOINT_EVERY)
                snapshot_path = training_snapshot_path(iter)
                save_model(model[], snapshot_path)
                println("  📦 Snapshot guardado en: $snapshot_path")
            end

            write_training_state(training_state_file_path(), iter, best_selection_score[])
            last_completed_iter[] = iter
        end

        save_model(model[], evaluation_checkpoint_path())
        println("\n--- Entrenamiento Finalizado ---")
        println(" Modelo final guardado en: $(evaluation_checkpoint_path())")

    else
        println("--- Entrenamiento ya completado. ---")
    end

    if start_iter[] > NUM_ITERATIONS && last_completed_iter[] == 0
        last_completed_iter[] = NUM_ITERATIONS
    end

    write_release_summary_file(
        release_summary_file;
        commit_sha=commit_sha,
        architecture=model_architecture_name(),
        release_id=release_id,
        checkpoint_dir=checkpoint_namespace_dir(),
        runtime_config_snapshot=runtime_config_snapshot,
        model_config_snapshot=model_config_snapshot,
        training_state_path=training_state_file_path(),
        last_checkpoint_path=training_last_checkpoint_path(),
        best_checkpoint_path=training_best_checkpoint_path(),
        final_checkpoint_path=evaluation_checkpoint_path(),
        last_iter=last_completed_iter[],
        best_selection_score=best_selection_score[],
        baseline_win_rate=last_baseline_win_rate[],
        final_loss=last_loss[],
        selection_current_best_rate=last_selection_current_best_rate[],
        selection_promoted=last_selection_promoted[],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
