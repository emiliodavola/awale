using Flux
include("src/Awale.jl")
using .Awale
using .Awale.Training: run_training_iteration
using .Awale.Model: save_model, load_model
using .Awale.Evaluation: RandomAgent, ModelAgent, evaluate_agents
using .Awale.MCTS: MCTSSearch
using .Awale.ReplayBuffers: ReplayBuffer
using Random
using TOML
using Dates

config = TOML.parsefile("config.toml")
training_cfg = config["training"]
eval_cfg = config["evaluation"]
mcts_cfg = config["mcts"]

NUM_ITERATIONS = Int(training_cfg["num_iterations"])
GAMES_PER_ITERATION = Int(training_cfg["games_per_iteration"])
SIMS_PER_MOVE = Int(training_cfg["sims_per_move"])
LEARNING_RATE = Float32(training_cfg["learning_rate"])
CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
REPLAY_BUFFER_CAPACITY = Int(get(training_cfg, "replay_buffer_capacity", 50_000))
BATCH_SIZE = Int(get(training_cfg, "batch_size", 128))
UPDATES_PER_ITERATION = Int(get(training_cfg, "updates_per_iteration", 64))
TEMPERATURE_MOVES = Int(get(training_cfg, "temperature_moves", 20))
CHECKPOINT_EVERY = Int(get(training_cfg, "checkpoint_every", 25))
MILESTONE_ITERATIONS = sort(unique(Int.(get(training_cfg, "milestone_iterations", Int[]))))
LAST_CHECKPOINT_PATH = String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin")))
BEST_CHECKPOINT_PATH = String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin")))
STATE_PATH = String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml")))
MODEL_CONFIG_PATH = String(get(training_cfg, "model_config_path", "src/Awale/config.toml"))
EVAL_GAMES = Int(eval_cfg["eval_games"])
SIMS_PER_EVAL = Int(eval_cfg["sims_per_eval"])
CHECKPOINT_PATH = String(eval_cfg["checkpoint_path"])
C_PUCT = Float32(mcts_cfg["c_puct"])

function write_training_state(path::String, last_iter::Int, best_win_rate::Float64)
    open(path, "w") do io
        println(io, "last_iter = $last_iter")
        println(io, "best_win_rate = $best_win_rate")
    end
end

function read_training_state(path::String)
    if !isfile(path)
        return 0, -1.0
    end

    state = TOML.parsefile(path)
    last_iter = Int(get(state, "last_iter", 0))
    best_win_rate = Float64(get(state, "best_win_rate", -1.0))
    return last_iter, best_win_rate
end

function save_run_config(log_dir::String)
    timestamp = Dates.format(Dates.now(), "yyyy_mm_dd_HH_mm")
    log_file = joinpath(log_dir, "training_config_$timestamp.toml")
    println("Registrando configuración en: $log_file")

    try
        data = read("config.toml", String)
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

should_save_snapshot(iter::Int, checkpoint_every::Int, milestone_iterations::Vector{Int}) =
    (checkpoint_every > 0 && iter % checkpoint_every == 0) || (iter in milestone_iterations)

function main(args::Vector{String}=Base.ARGS)
    println("--- Iniciando Entrenamiento y Evaluación de Awale ---")

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
    best_win_rate = Ref(-1.0)
    model = Ref(Awale.create_model(MODEL_CONFIG_PATH))

    if "--reset" in args
        println("⚠️ [RESTART] Modo reinicio activado. Ignorando checkpoints.")
    elseif isfile(LAST_CHECKPOINT_PATH) && isfile(STATE_PATH)
        last_iter, saved_best_win_rate = read_training_state(STATE_PATH)
        best_win_rate[] = saved_best_win_rate

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
                temperature_moves=TEMPERATURE_MOVES,
                rng=rng,
            )
            println("  Loss promedio: $(round(loss, digits=4))")
            println("  Replay buffer: $(length(replay_buffer)) muestras")

            agent_model = ModelAgent(evaluation_mcts, SIMS_PER_EVAL)
            results = evaluate_agents(agent_model, agent_random, EVAL_GAMES, Awale.GameConfig(), rng)
            win_rate = (results.wins / EVAL_GAMES) * 100
            println("  WinRate vs Random: $(round(win_rate, digits=2))% (W:$(results.wins) L:$(results.losses) D:$(results.draws))")

            save_model(model[], LAST_CHECKPOINT_PATH)

            if win_rate > best_win_rate[]
                best_win_rate[] = win_rate
                save_model(model[], BEST_CHECKPOINT_PATH)
                println("  ✅ Nuevo mejor modelo guardado en: $BEST_CHECKPOINT_PATH")
            end

            if should_save_snapshot(iter, CHECKPOINT_EVERY, MILESTONE_ITERATIONS)
                snapshot_path = joinpath(CHECKPOINT_DIR, "model_iter_$iter.bin")
                save_model(model[], snapshot_path)
                println("  📦 Snapshot guardado en: $snapshot_path")
            end

            write_training_state(STATE_PATH, iter, best_win_rate[])
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
