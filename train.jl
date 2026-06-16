using Flux
using Awale
using Awale.Training: run_training_iteration
using Awale.Model: save_model, load_model
using Awale.Evaluation: RandomAgent, ModelAgent, evaluate_agents
using Awale.MCTS: MCTSSearch
using Random
using JSON
using Dates

# Load configuration
config = JSON.parsefile("config.json")
training_cfg = config["training"]
eval_cfg = config["evaluation"]
mcts_cfg = config["mcts"]

# Mapping
NUM_ITERATIONS = Int(training_cfg["num_iterations"])
GAMES_PER_ITERATION = Int(training_cfg["games_per_iteration"])
SIMS_PER_MOVE = Int(training_cfg["sims_per_move"])
LEARNING_RATE = Float32(training_cfg["learning_rate"])
CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
EVAL_GAMES = Int(eval_cfg["eval_games"])
SIMS_PER_EVAL = Int(eval_cfg["sims_per_eval"])
CHECKPOINT_PATH = String(eval_cfg["checkpoint_path"])
C_PUCT = Float32(mcts_cfg["c_puct"])

function main(args::Vector{String}=Base.ARGS)
    println("--- Iniciando Entrenamiento y Evaluación de Awale ---")
    
    if !isdir(CHECKPOINT_DIR)
        mkdir(CHECKPOINT_DIR)
    end

    if !isdir(joinpath(CHECKPOINT_DIR, "log"))
        mkdir(joinpath(CHECKPOINT_DIR, "log"))
    end

    # --- Registro de Bitácora (Logbook) ---
    timestamp = Dates.format(Dates.now(), "yyyy_mm_dd_HH_mm")
    log_file = joinpath(CHECKPOINT_DIR, "log", "training_log_$timestamp.json")
    println("Registrando bitácora en: $log_file")
    
    try
        data = read("config.json")
        write(log_file, data)
    catch e
        println("Error al copiar bitácora: $e")
    end
    # --- Fin Registro de Bitácora ---

    rng = Random.MersenneTwister(42)
    
    # --- Lógica de Reanudación (Resume) ---
    start_iter = 1
    model = Awale.create_model()

    if "--reset" in args
        println("⚠️ [RESTART] Modo reinicio activado. Ignorando checkpoints.")
    elseif isdir(CHECKPOINT_DIR)
        files = readdir(CHECKPOINT_DIR)
        found_iters = Int[]
        
        # Prioridad máxima: El modelo final (si existe, es la base de la siguiente fase)
        if isfile(CHECKPOINT_PATH)
            println("¡Modelo Final detectado! Reanudando la siguiente fase...")
            model = load_model(CHECKPOINT_PATH)
            start_iter = NUM_ITERATIONS + 1
        else
            # Si no hay final, buscamos el último checkpoint intermedio
            for f in files
                m = match(r"model_iter_(\d+)\.bin", f)
                if m !== nothing
                    push!(found_iters, parse(Int, m.captures[1]))
                end
            end
            
            if !isempty(found_iters)
                last_iter = maximum(found_iters)
                if last_iter < NUM_ITERATIONS
                    start_iter = last_iter + 1
                    checkpoint_file = joinpath(CHECKPOINT_DIR, "model_iter_$last_iter.bin")
                    println("¡Checkpoint detectado! Reanudando desde la iteración $last_iter...")
                    println("Cargando modelo: $checkpoint_file")
                    model = load_model(checkpoint_file)
                else
                    println("El entrenamiento alcanzó la iteración máxima ($last_iter).")
                    start_iter = NUM_ITERATIONS + 1
                end
            end
        end
    end
    
    opt = Flux.setup(Flux.Adam(LEARNING_RATE), model)
    # --- Fin Lógica de Reanudación ---

    mcts = MCTSSearch(model, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}())
    agent_random = RandomAgent()

    if start_iter <= NUM_ITERATIONS
        for iter in start_iter:NUM_ITERATIONS
            println("\nIteración $iter / $NUM_ITERATIONS")
            
            # 1. Fase de Entrenamiento
            loss = run_training_iteration(mcts, opt, model, GAMES_PER_ITERATION, SIMS_PER_MOVE, rng)
            println("  Loss: $(round(loss, digits=4))")
            
            # 2. Fase de Evaluación (El "Termómetro")
            agent_model = ModelAgent(mcts, SIMS_PER_EVAL) 
            results = evaluate_agents(agent_model, agent_random, EVAL_GAMES)
            win_rate = (results.wins / EVAL_GAMES) * 100
            println("  WinRate vs Random: $(round(win_rate, digits=2))% (W:$(results.wins) L:$(results.losses) D:$(results.draws))")
            
            # Guardar checkpoint
            save_path = joinpath(CHECKPOINT_DIR, "model_iter_$iter.bin")
            save_model(model, save_path)
        end
        
        final_path = CHECKPOINT_PATH
        save_model(model, final_path)
        println("\n--- Entrenamiento Finalizado ---")
        println(" Modelo final guardado en: $final_path")
    else
        println("--- Entrenamiento ya completado. ---")
    end
end

main()
