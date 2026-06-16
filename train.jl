using Flux
using Awale
using Awale.Training: run_training_iteration
using Awale.Model: save_model, load_model
using Awale.Evaluation: RandomAgent, ModelAgent, evaluate_agents
using Awale.MCTS: MCTSSearch
using Random
using JSON

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

function main()
    println("--- Iniciando Entrenamiento y Evaluación de Awale ---")
    
    if !isdir(CHECKPOINT_DIR)
        mkdir(CHECKPOINT_DIR)
    end

    rng = Random.MersenneTwister(42)
    model = Awale.create_model()
    opt = Flux.setup(Flux.Adam(LEARNING_RATE), model)
    mcts = MCTSSearch(model, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}())
    
    agent_random = RandomAgent()

    for iter in 1:NUM_ITERATIONS
        println("\nIteración $iter / $NUM_ITERATIONS")
        
        # 1. Fase de Entrenamiento
        loss = run_training_iteration(mcts, opt, model, GAMES_PER_ITERATION, SIMS_PER_MOVE, rng)
        println("  Loss: $(round(loss, digits=4))")
        
        # 2. Fase de Evaluación (El "Termómetro")
        agent_model = ModelAgent(mcts, SIMS_PER_EVAL) # Más simulaciones para evaluar
        results = evaluate_agents(agent_model, agent_random, EVAL_GAMES)
        win_rate = (results.wins / EVAL_GAMES) * 100
        println("  WinRate vs Random: $(round(win_rate, digits=2))% (W:$(results.wins) L:$(results.losses) D:$(results.draws))")
        
        # Guardar checkpoint
        save_path = joinpath(CHECKPOINT_DIR, "model_iter_$iter.bin")
        save_model(model, save_path)
    end
    
    final_path = joinpath(CHECKPOINT_DIR, "model_final.bin")
    save_model(model, final_path)
    println("\n--- Entrenamiento Finalizado ---")
    println(" Modelo final guardado en: $final_path")
end

main()
