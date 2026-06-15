using Flux
using Awale
using Awale.Training: run_training_iteration
using Awale.Model: save_model, load_model
using Awale.Evaluation: RandomAgent, ModelAgent, evaluate_agents
using Awale.MCTS: MCTSSearch
using Random

# Configuración
NUM_ITERATIONS = 10
GAMES_PER_ITERATION = 10
SIMS_PER_MOVE = 20
LEARNING_RATE = 0.001f0
CHECKPOINT_DIR = "checkpoints"
EVAL_GAMES = 50 # Juegos para evaluar el progreso contra Random

function main()
    println("--- Iniciando Entrenamiento y Evaluación de Awale ---")
    
    if !isdir(CHECKPOINT_DIR)
        mkdir(CHECKPOINT_DIR)
    end

    rng = Random.MersenneTwister(42)
    model = Awale.create_model()
    opt = Flux.setup(Flux.Adam(LEARNING_RATE), model)
    mcts = Awale.MCTSSearch(model, 1.4f0)
    
    agent_random = RandomAgent()

    for iter in 1:NUM_ITERATIONS
        println("\nIteración $iter / $NUM_ITERATIONS")
        
        # 1. Fase de Entrenamiento
        loss = run_training_iteration(mcts, opt, model, GAMES_PER_ITERATION, SIMS_PER_MOVE, rng)
        println("  Loss: $(round(loss, digits=4))")
        
        # 2. Fase de Evaluación (El "Termómetro")
        agent_model = ModelAgent(mcts, SIMS_PER_MOVE * 2) # Más simulaciones para evaluar
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
