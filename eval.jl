using Flux
using Awale
using Awale.Evaluation: RandomAgent, HeuristicAgent, ModelAgent, evaluate_agents
using Awale.MCTS: MCTSSearch
using Awale.Model: create_model, load_model

function run_eval(name, agent_under_test, baseline_agent, n_games=20, sims=40)
    println("Evaluando $name vs $(typeof(baseline_agent)) sobre $n_games juegos...")
    results = evaluate_agents(agent_under_test, baseline_agent, n_games)
    
    win_rate = (results.wins / n_games) * 100
    loss_rate = (results.losses / n_games) * 100
    draw_rate = (results.draws / n_games) * 100
    
    println(" Resultdos: Wins: $(results.wins) ($(round(win_rate, digits=2))%) | Losses: $(results.losses) ($(round(loss_rate, digits=2))%) | Draws: $(results.draws) ($(round(draw_rate, digits=2))%) | Avg Turns: $(round(results.avg_turns, digits=2))")
    println(repeat("-", 50))
end

function main()
    println("--- Evaluación de Modelo Awale ---")
    
    # Configuración
    NUM_GAMES = 40
    SIMS_PER_MOVE = 40
    CHECKPOINT_PATH = "checkpoints/model_final.bin"
    
    if isfile(CHECKPOINT_PATH)
        println("Cargando modelo desde $CHECKPOINT_PATH...")
        model = load_model(CHECKPOINT_PATH)
    else
        println("No se encontró checkpoint. Usando modelo aleatorio.")
        model = create_model()
    end
    
    mcts = MCTSSearch(model, 1.4f0)
    agent_model = ModelAgent(mcts, SIMS_PER_MOVE)
    agent_random = RandomAgent()
    agent_heuristic = HeuristicAgent()
    
    run_eval("Model", agent_model, agent_random, NUM_GAMES, SIMS_PER_MOVE)
    run_eval("Model", agent_model, agent_heuristic, NUM_GAMES, SIMS_PER_MOVE)
    run_eval("Heuristic", agent_heuristic, agent_random, NUM_GAMES, SIMS_PER_MOVE)
end

main()
