using Flux
using Awale
using Awale.State: GameState, initial_state, GameConfig
using Awale.Env: is_terminal, transition
using Awale.Training: run_training_iteration
using Awale.Model: save_model, load_model
using Awale.Evaluation: RandomAgent, HeuristicAgent, ModelAgent, play_match, select_action
using Awale.MCTS: MCTSSearch
using Random
using JSON
using Dates

# Load configuration
config = JSON.parsefile("config.json")
eval_cfg = config["evaluation"]
mcts_cfg = config["mcts"]

# Mapping
EVAL_GAMES = Int(eval_cfg["eval_games"])
SIMS_PER_EVAL = Int(eval_cfg["sims_per_eval"])
CHECKPOINT_PATH = String(eval_cfg["checkpoint_path"])
C_PUCT = Float32(mcts_cfg["c_puct"])

function print_board(s::GameState)
    # Representación limpia: [0 4 4 4 4 4 | 4 4 4 4 4 4]
    board_str = join(map(x->string(x), s.board[1:6]), " ") * " | " * join(map(x->string(x), s.board[7:12]), " ")
    println("  Board: [$board_str]")
    println("  Captured: P1: $(s.captured[1]), P2: $(s.captured[2])")
    println("  To move: Player $(s.to_move)")
    println(repeat("-", 30))
end

function play_match_with_logs(agent_p1, agent_p2, config::GameConfig=GameConfig())
    s = initial_state(config)
    turn = 1
    turns_played = 0
    max_turns = 1000
    
    println("--- Comienzo del Partido ---")
    print_board(s)
    
    while !is_terminal(s) && turns_played < max_turns
        current_agent = (turn == 1) ? agent_p1 : agent_p2
        agent_name = (turn == 1) ? "Player 1" : "Player 2"
        
        action = select_action(current_agent, s)
        s_prev = s # Guardamos el estado anterior para detectar cambios
        s = transition(s, action)
        turns_played += 1
        
        # Imprimir SOLO si hay captura O es un turno de resumen (cada 5 turnos)
        if s.captured != s_prev.captured || turns_played % 5 == 0
            println("  $agent_name: Acción $action")
            print_board(s)
        end
    end
    
    res, turns = play_match(agent_p1, agent_p2, config)
    
    println("--- Fin del Partido ---")
    if res == 1
        println("🏆 GANADOR: Player 1")
    elseif res == -1
        println("🏆 GANADOR: Player 2")
    else
        println("🤝 RESULTADO: Empate")
    end
    println("  Duración: $turns turnos")
end

function main(args::Vector{String}=Base.ARGS)
    println("--- Visualizador de Juego de Awale ---")
    
    if isfile(CHECKPOINT_PATH)
        println("Cargando modelo entrenado desde: $CHECKPOINT_PATH")
        model = load_model(CHECKPOINT_PATH)
    else
        println("No se encontró modelo final. Usando modelo aleatorio.")
        model = create_model()
    end
    
    mcts = MCTSSearch(model, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}())
    agent_model = ModelAgent(mcts, SIMS_PER_EVAL)
    agent_heuristic = HeuristicAgent()
    agent_random = RandomAgent()

    println("\n[EXHIBICIÓN 1] Model vs Heuristic (Greedy)")
    play_match_with_logs(agent_model, agent_heuristic)

    println("\n[EXHIBICIÓN 2] Model vs Random")
    play_match_with_logs(agent_model, agent_random)
end

main()
