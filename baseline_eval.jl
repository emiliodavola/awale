include(joinpath(@__DIR__, "src", "Awale.jl"))
using .Awale
using .Awale.Evaluation: RandomAgent, ModelAgent, HeuristicAgent, evaluate_agents
using .Awale.MCTS: MCTSSearch
using .Awale.Model: create_model, load_model
using TOML

config = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
eval_cfg = config["evaluation"]
mcts_cfg = config["mcts"]

EVAL_GAMES = Int(eval_cfg["eval_games"])
SIMS_PER_EVAL = Int(eval_cfg["sims_per_eval"])
CHECKPOINT_PATH = String(eval_cfg["checkpoint_path"])
C_PUCT = Float32(mcts_cfg["c_puct"])

function run_eval(name, agent_under_test, baseline_agent, n_games=20, sims=40)
    println("Evaluando $name vs $(typeof(baseline_agent)) sobre $n_games juegos...")
    results = evaluate_agents(agent_under_test, baseline_agent, n_games)

    win_rate = (results.wins / n_games) * 100
    loss_rate = (results.losses / n_games) * 100
    draw_rate = (results.draws / n_games) * 100

    println(" Resultados: Wins: $(results.wins) ($(round(win_rate, digits=2))%) | Losses: $(results.losses) ($(round(loss_rate, digits=2))%) | Draws: $(results.draws) ($(round(draw_rate, digits=2))%) | Avg Turns: $(round(results.avg_turns, digits=2))")
    println(repeat("-", 50))
end

function main()
    println("--- Evaluación de Modelo Awale ---")

    if isfile(CHECKPOINT_PATH)
        println("Cargando modelo desde $CHECKPOINT_PATH...")
        model = load_model(CHECKPOINT_PATH)
    else
        println("No se encontró checkpoint. Usando modelo aleatorio.")
        model = create_model()
    end

    mcts = MCTSSearch(model, C_PUCT, Dict{UInt64, Tuple{Float32, Int64}}())
    agent_model = ModelAgent(mcts, SIMS_PER_EVAL)
    agent_random = RandomAgent()
    agent_heuristic = HeuristicAgent()

    run_eval("Model", agent_model, agent_random, EVAL_GAMES, SIMS_PER_EVAL)
    run_eval("Model", agent_model, agent_heuristic, EVAL_GAMES, SIMS_PER_EVAL)
    run_eval("Heuristic", agent_heuristic, agent_random, EVAL_GAMES, SIMS_PER_EVAL)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
