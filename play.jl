include(joinpath(@__DIR__, "src", "Awale.jl"))

using .Awale
using .Awale.State: GameState, initial_state, GameConfig
using .Awale.Env: is_terminal, transition, legal_actions
using .Awale.Model: load_model
using .Awale.Evaluation: ModelAgent, result_from_cutoff_state, result_from_terminal_state, select_action
using .Awale.MCTS: MCTSSearch, search
using .Awale.Utils: architecture_scoped_candidates, first_existing_path
using Random
using TOML

const ROOT_DIR = @__DIR__
config = TOML.parsefile(joinpath(ROOT_DIR, "config.toml"))
training_cfg = config["training"]
eval_cfg = config["evaluation"]
mcts_cfg = config["mcts"]

const DEFAULT_AGENT1_SPEC = "best"
const DEFAULT_AGENT2_SPEC = "human"
const DEFAULT_SIMS = Int(get(eval_cfg, "sims_per_eval", 100))
const MAX_TURNS = Int(training_cfg["max_turns"])
const C_PUCT = Float32(mcts_cfg["c_puct"])
const DIRICHLET_ALPHA = Float32(get(mcts_cfg, "dirichlet_alpha", 0.3))
const DIRICHLET_EPSILON = Float32(get(mcts_cfg, "dirichlet_epsilon", 0.25))
CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
MODEL_CONFIG_PATH = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))

"""
    HumanAgent

Agent that reads moves from standard input for interactive play.
Prompts the user for a pit choice and validates against legal moves.
"""
struct HumanAgent end

"""
    resolve_path(path::AbstractString) -> String

Resolve a potentially relative path against the repository root.
"""
function resolve_path(path::AbstractString)::String
    return isabspath(path) ? String(path) : joinpath(ROOT_DIR, String(path))
end

"""
    model_architecture_name() -> String

Return the architecture name from the model configuration TOML.
"""
function model_architecture_name()
    return Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"])
end

"""
    checkpoint_candidates(configured_path, default_filename) -> Vector{String}

Return ordered candidate checkpoint paths, resolved against the repo root.
"""
function checkpoint_candidates(configured_path::AbstractString, default_filename::AbstractString)
    return [resolve_path(path) for path in architecture_scoped_candidates(CHECKPOINT_DIR, model_architecture_name(), configured_path, default_filename)]
end

"""
    resolve_checkpoint_path(spec::AbstractString) -> String

Resolve an agent spec ("best", "last", "final", or an explicit path) to
a checkpoint file path.
"""
function resolve_checkpoint_path(spec::AbstractString)::String
    normalized = lowercase(strip(spec))
    if normalized == "best"
        candidates = checkpoint_candidates(String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin")
        found = first_existing_path(candidates)
        return found === nothing ? first(candidates) : found
    elseif normalized == "last"
        candidates = checkpoint_candidates(String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin")
        found = first_existing_path(candidates)
        return found === nothing ? first(candidates) : found
    elseif normalized == "final"
        candidates = checkpoint_candidates(String(get(eval_cfg, "checkpoint_path", joinpath(CHECKPOINT_DIR, "model_final.bin"))), "model_final.bin")
        found = first_existing_path(candidates)
        return found === nothing ? first(candidates) : found
    end

    return resolve_path(spec)
end

"""
    agent_label(spec::AbstractString) -> String

Derive a display label for an agent specification string.
"""
function agent_label(spec::AbstractString)::String
    normalized = lowercase(strip(spec))
    if normalized == "human"
        return "human"
    elseif normalized in ("best", "last", "final")
        return normalized
    end

    return basename(spec)
end

"""
    format_cell(label, seeds) -> String

Format a board pit cell as `[LL:SS]` for terminal display.
"""
function format_cell(label::Int, seeds)::String
    return "[" * lpad(string(label), 2) * ":" * lpad(string(Int(seeds)), 2) * "]"
end

"""
    print_legend(bottom_player::Int)

Print a legend explaining which player is at the bottom and top of the board display.
"""
function print_legend(bottom_player::Int)
    top_player = bottom_player == 1 ? 2 : 1
    println("Legend: P$bottom_player at the bottom, P$top_player at the top. Sowing is counterclockwise.")
    println("         The top row is shown in reverse order to follow the seed path.")
end

"""
    print_board(s::GameState; bottom_player::Int=1)

Render the Awale board to the terminal with pit labels and seed counts.
"""
function print_board(s::GameState; bottom_player::Int=1)
    top_player = bottom_player == 1 ? 2 : 1

    if bottom_player == 1
        top_labels = 12:-1:7
        top_row = s.board[12:-1:7]
        bottom_labels = 1:6
        bottom_row = s.board[1:6]
    else
        top_labels = 6:-1:1
        top_row = s.board[6:-1:1]
        bottom_labels = 1:6
        bottom_row = s.board[7:12]
    end

    println()
    println("                 P$(top_player) captured: $(Int(s.captured[top_player]))")
    println("      " * join((format_cell(label, seeds) for (label, seeds) in zip(top_labels, top_row)), " "))
    println("      " * join((format_cell(label, seeds) for (label, seeds) in zip(bottom_labels, bottom_row)), " "))
    println("                 P$(bottom_player) captured: $(Int(s.captured[bottom_player]))")
    println("                 Turn: P$(Int(s.to_move))")
end

"""
    prompt_human_action(s::GameState) -> Int

Prompt a human player for a legal move via stdin. Supports 'h' for help and 'q' to quit.
"""
function prompt_human_action(s::GameState)
    legal = legal_actions(s)
    println("Legal moves: $(join(legal, ", "))")

    while true
        print("Choose a pit [1-6], 'h' for help, or 'q' to quit: ")
        flush(stdout)
        raw = read_human_choice()

        if isempty(raw) && eof(stdin)
            throw(InterruptException())
        end

        if raw in ("q", "quit", "exit")
            throw(InterruptException())
        end

        if raw in ("h", "help", "?")
            println("Enter a number between 1 and 6 that appears in the legal moves list.")
            println("'q' leaves the game.")
            continue
        end

        action = try
            parse(Int, raw)
        catch
            0
        end

        if action in legal
            return action
        end

        println("[!] Invalid move. Try again.")
    end
end

"""
    resolve_agent(spec::AbstractString, sims::Int) -> Tuple{Any, String}

Resolve an agent spec string ("human", "best", "last", "final", or path)
to an agent instance and its display label.
"""
function resolve_agent(spec::AbstractString, sims::Int)
    normalized = lowercase(strip(spec))
    if normalized == "human"
        return HumanAgent(), "human"
    end

    path = resolve_checkpoint_path(spec)
    isfile(path) || throw(ArgumentError("Checkpoint not found for '$spec' at '$path'"))

    model = load_model(path)
    mcts = MCTSSearch(model, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}())
    return ModelAgent(mcts, sims), agent_label(spec)
end

"""
    final_result(s::GameState) -> Int

Compute the match result from player 1's perspective: 1 (win), -1 (loss), or 0 (draw).
Delegates to `result_from_terminal_state` or `result_from_cutoff_state`.
"""
function final_result(s::GameState)::Int
    if is_terminal(s)
        return result_from_terminal_state(s)
    end

    return result_from_cutoff_state(s)
end

"""
    print_help()

Print usage information for the play.jl script.
"""
function print_help()
    println("Awale terminal play")
    println()
    println("Usage:")
    println("  julia --project=. play.jl [--agent1 SPEC] [--agent2 SPEC] [--sims N] [--max-turns N] [--seed N] [--deterministic]")
    println()
    println("Agent specs:")
    println("  human   - interactive terminal player")
    println("  best    - training best checkpoint")
    println("  last    - last checkpoint")
    println("  final   - final evaluation checkpoint")
    println("  path    - explicit checkpoint path")
    println()
    println("Exhibition mode:")
    println("  --deterministic  disable stochastic AI move selection")
    println("  --seed N         reproduce a specific stochastic exhibition")
    println()
    println("Examples:")
    println("  julia --project=. play.jl --agent1 best --agent2 human")
    println("  julia --project=. play.jl --agent1 best --agent2 final")
    println("  julia --project=. play.jl --agent1 best --agent2 final --seed 42")
    println("  julia --project=. play.jl --agent1 checkpoints/model_best.bin --agent2 human --deterministic")
end

"""
    parse_args(args::Vector{String}) -> Union{Nothing, Dict{String, String}}

Parse command-line arguments for the play.jl script.
Returns `nothing` if `--help` was requested.
"""
function parse_args(args::Vector{String})
    opts = Dict{String, String}("agent1" => DEFAULT_AGENT1_SPEC, "agent2" => DEFAULT_AGENT2_SPEC, "sims" => string(DEFAULT_SIMS), "max-turns" => string(MAX_TURNS))
    i = 1

    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            return nothing
        elseif arg in ("--agent1", "--agent2", "--sims", "--max-turns", "--seed")
            i == length(args) && throw(ArgumentError("Missing value for $arg"))
            opts[replace(arg, "--" => "")] = args[i + 1]
            i += 2
        elseif arg == "--deterministic"
            opts["deterministic"] = "true"
            i += 1
        else
            throw(ArgumentError("Unknown argument: $arg"))
        end
    end

    return opts
end

"""
    parse_int_option(name, value) -> Int

Parse an integer CLI option, throwing `ArgumentError` on invalid input.
"""
function parse_int_option(name::AbstractString, value::AbstractString)::Int
    parsed = tryparse(Int, strip(value))
    parsed === nothing && throw(ArgumentError("$name must be an integer, got: '$value'"))
    return parsed
end

"""
    exhibition_stochastic(opts) -> Bool

Check whether the exhibition match should use stochastic (noisy) move selection.
"""
function exhibition_stochastic(opts::Dict{String, String})::Bool
    return !haskey(opts, "deterministic")
end

"""
    print_turn_banner(turn_no, player, label)

Print a turn header showing turn number, current player, and agent label.
"""
function print_turn_banner(turn_no::Int, player::Int, label::String)
    println()
    println("=== TURN $turn_no | P$player | agent: $label ===")
end

"""
    read_human_choice() -> String

Read a single line of human input from stdin, trimmed and lowercased.
Throws `InterruptException` on EOF.
"""
function read_human_choice()::String
    try
        return lowercase(strip(readline()))
    catch err
        if err isa EOFError
            throw(InterruptException())
        end
        rethrow()
    end
end

"""
    print_turn_action(player, action)

Print which player moved from which pit.
"""
function print_turn_action(player::Int, action::Int)
    println("P$player moves from pit $action")
    println("Move: $action")
end

"""
    select_exhibition_action(agent, state, rng; stochastic) -> Int

Select an action for exhibition play: humans are prompted, stochastic model
agents use noisy search, and deterministic model agents use argmax selection.
"""
function select_exhibition_action(agent, state::GameState, rng; stochastic::Bool)
    if agent isa HumanAgent
        return prompt_human_action(state)
    elseif stochastic
        return search(agent.mcts, state, agent.sims, rng; add_root_noise=true)
    else
        return select_action(agent, state, rng)
    end
end

"""
    print_turn_separator()

Print a visual separator between turns in the terminal display.
"""
function print_turn_separator()
    println(repeat("-", 40))
end

"""
    play_match_with_logs(agent1, label1, agent2, label2; config, max_turns, bottom_player, rng, stochastic) -> Union{Int, Nothing}

Play an exhibition match between two agents with verbose terminal logging.
Returns the result (1, -1, 0) or `nothing` if the game was aborted.
"""
function play_match_with_logs(agent1, label1::String, agent2, label2::String; config::GameConfig=GameConfig(), max_turns::Int=MAX_TURNS, bottom_player::Int=1, rng=Random.default_rng(), stochastic::Bool=true)
    state = initial_state(config)
    turns_played = 0

    println("--- Exhibition match ---")
    println("Mode: $(stochastic ? "stochastic" : "deterministic")")
    println("=== INITIAL STATE ===")
    print_legend(bottom_player)
    print_board(state; bottom_player=bottom_player)
    print_turn_separator()

    try
        while !is_terminal(state) && turns_played < max_turns
            current_player = Int(state.to_move)
            current_agent = current_player == 1 ? agent1 : agent2
            current_label = current_player == 1 ? label1 : label2
            turn_no = turns_played + 1

            print_turn_banner(turn_no, current_player, current_label)
            action = select_exhibition_action(current_agent, state, rng; stochastic=stochastic)
            print_turn_action(current_player, action)

            state = transition(state, action)
            turns_played += 1
            print_board(state; bottom_player=bottom_player)
            print_turn_separator()
        end
    catch err
        if err isa InterruptException
            println()
            println("[!] Game aborted by the user.")
            return nothing
        end
        rethrow()
    end

    result = final_result(state)
    println()
    println("--- End of match ---")
    if result == 1
        println("🏆 WINNER: P1")
    elseif result == -1
        println("🏆 WINNER: P2")
    else
        println("🤝 RESULT: Draw")
    end
    println("Duration: $turns_played turns")

    if turns_played >= max_turns && !is_terminal(state)
        println("[WARN] The match reached max_turns before a natural finish.")
    end

    return result
end

"""
    main(args::Vector{String}=Base.ARGS)

Entry point for the play.jl exhibition script. Parses CLI arguments,
resolves agents, and runs the match with verbose logging.
"""
function main(args::Vector{String}=Base.ARGS)
    opts = parse_args(args)
    opts === nothing && return print_help()

    println("--- Awale game viewer ---")

    sims = parse_int_option("--sims", opts["sims"])
    max_turns = parse_int_option("--max-turns", opts["max-turns"])
    stochastic = exhibition_stochastic(opts)
    exhibition_rng = haskey(opts, "seed") ? Random.MersenneTwister(parse_int_option("--seed", opts["seed"])) : Random.default_rng()

    agent1, label1 = resolve_agent(opts["agent1"], sims)
    agent2, label2 = resolve_agent(opts["agent2"], sims)

    bottom_player = agent1 isa HumanAgent ? 1 : agent2 isa HumanAgent ? 2 : 1
    println("Agents: P1=$label1 | P2=$label2")

    play_match_with_logs(agent1, label1, agent2, label2; max_turns=max_turns, bottom_player=bottom_player, rng=exhibition_rng, stochastic=stochastic)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
