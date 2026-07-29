using Flux
const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))
using .Awale
using .Awale.Training: run_training_iteration
using .Awale.State: GameState, GameConfig, initial_state, canonicalize, encode_state
using .Awale.Env: legal_actions, transition, is_terminal
using .Awale.Model: save_model, load_model, atomic_write, predict_raw, with_inference_mode
using .Awale.Evaluation: HeuristicAgent, RandomAgent, ModelAgent, evaluate_agents, evaluate_agents_on_openings, generate_opening_suite
using .Awale.MCTS: MCTSSearch
using .Awale.ReplayBuffers: ReplayBuffer
using .Awale.Publication: release_summary_path, release_id_slug, release_timestamp, runtime_config_snapshot_path, model_config_snapshot_path, write_release_summary
using .Awale.Utils: architecture_slug, architecture_scoped_path, architecture_scoped_candidates, first_existing_path
using Random
using TOML
using JSON
using SHA
using Statistics
import Dates: now as date_now, format as date_format, UTC

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
DIRICHLET_ALPHA = Float32(get(mcts_cfg, "dirichlet_alpha", 0.3))
DIRICHLET_EPSILON = Float32(get(mcts_cfg, "dirichlet_epsilon", 0.25))
const INITIAL_MODEL_SEED = Int(training_cfg["initial_model_seed"])
const BOOTSTRAP_RNG_SEED = Int(training_cfg["bootstrap_rng_seed"])
const MAX_TURNS = Int(training_cfg["max_turns"])
const TRAINING_STATE_RESUME_CONTRACT = "weights-only"

"""
    model_architecture_name() -> String

Return the architecture name from the model configuration TOML.
"""
function model_architecture_name()
    return Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"])
end

"""
    checkpoint_namespace_dir() -> String

Return the architecture-scoped checkpoint directory.
"""
function checkpoint_namespace_dir()
    return joinpath(CHECKPOINT_DIR, architecture_slug(model_architecture_name()))
end

"""
    checkpoint_write_path(configured_path, default_filename) -> String

Resolve a checkpoint write path under the architecture-scoped checkpoint directory.
"""
function checkpoint_write_path(configured_path::AbstractString, default_filename::AbstractString)
    return architecture_scoped_path(CHECKPOINT_DIR, model_architecture_name(), configured_path, default_filename)
end

"""
    checkpoint_candidates(configured_path, default_filename) -> Vector{String}

Return ordered candidate paths for checkpoint lookup (namespaced first, then legacy).
"""
function checkpoint_candidates(configured_path::AbstractString, default_filename::AbstractString)
    return architecture_scoped_candidates(CHECKPOINT_DIR, model_architecture_name(), configured_path, default_filename)
end

"""
    checkpoint_existing_path(configured_path, default_filename) -> Union{String, Nothing}

Return the path of an existing checkpoint file, or `nothing` if none exists.
"""
function checkpoint_existing_path(configured_path::AbstractString, default_filename::AbstractString)
    return first_existing_path(checkpoint_candidates(configured_path, default_filename))
end

"""
    training_log_dir() -> String

Return the log directory nested under the checkpoint namespace.
"""
function training_log_dir()
    return joinpath(checkpoint_namespace_dir(), "log")
end

"""
    training_log_file_path() -> String

Return a timestamped path for a training configuration log file.
"""
function training_log_file_path()
    timestamp = date_format(date_now(), "yyyy_mm_dd_HH_mm")
    architecture = architecture_slug(model_architecture_name())
    return joinpath(training_log_dir(), "training_config_$(architecture)_$timestamp.toml")
end

"""
    current_commit_sha() -> String

Return the current Git commit SHA of the repository.
"""
function current_commit_sha()
    return readchomp(`git -C $ROOT_DIR rev-parse HEAD`)
end

"""
    training_snapshot_path(iter::Int) -> String

Return the path for an iteration snapshot checkpoint.
"""
function training_snapshot_path(iter::Int)
    return joinpath(checkpoint_namespace_dir(), "model_iter_$(iter).bin")
end

"""
    training_last_checkpoint_path() -> String

Return the write path for the "last" checkpoint (latest trained model).
"""
function training_last_checkpoint_path()
    return checkpoint_write_path(String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin")
end

"""
    training_last_checkpoint_existing_path() -> Union{String, Nothing}

Return the existing "last" checkpoint path, or `nothing`.
"""
function training_last_checkpoint_existing_path()
    return checkpoint_existing_path(String(get(training_cfg, "last_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_last.bin"))), "model_last.bin")
end

"""
    training_best_checkpoint_path() -> String

Return the write path for the "best" checkpoint (promoted model).
"""
function training_best_checkpoint_path()
    return checkpoint_write_path(String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin")
end

"""
    training_best_checkpoint_existing_path() -> Union{String, Nothing}

Return the existing "best" checkpoint path, or `nothing`.
"""
function training_best_checkpoint_existing_path()
    return checkpoint_existing_path(String(get(training_cfg, "best_checkpoint_path", joinpath(CHECKPOINT_DIR, "model_best.bin"))), "model_best.bin")
end

"""
    training_state_file_path() -> String

Return the write path for the training state TOML file.
"""
function training_state_file_path()
    return checkpoint_write_path(String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml"))), "training_state.toml")
end

"""
    training_state_existing_path() -> Union{String, Nothing}

Return the existing training state path, or `nothing`.
"""
function training_state_existing_path()
    return checkpoint_existing_path(String(get(training_cfg, "state_path", joinpath(CHECKPOINT_DIR, "training_state.toml"))), "training_state.toml")
end

"""
    evaluation_checkpoint_path() -> String

Return the write path for the final evaluation checkpoint.
"""
function evaluation_checkpoint_path()
    return checkpoint_write_path(String(get(eval_cfg, "checkpoint_path", joinpath(CHECKPOINT_DIR, "model_final.bin"))), "model_final.bin")
end

"""
    evaluation_checkpoint_existing_path() -> Union{String, Nothing}

Return the existing evaluation checkpoint path, or `nothing`.
"""
function evaluation_checkpoint_existing_path()
    return checkpoint_existing_path(String(get(eval_cfg, "checkpoint_path", joinpath(CHECKPOINT_DIR, "model_final.bin"))), "model_final.bin")
end

"""
    write_training_state(path, last_iter, best_selection_score)

Atomically write training state metadata (last iteration, best score, resume contract) to a TOML file.
"""
function write_training_state(path::String, last_iter::Int, best_selection_score::Float64)
    atomic_write(path) do io
        println(io, "resume_contract = \"$(TRAINING_STATE_RESUME_CONTRACT)\"")
        println(io, "last_iter = $last_iter")
        println(io, "best_selection_score = $best_selection_score")
    end
end

"""
    read_training_state(path::String) -> Tuple{Int, Float64, String}

Read training state from a TOML file. Returns `(last_iter, best_selection_score, resume_contract)`.
Returns defaults `(0, -1.0, "weights-only")` if the file does not exist.
"""
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

"""
    decided_win_rate(results) -> Float64

Compute the win rate as a percentage of decided games (wins / (wins + losses)).
Returns 50% if no decided games exist.
"""
function decided_win_rate(results)::Float64
    decided = results.wins + results.losses
    return decided == 0 ? 50.0 : (results.wins / decided) * 100.0
end

"""
    validate_training_config()

Check training configuration invariants. Throws `ArgumentError` if any constraint is violated.
"""
function validate_training_config()
    UPDATES_PER_ITERATION > 0 || throw(ArgumentError("training.updates_per_iteration must be > 0"))
    0.0 <= REPLAY_RECENT_FRACTION <= 1.0 || throw(ArgumentError("training.replay_recent_fraction must be between 0 and 1"))
    REPLAY_RECENT_WINDOW >= 0 || throw(ArgumentError("training.replay_recent_window must be >= 0"))
    REPLAY_RECENT_FRACTION == 0.0 || REPLAY_RECENT_WINDOW > 0 || throw(ArgumentError("training.replay_recent_window must be > 0 when replay_recent_fraction > 0"))
    return nothing
end

"""
    validate_selection_config(games, opening_plies, openings_per_ply)

Check selection/promotion configuration invariants.
Throws `ArgumentError` if any constraint is violated.
"""
function validate_selection_config(games::Int, opening_plies, openings_per_ply::Int)
    BEST_TARGET_SIMS >= 0 || throw(ArgumentError("selection.target_sims must be >= 0"))
    games > 0 || throw(ArgumentError("selection.promotion_games must be > 0"))
    0.0 <= BEST_PROMOTION_THRESHOLD <= 100.0 || throw(ArgumentError("selection.promotion_threshold must be between 0 and 100"))
    0.0 <= ANCHOR_MIN_DECIDED_WIN_RATE <= 100.0 || throw(ArgumentError("selection.anchor_min_decided_win_rate must be between 0 and 100"))
    !isempty(opening_plies) || throw(ArgumentError("selection.opening_plies must not be empty"))
    openings_per_ply > 0 || throw(ArgumentError("selection.openings_per_ply must be > 0"))
end

"""
    build_selection_openings() -> Vector{GameState}

Generate and return the opening positions used for model selection evaluation.
"""
function build_selection_openings()
    validate_selection_config(BEST_PROMOTION_GAMES, BEST_OPENING_PLIES, BEST_OPENINGS_PER_PLY)
    return generate_opening_suite(
        plies=BEST_OPENING_PLIES,
        openings_per_ply=BEST_OPENINGS_PER_PLY,
        seed=BEST_OPENING_SEED,
    )
end

"""
    selection_rng(offset::Int) -> MersenneTwister

Create a seeded RNG for deterministic selection evaluation.
"""
selection_rng(offset::Int) = Random.MersenneTwister(BEST_SELECTION_SEED + offset)

"""
    create_initial_model() -> AwaleModel

Create and seed the initial neural network model.
"""
function create_initial_model()
    println("Initializing base model with fixed seed: $INITIAL_MODEL_SEED")
    Random.seed!(INITIAL_MODEL_SEED)
    return Awale.create_model(MODEL_CONFIG_PATH)
end

"""
    selection_gate_status(current_best_rate, anchor_reports) -> NamedTuple

Evaluate whether a candidate model passes all promotion gates:
best-model threshold and anchor-agent win-rate floors.
Returns a NamedTuple with `promoted`, `passes_best`, `passes_anchors`, and `reasons`.
"""
function selection_gate_status(current_best_rate, anchor_reports)
    passes_best = current_best_rate === nothing || current_best_rate >= BEST_PROMOTION_THRESHOLD
    passes_anchors = all(report.decided_win_rate >= ANCHOR_MIN_DECIDED_WIN_RATE for report in anchor_reports)
    promoted = passes_best && passes_anchors
    reasons = String[]
    !passes_best && push!(reasons, "current-best threshold")
    !passes_anchors && push!(reasons, "anchor floor")
    return (passes_best=passes_best, passes_anchors=passes_anchors, promoted=promoted, reasons=reasons)
end

"""
    evaluate_best_promotion(candidate_model) -> NamedTuple

Evaluate a candidate model against the current best and anchor agents (random, heuristic).
Returns promotion status, rates, and per-agent reports.
"""
function evaluate_best_promotion(candidate_model)
    selection_openings = build_selection_openings()
    candidate_agent = ModelAgent(MCTSSearch(candidate_model, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}()), BEST_TARGET_SIMS)
    current_best_results = nothing
    current_best_rate = nothing
    best_checkpoint_path = training_best_checkpoint_existing_path()

    if best_checkpoint_path !== nothing
        best_model = load_model(best_checkpoint_path)
        best_agent = ModelAgent(MCTSSearch(best_model, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}()), BEST_TARGET_SIMS)
        current_best_results = evaluate_agents_on_openings(candidate_agent, best_agent, selection_openings, BEST_PROMOTION_GAMES, selection_rng(1); max_turns=MAX_TURNS)
        current_best_rate = decided_win_rate(current_best_results)
    end

    anchor_reports = NamedTuple[]
    if USE_RANDOM_ANCHOR
        random_results = evaluate_agents_on_openings(candidate_agent, RandomAgent(), selection_openings, BEST_PROMOTION_GAMES, selection_rng(2); max_turns=MAX_TURNS)
        push!(anchor_reports, (name="random", results=random_results, decided_win_rate=decided_win_rate(random_results)))
    end
    if USE_HEURISTIC_ANCHOR
        heuristic_results = evaluate_agents_on_openings(candidate_agent, HeuristicAgent(), selection_openings, BEST_PROMOTION_GAMES, selection_rng(3); max_turns=MAX_TURNS)
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

"""
    maybe_promote_best!(model, best_selection_score_ref, selection) -> Bool

Promote the current model to "best" if the selection gate passed.
Updates the best score reference and saves the checkpoint.
"""
function maybe_promote_best!(model, best_selection_score_ref, selection)
    if !selection.promoted
        return false
    end

    best_selection_score_ref[] = selection.promotion_score
    save_model(model, training_best_checkpoint_path())
    return true
end

"""
    write_release_summary_file(release_summary_file; kwargs...) -> String

Write a release summary TOML with run metadata and metrics.
Convenience wrapper around `Awale.Publication.write_release_summary`.
"""
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
    println(" 📄 Release summary saved to: $release_summary_file")
    return release_summary_file
end

"""
    snapshot_run_configs(log_dir, architecture, release_id) -> Tuple{String, String}

Copy the current runtime configuration and model configuration TOML files
into the log directory with an architecture- and release-specific filename.
Returns `(runtime_config_path, model_config_path)`.
"""
function snapshot_run_configs(log_dir::String, architecture::AbstractString, release_id::AbstractString)
    arch = architecture_slug(architecture)
    runtime_config_path = runtime_config_snapshot_path(log_dir, arch, release_id)
    model_config_path = model_config_snapshot_path(log_dir, arch, release_id)
    model_config_source = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))

    println("Writing runtime configuration for architecture $arch to: $runtime_config_path")
    println("Writing model configuration for architecture $arch to: $model_config_path")

    runtime_data = read(joinpath(ROOT_DIR, "config.toml"), String)
    runtime_header = "# training_architecture = $arch\n# checkpoint_namespace = $(checkpoint_namespace_dir())\n"
    write(runtime_config_path, runtime_header * runtime_data)
    write(model_config_path, read(model_config_source, String))

    return runtime_config_path, model_config_path
end

"""
    maybe_resume_from_legacy_checkpoint!(model_ref, start_iter_ref)

Scan for legacy iteration checkpoints (model_iter_N.bin) and resume training
from the latest found iteration. Mutates `model_ref` and `start_iter_ref` in-place.
"""
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
        println("Legacy checkpoint detected! Resuming from iteration $last_iter...")
        println("Loading model: $checkpoint_file")
        model_ref[] = load_model(checkpoint_file)
    else
        println("Legacy training reached the maximum iteration ($last_iter).")
        start_iter_ref[] = NUM_ITERATIONS + 1
    end
end

"""
    is_power_of_two(value::Int) -> Bool

Check whether `value` is a power of two (used for snapshot scheduling).
"""
is_power_of_two(value::Int) = value > 0 && (value & (value - 1)) == 0

"""
    should_save_snapshot(iter, num_iterations, checkpoint_every) -> Bool

Determine whether an iteration checkpoint snapshot should be saved:
on iteration 1, on power-of-two iterations, and on intervals of `checkpoint_every`.
"""
function should_save_snapshot(iter::Int, num_iterations::Int, checkpoint_every::Int)
    return iter == 1 ||
        is_power_of_two(iter) ||
        (checkpoint_every > 0 && iter % checkpoint_every == 0)
end

"""
    save_promotion_history(pt::Awale.Metrics.ProgressTracker, path::String)

Serialize ProgressTracker state and promotion records to a TOML file for resume support.
"""
function save_promotion_history(pt::Awale.Metrics.ProgressTracker, path::String)
    open(path, "w") do io
        println(io, "[promotion_history]")
        println(io, "total_promotions = $(pt.total_promotions)")
        println(io, "longest_streak = $(pt.longest_streak)")
        println(io, "current_streak = $(pt.current_streak)")
        println(io, "last_best_iter = $(pt.last_best_iter)")
        if !isempty(pt.inter_promotion_gaps)
            gaps_str = "[" * join(string.(pt.inter_promotion_gaps), ", ") * "]"
            println(io, "inter_promotion_gaps = $gaps_str")
        end
        for (i, rec) in enumerate(pt.promotions)
            println(io, "[[promotions]]")
            println(io, "iteration = $(rec.iteration)")
            println(io, "win_rate_vs_best = $(rec.win_rate_vs_best)")
            println(io, "wins = $(rec.wins)")
            println(io, "losses = $(rec.losses)")
            println(io, "draws = $(rec.draws)")
            if rec.random_anchor_wr !== nothing
                println(io, "random_anchor_wr = $(rec.random_anchor_wr)")
            end
            println(io, "promotion_score = $(rec.promotion_score)")
            println(io, "gap_since_last = $(rec.gap_since_last)")
            println(io, "total_promotions_at_event = $(rec.total_promotions_at_event)")
            println(io, "elo_candidate = $(rec.elo_candidate)")
            println(io, "elo_best = $(rec.elo_best)")
            println(io, "timestamp = \"$(rec.timestamp)\"")
        end
    end
end

"""
    load_promotion_history!(pt::Awale.Metrics.ProgressTracker, path::String)

Deserialize ProgressTracker state and promotion records from a TOML file for resume support.
Silently handles missing or corrupt files.
"""
function load_promotion_history!(pt::Awale.Metrics.ProgressTracker, path::String)
    if !isfile(path)
        return
    end
    data = try
        TOML.parsefile(path)
    catch
        @warn "Corrupt promotion history file, starting fresh: $path"
        return
    end
    hist = get(data, "promotion_history", Dict())
    pt.total_promotions = Int(get(hist, "total_promotions", 0))
    pt.longest_streak = Int(get(hist, "longest_streak", 0))
    pt.current_streak = Int(get(hist, "current_streak", 0))
    pt.last_best_iter = Int(get(hist, "last_best_iter", 0))
    gaps = get(hist, "inter_promotion_gaps", Int[])
    pt.inter_promotion_gaps = gaps
    # Reload per-promotion records
    raw_promos = get(data, "promotions", [])
    for p in raw_promos
        promo = Awale.Metrics.PromotionRecord(
            Int(get(p, "iteration", 0)),
            Float64(get(p, "win_rate_vs_best", 50.0)),
            Int(get(p, "wins", 0)),
            Int(get(p, "losses", 0)),
            Int(get(p, "draws", 0)),
            get(p, "random_anchor_wr", nothing),
            Float64(get(p, "promotion_score", 0.0)),
            Int(get(p, "gap_since_last", 0)),
            Int(get(p, "total_promotions_at_event", 0)),
            Float64(get(p, "elo_candidate", 1500.0)),
            Float64(get(p, "elo_best", 1500.0)),
            String(get(p, "timestamp", "")),
        )
        push!(pt.promotions, promo)
    end
end

"""
    main(args::Vector{String}=Base.ARGS)

Entry point for the training pipeline. Parses CLI args (e.g. `--reset`),
loads or creates a model, runs training iterations with evaluation,
promotion, and snapshotting, and writes a release summary on completion.
"""
function main(args::Vector{String}=Base.ARGS)
    println("--- Starting Awale Training and Evaluation ---")

    validate_training_config()
    validate_selection_config(BEST_PROMOTION_GAMES, BEST_OPENING_PLIES, BEST_OPENINGS_PER_PLY)

    checkpoint_root = checkpoint_namespace_dir()
    mkpath(checkpoint_root)
    mkpath(training_log_dir())

    println("Active architecture: $(architecture_slug(model_architecture_name()))")
    println("Checkpoint namespace: $checkpoint_root")

    release_id = release_id_slug()
    commit_sha = current_commit_sha()
    runtime_config_snapshot, model_config_snapshot = snapshot_run_configs(training_log_dir(), model_architecture_name(), release_id)
    release_summary_file = release_summary_path(CHECKPOINT_DIR, model_architecture_name(), release_id)

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
        println("⚠️ [RESTART] Reset mode enabled. Ignoring checkpoints.")
    elseif last_checkpoint_path !== nothing && training_state_path !== nothing
        last_iter, saved_best_selection_score, resume_contract = read_training_state(training_state_path)
        best_selection_score[] = saved_best_selection_score

        if last_iter < NUM_ITERATIONS
            start_iter[] = last_iter + 1
            println("Checkpoint detected! Resuming from iteration $last_iter...")
            println("Resume contract: $resume_contract (weights only; optimizer/replay/RNG are not persisted).")
            println("Loading model: $last_checkpoint_path")
            model[] = load_model(last_checkpoint_path)
        else
            println("Training reached the maximum iteration ($last_iter).")
            if checkpoint_path !== nothing
                model[] = load_model(checkpoint_path)
            else
                model[] = load_model(last_checkpoint_path)
            end
            start_iter[] = NUM_ITERATIONS + 1
            last_completed_iter[] = last_iter
        end
    elseif checkpoint_path !== nothing
        println("Final model detected! Training has already completed.")
        model[] = load_model(checkpoint_path)
        last_completed_iter[] = NUM_ITERATIONS
        start_iter[] = NUM_ITERATIONS + 1
    else
        maybe_resume_from_legacy_checkpoint!(model, start_iter)
    end

    optimizer = Flux.setup(Flux.Adam(LEARNING_RATE), model[])
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    training_mcts = MCTSSearch(model[], C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}())
    evaluation_mcts = MCTSSearch(model[], C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, Dict{UInt64, Tuple{Float64, Int64}}())
    agent_random = RandomAgent()

    # ── Network drift reference set ────────────────────────
    drift_rng = Random.MersenneTwister(42)
    reference_states = GameState[]
    for _ in 1:20
        state = initial_state(GameConfig())
        for _ in 1:50
            is_terminal(state) && break
            push!(reference_states, canonicalize(state))
            actions = legal_actions(state)
            isempty(actions) && break
            state = transition(state, actions[rand(drift_rng, 1:length(actions))])
        end
    end
    if length(reference_states) > 200
        reference_states = reference_states[1:200]
    end
    println("  Network drift reference set: $(length(reference_states)) states")
    drift_X = hcat([vec(encode_state(canonicalize(s))) for s in reference_states]...)

    # ── Training-progress trackers ──────────────────────────
    elo_tracker = Awale.Metrics.EloTracker()
    progress_tracker = Awale.Metrics.ProgressTracker()
    promotion_history_file = joinpath(checkpoint_namespace_dir(), "promotion_history.toml")
    learning_curve_file = joinpath(training_log_dir(), "learning_curve_$(architecture_slug(model_architecture_name()))_$(release_id).csv")

    # ── JSONL metrics setup ──────────────────────────────────
    METRIC_VERSION = "1.0.0"
    GIT_COMMIT = commit_sha
    ARCHITECTURE = architecture_slug(model_architecture_name())
    config_bytes = read(joinpath(ROOT_DIR, "config.toml"))
    CONFIG_HASH = bytes2hex(sha256(config_bytes))
    jsonl_file = joinpath(training_log_dir(), "metrics_$(architecture_slug(model_architecture_name()))_$(release_id).jsonl")

    # CSV: append on resume, write header only for new files
    csv_fresh = !isfile(learning_curve_file) || filesize(learning_curve_file) == 0
    csv_io = open(learning_curve_file, "a")

    # JSONL: append mode, no header
    jsonl_io = open(jsonl_file, "a")

    # Promotion history: restore tracked state on resume
    if isfile(promotion_history_file)
        println("Loading promotion history from: $promotion_history_file")
        load_promotion_history!(progress_tracker, promotion_history_file)
    end

    # ── Δ-metrics & moving average state ─────────────────────
    prev_kl_mean = NaN
    prev_policy_dist = NaN
    prev_top1_pct = NaN
    prev_drift_kl = NaN
    prev_grad_norm = NaN
    prev_param_update = NaN
    ma_buf_loss = Float32[]
    ma_buf_policy_loss = Float32[]
    ma_buf_value_loss = Float32[]
    ma_buf_kl = Float32[]
    ma_buf_top1 = Float32[]
    ma_buf_drift = Float32[]
    ma_buf_rating = Float32[]
    ma_buf_entropy = Float32[]

    try
    if csv_fresh
        Awale.Metrics.write_csv_header(csv_io)
    end

    if start_iter[] <= NUM_ITERATIONS
        for iter in start_iter[]:NUM_ITERATIONS
            println("\nIteration $iter / $NUM_ITERATIONS")
            iter_start_ns = time_ns()

            # Compute network drift before this iteration
            before_logits, _ = with_inference_mode(() -> predict_raw(model[], drift_X), model[])

            old_params_vec, _ = Flux.destructure(model[])
            training_result, calib_data = run_training_iteration(
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
            iter_duration_s = (time_ns() - iter_start_ns) / 1e9
            new_params_vec, _ = Flux.destructure(model[])
            param_update_norm = sqrt(sum((new_params_vec .- old_params_vec) .^ 2))
            loss = training_result.avg_loss
            println("  Average loss: $(round(loss, digits=4))")
            println("  Param update norm: $(round(param_update_norm, digits=6))")
            println("  Replay buffer: $(length(replay_buffer)) samples")
            last_loss[] = Float64(loss)

            # ── Network Drift ─────────────────────────────
            after_logits, _ = with_inference_mode(() -> predict_raw(model[], drift_X), model[])
            before_log_probs = Flux.logsoftmax(before_logits, dims=1)
            after_probs_drift = softmax(after_logits, dims=1)
            drift_kl = sum(exp.(before_log_probs) .* (before_log_probs .- log.(clamp.(after_probs_drift, 1.0f-10, 1.0f0))), dims=1)
            avg_drift_kl = sum(drift_kl) / length(drift_kl)
            println("  ── Network Drift ─────────────────────────────")
            println("    Avg KL(network_before || network_after): $(round(avg_drift_kl, digits=4))")
            println("  ────────────────────────────────────────────────")

            # Value calibration from last training batch
            value_cal = if !isempty(calib_data.v_pred)
                Awale.Metrics.compute_value_calibration(vec(calib_data.v_pred), vec(calib_data.v_target))
            else
                (mae=NaN, pearson_r=NaN, spearman_rho=NaN)
            end

            agent_model = ModelAgent(evaluation_mcts, SIMS_PER_EVAL)
            results = evaluate_agents(agent_model, agent_random, EVAL_GAMES, Awale.GameConfig(), rng; max_turns=MAX_TURNS)
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

            # Update Elo from candidate vs current-best results
            if selection.current_best_results !== nothing
                br = selection.current_best_results
                Awale.Metrics.update_elo!(elo_tracker, br.wins, br.losses, br.draws, iter)
            end

            # ── Elo expected / actual / delta / upset ───────────
            elo_expected = 1.0 / (1.0 + 10.0 ^ ((elo_tracker.best_rating - elo_tracker.candidate_rating) / 400.0))
            if selection.current_best_results !== nothing
                br = selection.current_best_results
                total_s = br.wins + br.losses + br.draws
                elo_actual = (br.wins + 0.5 * br.draws) / max(1, total_s)
            else
                elo_actual = NaN
            end
            elo_delta = elo_tracker.k * (elo_actual - elo_expected)
            elo_upset = abs(elo_actual - elo_expected)

            promoted_flag = false
            if maybe_promote_best!(model[], best_selection_score, selection)
                println("  ✅ New best model saved to: $(training_best_checkpoint_path())")
                promoted_flag = true

                # Build promotion record
                random_wr = nothing
                for report in selection.anchor_reports
                    if report.name == "random"
                        random_wr = report.decided_win_rate
                    end
                end
                promo_record = Awale.Metrics.PromotionRecord(
                    iter,
                    selection.current_best_rate === nothing ? 50.0 : selection.current_best_rate,
                    selection.current_best_results === nothing ? 0 : selection.current_best_results.wins,
                    selection.current_best_results === nothing ? 0 : selection.current_best_results.losses,
                    selection.current_best_results === nothing ? 0 : selection.current_best_results.draws,
                    random_wr,
                    selection.promotion_score,
                    iter - progress_tracker.last_best_iter,
                    progress_tracker.total_promotions + 1,
                    elo_tracker.candidate_rating,
                    elo_tracker.best_rating,
                    date_format(date_now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
                )
                Awale.Metrics.record_promotion!(progress_tracker, iter, promo_record)
                Awale.Metrics.promote_elo!(elo_tracker)

                # Promotion event print (R8)
                gap = progress_tracker.total_promotions <= 1 ? iter : progress_tracker.inter_promotion_gaps[end]
                Awale.Metrics.print_promotion_event(
                    progress_tracker.total_promotions,
                    gap,
                    selection.current_best_rate,
                )
                save_promotion_history(progress_tracker, promotion_history_file)
            else
                Awale.Metrics.record_non_promotion!(progress_tracker)
            end

            if should_save_snapshot(iter, NUM_ITERATIONS, CHECKPOINT_EVERY)
                snapshot_path = training_snapshot_path(iter)
                save_model(model[], snapshot_path)
                println("  📦 Snapshot saved to: $snapshot_path")
            end

            # ── Δ-metrics ──────────────────────────────────────
            delta_kl = isfinite(prev_kl_mean) ? abs(training_result.kl_mean - prev_kl_mean) : NaN
            delta_policy_dist = isfinite(prev_policy_dist) ? abs(training_result.l1_mean - prev_policy_dist) : NaN
            delta_top1 = isfinite(prev_top1_pct) ? abs(training_result.top1_pct - prev_top1_pct) : NaN
            delta_drift = isfinite(prev_drift_kl) ? abs(avg_drift_kl - prev_drift_kl) : NaN
            delta_grad_norm = isfinite(prev_grad_norm) ? abs(training_result.avg_grad_norm - prev_grad_norm) : NaN
            delta_param_update = isfinite(prev_param_update) ? abs(param_update_norm - prev_param_update) : NaN
            prev_kl_mean = training_result.kl_mean
            prev_policy_dist = training_result.l1_mean
            prev_top1_pct = training_result.top1_pct
            prev_drift_kl = avg_drift_kl
            prev_grad_norm = training_result.avg_grad_norm
            prev_param_update = param_update_norm

            # ── Update moving average buffers ──────────────────
            push!(ma_buf_loss, training_result.avg_loss)
            push!(ma_buf_policy_loss, training_result.avg_policy_loss)
            push!(ma_buf_value_loss, training_result.avg_value_loss)
            push!(ma_buf_kl, training_result.kl_mean)
            push!(ma_buf_top1, training_result.top1_pct)
            push!(ma_buf_drift, avg_drift_kl)
            push!(ma_buf_rating, Float32(elo_tracker.candidate_rating))
            push!(ma_buf_entropy, training_result.avg_pred_entropy)

            ma_win(buf, n) = length(buf) >= n ? sum(buf[end-n+1:end]) / n : NaN
            ma_loss_5 = ma_win(ma_buf_loss, 5)
            ma_loss_10 = ma_win(ma_buf_loss, 10)
            ma_loss_20 = ma_win(ma_buf_loss, 20)
            ma_policy_loss_5 = ma_win(ma_buf_policy_loss, 5)
            ma_value_loss_5 = ma_win(ma_buf_value_loss, 5)
            ma_kl_5 = ma_win(ma_buf_kl, 5)
            ma_kl_10 = ma_win(ma_buf_kl, 10)
            ma_kl_20 = ma_win(ma_buf_kl, 20)
            ma_top1_5 = ma_win(ma_buf_top1, 5)
            ma_top1_10 = ma_win(ma_buf_top1, 10)
            ma_top1_20 = ma_win(ma_buf_top1, 20)
            ma_drift_5 = ma_win(ma_buf_drift, 5)
            ma_drift_10 = ma_win(ma_buf_drift, 10)
            ma_drift_20 = ma_win(ma_buf_drift, 20)
            ma_rating_5 = ma_win(ma_buf_rating, 5)
            ma_rating_10 = ma_win(ma_buf_rating, 10)
            ma_rating_20 = ma_win(ma_buf_rating, 20)
            ma_entropy_5 = ma_win(ma_buf_entropy, 5)
            ma_entropy_10 = ma_win(ma_buf_entropy, 10)
            ma_entropy_20 = ma_win(ma_buf_entropy, 20)

            # ── Convergence stability (passive) ────────────────
            window_size = min(20, length(ma_buf_kl))
            if window_size >= 5
                kl_window = ma_buf_kl[end-window_size+1:end]
                drift_window = ma_buf_drift[end-window_size+1:end]
                top1_window = ma_buf_top1[end-window_size+1:end]
                param_window = ma_buf_policy_loss[end-window_size+1:end]
                kl_stable = var(kl_window) < 0.001f0 ? "STALLED" : "ACTIVE"
                drift_stable = var(drift_window) < 0.0001f0 ? "STALLED" : "ACTIVE"
                top1_stable = var(top1_window) < 1.0f0 ? "STALLED" : "ACTIVE"
                param_stable = var(param_window) < 0.01f0 ? "STALLED" : "ACTIVE"
                stability_str = " KL:$kl_stable Drift:$drift_stable Top1:$top1_stable Param:$param_stable"
            else
                stability_str = " KL:BOOTSTRAP Drift:BOOTSTRAP Top1:BOOTSTRAP Param:BOOTSTRAP"
            end

            # ── Health dashboard ──────────────────────────────
            net_health = delta_kl > 0.01f0 || param_update_norm > 0.01f0 ? "ACTIVE" : "STALLED"
            srch_health = training_result.top1_pct < 80.0f0 ? "HIGH" : "LOW"
            drift_health = avg_drift_kl < 0.01f0 ? "LOW" : (avg_drift_kl < 0.05f0 ? "MEDIUM" : "HIGH")
            promo_since = progress_tracker.last_best_iter > 0 ? iter - progress_tracker.last_best_iter : iter
            cal_health = if isfinite(value_cal.mae)
                value_cal.mae < 0.5f0 ? "OK" : "HIGH"
            else
                "N/A"
            end
            health_line = "Net:$net_health Srch:$srch_health Drift:$drift_health Promo:$promo_since/$(progress_tracker.total_promotions) Rply:$(training_result.replay_pct)% ValCal:$cal_health"
            println("  ── Health ─────────────────────────────────────")
            println("    $health_line$stability_str")
            println("  ────────────────────────────────────────────────")

            # ── Warnings ───────────────────────────────────────
            if training_result.top1_pct > 95.0f0
                println("  ⚠ Top-1 agreement $(round(training_result.top1_pct, digits=1))% — search may no longer improve policy")
            end
            if avg_drift_kl < 1.0f-6
                println("  ⚠ Network drift near zero — training may have converged")
            end
            rolling_mean = ma_win(ma_buf_policy_loss, 5)
            if isfinite(rolling_mean) && rolling_mean > 0.0f0 && param_update_norm > 5.0f0 * rolling_mean
                println("  ⚠ Parameter update unusually large — possible instability")
            end

            # Write learning curve CSV row
            Awale.Metrics.write_csv_row(
                csv_io,
                iter,
                training_result.avg_loss,
                training_result.avg_policy_loss,
                training_result.avg_value_loss,
                training_result.avg_grad_norm,
                training_result.avg_pred_entropy,
                training_result.avg_target_entropy,
                param_update_norm,
                training_result.replay_pct,
                training_result.avg_game_len,
                win_rate,
                selection.current_best_rate,
                promoted_flag,
                elo_tracker.candidate_rating,
                elo_tracker.best_rating,
            )
            flush(csv_io)

            # ── Assemble JSONL entry ───────────────────────────
            mcts_root_q = length(calib_data.v_pred) > 0 ? mean(vec(calib_data.v_pred)) : NaN
            jsonl_dict = Dict{String, Any}(
                # Versioning
                "metric_version" => METRIC_VERSION,
                "git_commit" => GIT_COMMIT,
                "architecture" => ARCHITECTURE,
                "config_hash" => CONFIG_HASH,
                # Core
                "iter" => iter,
                "timestamp" => date_format(date_now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
                "duration_s" => round(iter_duration_s, digits=2),
                # Existing CSV fields
                "avg_loss" => Float64(round(training_result.avg_loss, digits=6)),
                "policy_loss" => Float64(round(training_result.avg_policy_loss, digits=6)),
                "value_loss" => Float64(round(training_result.avg_value_loss, digits=6)),
                "grad_norm" => Float64(round(training_result.avg_grad_norm, digits=6)),
                "pred_entropy" => Float64(round(training_result.avg_pred_entropy, digits=6)),
                "target_entropy" => Float64(round(training_result.avg_target_entropy, digits=6)),
                "param_update_norm" => Float64(round(param_update_norm, digits=6)),
                "replay_fill_pct" => Float64(training_result.replay_pct),
                "avg_game_len" => Float64(round(training_result.avg_game_len, digits=2)),
                "baseline_wr" => Float64(round(win_rate, digits=2)),
                "candidate_vs_best_wr" => selection.current_best_rate,
                "promoted" => promoted_flag,
                "elo_candidate" => Float64(round(elo_tracker.candidate_rating, digits=2)),
                "elo_best" => Float64(round(elo_tracker.best_rating, digits=2)),
                # MCTS aggregates
                "mcts_kl_mean" => Float64(round(training_result.kl_mean, digits=6)),
                "mcts_kl_median" => Float64(round(training_result.kl_median, digits=6)),
                "mcts_top1_pct" => Float64(round(training_result.top1_pct, digits=2)),
                "mcts_top2_pct" => Float64(round(training_result.top2_pct, digits=2)),
                "mcts_top3_pct" => Float64(round(training_result.top3_pct, digits=2)),
                "mcts_root_conf_mean" => Float64(round(training_result.root_conf_mean, digits=6)),
                "mcts_l1_mean" => Float64(round(training_result.l1_mean, digits=6)),
                # Drift & root Q
                "drift_kl" => Float64(round(avg_drift_kl, digits=6)),
                "mcts_root_q" => Float64(round(mcts_root_q, digits=6)),
                # Value calibration
                "value_cal_mae" => Float64(round(value_cal.mae, digits=6)),
                "value_cal_pearson" => Float64(round(value_cal.pearson_r, digits=6)),
                "value_cal_spearman" => Float64(round(value_cal.spearman_rho, digits=6)),
                # Elo
                "elo_expected" => Float64(round(elo_expected, digits=4)),
                "elo_actual" => Float64(round(elo_actual, digits=4)),
                "elo_delta" => Float64(round(elo_delta, digits=4)),
                "elo_upset" => Float64(round(elo_upset, digits=4)),
                # Δ-metrics
                "delta_kl" => Float64(round(delta_kl, digits=6)),
                "delta_policy_dist" => Float64(round(delta_policy_dist, digits=6)),
                "delta_top1" => Float64(round(delta_top1, digits=6)),
                "delta_drift" => Float64(round(delta_drift, digits=6)),
                "delta_grad_norm" => Float64(round(delta_grad_norm, digits=6)),
                "delta_param_update" => Float64(round(delta_param_update, digits=6)),
                # Moving averages
                "ma_loss_5" => Float64(round(ma_loss_5, digits=6)),
                "ma_loss_10" => Float64(round(ma_loss_10, digits=6)),
                "ma_loss_20" => Float64(round(ma_loss_20, digits=6)),
                "ma_policy_loss_5" => Float64(round(ma_policy_loss_5, digits=6)),
                "ma_value_loss_5" => Float64(round(ma_value_loss_5, digits=6)),
                "ma_kl_5" => Float64(round(ma_kl_5, digits=6)),
                "ma_kl_10" => Float64(round(ma_kl_10, digits=6)),
                "ma_kl_20" => Float64(round(ma_kl_20, digits=6)),
                "ma_top1_5" => Float64(round(ma_top1_5, digits=6)),
                "ma_top1_10" => Float64(round(ma_top1_10, digits=6)),
                "ma_top1_20" => Float64(round(ma_top1_20, digits=6)),
                "ma_drift_5" => Float64(round(ma_drift_5, digits=6)),
                "ma_drift_10" => Float64(round(ma_drift_10, digits=6)),
                "ma_drift_20" => Float64(round(ma_drift_20, digits=6)),
                "ma_rating_5" => Float64(round(ma_rating_5, digits=6)),
                "ma_rating_10" => Float64(round(ma_rating_10, digits=6)),
                "ma_rating_20" => Float64(round(ma_rating_20, digits=6)),
                "ma_entropy_5" => Float64(round(ma_entropy_5, digits=6)),
                "ma_entropy_10" => Float64(round(ma_entropy_10, digits=6)),
                "ma_entropy_20" => Float64(round(ma_entropy_20, digits=6)),
            )
            # Convert NaN/Inf values to JSON null
            cleaned = Dict{String, Any}()
            for (k, v) in jsonl_dict
                cleaned[k] = (v isa AbstractFloat && !isfinite(v)) ? nothing : v
            end
            jsonl_str = JSON.json(cleaned)
            println(jsonl_io, jsonl_str)
            flush(jsonl_io)

            Awale.Metrics.print_progress_diagnostics(param_update_norm, progress_tracker, elo_tracker, iter)

            write_training_state(training_state_file_path(), iter, best_selection_score[])
            last_completed_iter[] = iter
        end

        save_model(model[], evaluation_checkpoint_path())
        println("\n--- Training Complete ---")
        println(" Final model saved to: $(evaluation_checkpoint_path())")

    else
        println("--- Training already completed. ---")
    end

    finally
        close(csv_io)
        close(jsonl_io)
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
