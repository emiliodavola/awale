const ROOT_DIR = @__DIR__
include(joinpath(ROOT_DIR, "src", "Awale.jl"))

using .Awale
using .Awale.Publication: default_repo_path, latest_release_summary_path, plan_release_bundle, publish_release_bundle, read_release_summary, resolve_repo_path, stage_release_bundle
using .Awale.Utils: architecture_slug
using TOML

config = TOML.parsefile(joinpath(ROOT_DIR, "config.toml"))
training_cfg = config["training"]
CHECKPOINT_DIR = String(training_cfg["checkpoint_dir"])
MODEL_CONFIG_PATH = abspath(ROOT_DIR, String(get(training_cfg, "model_config_path", joinpath("src", "Awale", "config.toml"))))

"""
    model_architecture_name() -> String

Return the architecture name from the model configuration TOML.
"""
function model_architecture_name()
    return Awale.Model.model_architecture(TOML.parsefile(MODEL_CONFIG_PATH)["model"])
end

"""
    resolve_path(path::AbstractString) -> String

Resolve a potentially relative path against the repository root.
"""
function resolve_path(path::AbstractString)::String
    return resolve_repo_path(ROOT_DIR, path)
end

"""
    print_help()

Print usage information for the publish_hf.jl script.
"""
function print_help()
    println("Awale Hugging Face publisher")
    println()
    println("Usage:")
    println("  julia --project=. publish_hf.jl [--dry-run|--stage|--publish] [--summary-path PATH] [--architecture NAME] [--repo-id REPO] [--repo-path PATH_IN_REPO]")
    println()
    println("Modes:")
    println("  --dry-run   validate the release bundle without writing files")
    println("  --stage     assemble the bundle locally under checkpoints/<architecture>/release/<release_id>/")
    println("  --publish   export safe public release artifacts and upload with `hf upload REPO_ID LOCAL_PATH PATH_IN_REPO`")
    println()
    println("Authentication:")
    println("  export HF_TOKEN=...")
    println("  hf auth login --token \$HF_TOKEN --add-to-git-credential")
    println()
    println("Examples:")
    println("  julia --project=. publish_hf.jl --dry-run")
    println("  julia --project=. publish_hf.jl --stage")
    println("  HF_TOKEN=... julia --project=. publish_hf.jl --publish --repo-id user/awale-results")
end

"""
    parse_args(args::Vector{String}) -> Union{Nothing, Dict{String, String}}

Parse command-line arguments for the publish_hf.jl script.
Returns `nothing` if `--help` was requested.
"""
function parse_args(args::Vector{String})
    opts = Dict{String, String}()
    i = 1

    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            return nothing
        elseif arg in ("--dry-run", "--stage", "--publish")
            haskey(opts, "mode") && opts["mode"] != replace(arg, "--" => "") && throw(ArgumentError("Use only one mode flag: --dry-run, --stage, or --publish"))
            opts["mode"] = replace(arg, "--" => "")
            i += 1
        elseif arg in ("--summary-path", "--architecture", "--repo-id", "--repo-path")
            i == length(args) && throw(ArgumentError("Missing value for $arg"))
            opts[replace(arg, "--" => "")] = args[i + 1]
            i += 2
        else
            throw(ArgumentError("Unknown argument: $arg"))
        end
    end

    return opts
end

"""
    resolve_summary_path(opts) -> String

Resolve the release summary TOML path from CLI options or auto-detect
from the latest release in the checkpoint directory for the given architecture.
"""
function resolve_summary_path(opts::Dict{String, String})::String
    if haskey(opts, "summary-path")
        return resolve_path(opts["summary-path"])
    end

    architecture = haskey(opts, "architecture") ? opts["architecture"] : model_architecture_name()
    checkpoint_root = resolve_path(CHECKPOINT_DIR)
    summary_path = latest_release_summary_path(checkpoint_root, architecture)
    release_root = joinpath(checkpoint_root, architecture_slug(architecture), "release")
    summary_path === nothing && throw(ArgumentError("No release summaries found for architecture '$architecture' under $release_root"))
    return summary_path
end

"""
    summarize_release(summary, bundle_dir)

Print a human-readable summary of the release bundle and its metrics.
"""
function summarize_release(summary, bundle_dir::String)
    run = summary["run"]
    metrics = summary["metrics"]
    println("Bundle ready: $bundle_dir")
    println("Architecture: $(run["architecture"]) | release_id: $(run["release_id"]) | commit: $(run["commit_sha"])")
    println("Metrics: last_iter=$(metrics["last_iter"]) | best_selection_score=$(metrics["best_selection_score"]) | baseline_win_rate=$(metrics["baseline_win_rate"])%")
    if haskey(metrics, "final_loss")
        println("Final loss: $(metrics["final_loss"])" )
    end
end

"""
    main(args::Vector{String}=Base.ARGS)

Entry point for the Hugging Face publisher script. Parses CLI arguments,
resolves the release summary, and runs the requested mode: dry-run, stage, or publish.
"""
function main(args::Vector{String}=Base.ARGS)
    opts = parse_args(args)
    opts === nothing && return print_help()

    summary_path = resolve_summary_path(opts)
    summary = read_release_summary(summary_path)
    run = summary["run"]
    if haskey(opts, "architecture") && opts["architecture"] != String(run["architecture"])
        throw(ArgumentError("Requested architecture '$(opts["architecture"])' does not match summary architecture '$(run["architecture"])'"))
    end

    mode = get(opts, "mode", "stage")
    if mode == "dry-run"
        planned = plan_release_bundle(summary_path; root_dir=ROOT_DIR)
        println("Dry run only — no files written.")
        summarize_release(summary, planned.bundle_dir)
        println("Expected repo path: $(default_repo_path(String(run["architecture"]), String(run["release_id"])))")
        return nothing
    elseif mode == "stage"
        bundle_dir = stage_release_bundle(summary_path; root_dir=ROOT_DIR)
        summarize_release(summary, bundle_dir)
        println("Manifest: $(joinpath(bundle_dir, "manifest.toml"))")
        return nothing
    elseif mode == "publish"
        haskey(opts, "repo-id") || throw(ArgumentError("--repo-id is required for publish mode"))
        repo_path = get(opts, "repo-path", default_repo_path(String(run["architecture"]), String(run["release_id"])))
        bundle_dir = publish_release_bundle(summary_path, opts["repo-id"]; repo_path=repo_path, root_dir=ROOT_DIR)
        summarize_release(summary, bundle_dir)
        println("Uploaded to: $(opts["repo-id"]) / $repo_path")
        return nothing
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
