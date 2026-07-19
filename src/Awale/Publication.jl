module Publication

using Dates
using TOML

using ..Model: atomic_write
using ..Utils: architecture_slug

export release_id_slug,
       release_timestamp,
       release_summary_path,
       latest_release_summary_path,
       release_bundle_dir,
       runtime_config_snapshot_path,
       model_config_snapshot_path,
       plan_release_bundle,
       write_release_summary,
       read_release_summary,
       stage_release_bundle,
       publish_release_bundle,
       default_repo_path,
       resolve_repo_path

const RELEASE_SUBDIR = "release"
const ARTIFACT_SUBDIR = "artifacts"
const MANIFEST_FILE = "manifest.toml"
const RELEASE_SUMMARY_FILE = "release_summary.toml"
const DEFAULT_ROOT_DIR = abspath(joinpath(@__DIR__, "..", ".."))

function release_timestamp(now::DateTime=Dates.now())::String
    return Dates.format(now, "yyyy-mm-ddTHH:MM:SS")
end

function release_id_slug(now::DateTime=Dates.now())::String
    return Dates.format(now, "yyyymmdd_HHMMSS")
end

function posix_join(parts...)::String
    return join(String.(parts), "/")
end

function repo_relpath(parts...)::String
    return posix_join(parts...)
end

function path_within_root(root_dir::AbstractString, candidate::AbstractString)::Bool
    root = abspath(String(root_dir))
    resolved = abspath(String(candidate))
    relpath_value = try
        relpath(resolved, root)
    catch
        return false
    end

    parts = splitpath(relpath_value)
    return isempty(parts) || first(parts) != ".."
end

function resolve_repo_path(root_dir::AbstractString, repo_path::AbstractString)::String
    normalized = replace(String(repo_path), '\\' => '/')
    root = abspath(String(root_dir))
    candidate = isabspath(normalized) ? abspath(normalized) : abspath(joinpath(root, split(normalized, '/')...))
    path_within_root(root, candidate) || throw(ArgumentError("Path escapes workspace root: $repo_path"))
    return candidate
end

function release_namespace_dir(checkpoint_dir::AbstractString, architecture::AbstractString)::String
    return joinpath(String(checkpoint_dir), architecture_slug(architecture))
end

function release_summary_path(checkpoint_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(release_namespace_dir(checkpoint_dir, architecture), RELEASE_SUBDIR, String(release_id), RELEASE_SUMMARY_FILE)
end

function release_bundle_summary_path(checkpoint_dir::AbstractString, release_id::AbstractString)::String
    return joinpath(String(checkpoint_dir), RELEASE_SUBDIR, String(release_id), RELEASE_SUMMARY_FILE)
end

function latest_release_summary_path(checkpoint_dir::AbstractString, architecture::AbstractString)::Union{String, Nothing}
    release_root = joinpath(release_namespace_dir(checkpoint_dir, architecture), RELEASE_SUBDIR)
    isdir(release_root) || return nothing

    release_ids = filter(release_id -> isdir(joinpath(release_root, release_id)) && isfile(joinpath(release_root, release_id, RELEASE_SUMMARY_FILE)), readdir(release_root))
    isempty(release_ids) && return nothing

    sort!(release_ids)
    return joinpath(release_root, release_ids[end], RELEASE_SUMMARY_FILE)
end

function release_bundle_dir(checkpoint_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    namespace = architecture_slug(architecture)
    basename(String(checkpoint_dir)) == namespace || throw(ArgumentError("Release bundle checkpoint dir must already be architecture-scoped: $checkpoint_dir"))
    return joinpath(String(checkpoint_dir), RELEASE_SUBDIR, String(release_id))
end

function expected_release_checkpoint_dir(root_dir::AbstractString, architecture::AbstractString)::String
    return resolve_repo_path(root_dir, joinpath("checkpoints", architecture_slug(architecture)))
end

function expected_release_artifact_paths(checkpoint_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)
    arch = architecture_slug(architecture)
    return Dict{String, String}(
        "runtime_config_snapshot" => joinpath(checkpoint_dir, "log", "training_config_$(arch)_$(release_id).toml"),
        "model_config_snapshot" => joinpath(checkpoint_dir, "log", "model_config_$(arch)_$(release_id).toml"),
        "training_state_path" => joinpath(checkpoint_dir, "training_state.toml"),
        "last_checkpoint_path" => joinpath(checkpoint_dir, "model_last.bin"),
        "best_checkpoint_path" => joinpath(checkpoint_dir, "model_best.bin"),
        "final_checkpoint_path" => joinpath(checkpoint_dir, "model_final.bin"),
    )
end

function runtime_config_snapshot_path(log_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(String(log_dir), "training_config_$(architecture_slug(architecture))_$(release_id).toml")
end

function model_config_snapshot_path(log_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(String(log_dir), "model_config_$(architecture_slug(architecture))_$(release_id).toml")
end

function write_release_summary(
    path::AbstractString;
    commit_sha::AbstractString,
    architecture::AbstractString,
    release_id::AbstractString,
    timestamp::AbstractString,
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
    summary = Dict{String, Any}(
        "release_version" => 1,
        "run" => Dict{String, Any}(
            "commit_sha" => String(commit_sha),
            "architecture" => String(architecture),
            "release_id" => String(release_id),
            "timestamp" => String(timestamp),
            "checkpoint_dir" => String(checkpoint_dir),
        ),
        "paths" => Dict{String, Any}(
            "runtime_config_snapshot" => String(runtime_config_snapshot),
            "model_config_snapshot" => String(model_config_snapshot),
            "training_state_path" => String(training_state_path),
            "last_checkpoint_path" => String(last_checkpoint_path),
            "best_checkpoint_path" => String(best_checkpoint_path),
            "final_checkpoint_path" => String(final_checkpoint_path),
        ),
        "metrics" => Dict{String, Any}(
            "last_iter" => Int(last_iter),
            "best_selection_score" => Float64(best_selection_score),
            "baseline_win_rate" => Float64(baseline_win_rate),
            "final_loss" => Float64(final_loss),
        ),
    )

    if selection_current_best_rate !== nothing
        summary["metrics"]["selection_current_best_rate"] = Float64(selection_current_best_rate)
    end
    if selection_promoted !== nothing
        summary["metrics"]["selection_promoted"] = Bool(selection_promoted)
    end

    atomic_write(path) do io
        TOML.print(io, summary)
    end

    return path
end

function read_release_summary(path::AbstractString)::Dict{String, Any}
    isfile(path) || throw(ArgumentError("Missing release summary: $path"))
    return TOML.parsefile(path)
end

function required_release_keys(summary::Dict{String, Any})
    haskey(summary, "run") || throw(ArgumentError("Release summary missing [run]"))
    haskey(summary, "paths") || throw(ArgumentError("Release summary missing [paths]"))
    haskey(summary, "metrics") || throw(ArgumentError("Release summary missing [metrics]"))

    run = summary["run"]
    paths = summary["paths"]
    metrics = summary["metrics"]

    for key in ("commit_sha", "architecture", "release_id", "timestamp", "checkpoint_dir")
        haskey(run, key) || throw(ArgumentError("Release summary missing run.$key"))
    end

    for key in ("runtime_config_snapshot", "model_config_snapshot", "training_state_path", "last_checkpoint_path", "best_checkpoint_path", "final_checkpoint_path")
        haskey(paths, key) || throw(ArgumentError("Release summary missing paths.$key"))
    end

    for key in ("last_iter", "best_selection_score", "baseline_win_rate", "final_loss")
        haskey(metrics, key) || throw(ArgumentError("Release summary missing metrics.$key"))
    end

    return nothing
end

function bundle_artifact_specs(summary::Dict{String, Any}, root_dir::AbstractString, summary_path::AbstractString, checkpoint_dir::AbstractString)
    run = summary["run"]
    paths = summary["paths"]
    architecture = architecture_slug(String(run["architecture"]))
    release_id = String(run["release_id"])
    summary_source = resolve_repo_path(root_dir, summary_path)
    expected_summary_source = release_bundle_summary_path(checkpoint_dir, release_id)
    summary_source == expected_summary_source || throw(ArgumentError("Release summary path does not match expected layout: $summary_source"))

    expected_paths = expected_release_artifact_paths(checkpoint_dir, architecture, release_id)

    for (key, expected_path) in expected_paths
        actual_path = resolve_repo_path(root_dir, String(paths[key]))
        actual_path == expected_path || throw(ArgumentError("Release summary path does not match expected layout: $key"))
    end

    return Dict{String, String}(
        repo_relpath("release_summary.toml") => summary_source,
        repo_relpath(ARTIFACT_SUBDIR, "model_final.bin") => expected_paths["final_checkpoint_path"],
        repo_relpath(ARTIFACT_SUBDIR, "model_best.bin") => expected_paths["best_checkpoint_path"],
        repo_relpath(ARTIFACT_SUBDIR, "model_last.bin") => expected_paths["last_checkpoint_path"],
        repo_relpath(ARTIFACT_SUBDIR, "training_state.toml") => expected_paths["training_state_path"],
        repo_relpath(ARTIFACT_SUBDIR, "training_config.toml") => expected_paths["runtime_config_snapshot"],
        repo_relpath(ARTIFACT_SUBDIR, "model_config.toml") => expected_paths["model_config_snapshot"],
    )
end

function bundle_manifest(summary::Dict{String, Any}, artifact_specs::Dict{String, String})
    run = summary["run"]
    paths = summary["paths"]
    metrics = summary["metrics"]

    artifact_entries = Dict{String, Any}()
    for bundle_relpath in keys(artifact_specs)
        artifact_file = basename(bundle_relpath)
        artifact_label = if artifact_file == RELEASE_SUMMARY_FILE
            "release_summary"
        elseif artifact_file == "model_final.bin"
            "model_final"
        elseif artifact_file == "model_best.bin"
            "model_best"
        elseif artifact_file == "model_last.bin"
            "model_last"
        elseif artifact_file == "training_state.toml"
            "training_state"
        elseif artifact_file == "training_config.toml"
            "runtime_config_snapshot"
        elseif artifact_file == "model_config.toml"
            "model_config_snapshot"
        else
            replace(artifact_file, "." => "_")
        end
        artifact_entries[artifact_label] = bundle_relpath
    end

    return Dict{String, Any}(
        "manifest_version" => 1,
        "run" => Dict{String, Any}(
            "commit_sha" => String(run["commit_sha"]),
            "architecture" => String(run["architecture"]),
            "release_id" => String(run["release_id"]),
            "timestamp" => String(run["timestamp"]),
            "checkpoint_dir" => String(run["checkpoint_dir"]),
        ),
        "source_paths" => Dict{String, Any}(
            "runtime_config_snapshot" => String(paths["runtime_config_snapshot"]),
            "model_config_snapshot" => String(paths["model_config_snapshot"]),
            "training_state_path" => String(paths["training_state_path"]),
            "last_checkpoint_path" => String(paths["last_checkpoint_path"]),
            "best_checkpoint_path" => String(paths["best_checkpoint_path"]),
            "final_checkpoint_path" => String(paths["final_checkpoint_path"]),
        ),
        "metrics" => Dict{String, Any}(pairs(metrics)...),
        "artifacts" => artifact_entries,
    )
end

function copy_artifact!(source::AbstractString, destination::AbstractString)
    isfile(source) || throw(ArgumentError("Missing release artifact: $source"))
    abspath(source) == abspath(destination) && return destination
    mkpath(dirname(destination))
    cp(source, destination; force=true)
    return destination
end

function plan_release_bundle(summary_path::AbstractString; root_dir::AbstractString=DEFAULT_ROOT_DIR)
    summary = read_release_summary(summary_path)
    required_release_keys(summary)

    run = summary["run"]
    release_id = String(run["release_id"])
    architecture = String(run["architecture"])
    checkpoint_dir = resolve_repo_path(root_dir, String(run["checkpoint_dir"]))
    expected_checkpoint_dir = expected_release_checkpoint_dir(root_dir, architecture)
    checkpoint_dir == expected_checkpoint_dir || throw(ArgumentError("Release summary checkpoint_dir does not match expected layout: $checkpoint_dir"))
    bundle_dir = release_bundle_dir(expected_checkpoint_dir, architecture, release_id)
    artifact_specs = bundle_artifact_specs(summary, root_dir, summary_path, expected_checkpoint_dir)

    for (bundle_relpath, source_path) in artifact_specs
        isfile(source_path) || throw(ArgumentError("Missing release artifact: $source_path (for $bundle_relpath)"))
    end

    return (bundle_dir=bundle_dir, artifact_specs=artifact_specs, summary=summary)
end

function stage_release_bundle(summary_path::AbstractString; root_dir::AbstractString=DEFAULT_ROOT_DIR)
    planned = plan_release_bundle(summary_path; root_dir=root_dir)
    bundle_dir = planned.bundle_dir
    summary = planned.summary
    manifest_path = joinpath(bundle_dir, MANIFEST_FILE)

    if isfile(manifest_path)
        return bundle_dir
    end

    mkpath(bundle_dir)

    for (bundle_relpath, source_path) in planned.artifact_specs
        copy_artifact!(source_path, joinpath(bundle_dir, split(bundle_relpath, '/')...))
    end

    atomic_write(manifest_path) do io
        TOML.print(io, bundle_manifest(summary, planned.artifact_specs))
    end

    return bundle_dir
end

function default_repo_path(architecture::AbstractString, release_id::AbstractString)::String
    return repo_relpath("releases", architecture_slug(architecture), release_id)
end

function hf_upload_command(repo_id::AbstractString, local_path::AbstractString, repo_path::AbstractString)
    return `hf upload $repo_id $local_path $repo_path`
end

function publish_release_bundle(summary_path::AbstractString, repo_id::AbstractString; repo_path::Union{Nothing, AbstractString}=nothing, root_dir::AbstractString=DEFAULT_ROOT_DIR)
    summary = read_release_summary(summary_path)
    required_release_keys(summary)
    bundle_dir = stage_release_bundle(summary_path; root_dir=root_dir)
    run_info = summary["run"]
    remote_path = repo_path === nothing ? default_repo_path(String(run_info["architecture"]), String(run_info["release_id"])) : String(repo_path)
    run(hf_upload_command(repo_id, bundle_dir, remote_path))
    return bundle_dir
end

end # module
