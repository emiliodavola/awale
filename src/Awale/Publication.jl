"""
    Publication

Release bundling and publication: model cards, bundle manifests, artifact staging,
and Hugging Face upload commands.
"""
module Publication

using Dates
using SHA
using TOML

using ..Model: atomic_write, load_model, save_public_model
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
       stage_public_release_bundle,
       publish_release_bundle,
       default_repo_path,
       resolve_repo_path,
       public_release_bundle_dir

const RELEASE_SUBDIR = "release"
const ARTIFACT_SUBDIR = "artifacts"
const MANIFEST_FILE = "manifest.toml"
const RELEASE_SUMMARY_FILE = "release_summary.toml"
const MODEL_CARD_FILE = "README.md"
const MODEL_CARD_GENERATOR_VERSION = 1
const PUBLIC_MODEL_FILE_EXT = ".f32"
const DEFAULT_ROOT_DIR = abspath(joinpath(@__DIR__, "..", ".."))

"""
    release_timestamp(now::DateTime=Dates.now()) -> String

Format the current time (or `now`) as an ISO-8601 timestamp string.
"""
function release_timestamp(now::DateTime=Dates.now())::String
    return Dates.format(now, "yyyy-mm-ddTHH:MM:SS")
end

"""
    release_id_slug(now::DateTime=Dates.now()) -> String

Generate a filesystem-safe release identifier from the current timestamp.
"""
function release_id_slug(now::DateTime=Dates.now())::String
    return Dates.format(now, "yyyymmdd_HHMMSS")
end

"""
    posix_join(parts...) -> String

Join path parts with forward slashes (/), producing a POSIX-style path regardless of OS.
"""
function posix_join(parts...)::String
    return join(String.(parts), "/")
end

"""
    repo_relpath(parts...) -> String

Join path parts into a repository-relative POSIX path.
Convenience wrapper around `posix_join` for semantic clarity.
"""
function repo_relpath(parts...)::String
    return posix_join(parts...)
end

"""
    path_within_root(root_dir::AbstractString, candidate::AbstractString) -> Bool

Check whether `candidate` resolves to a path inside `root_dir`.
Returns `false` if `candidate` escapes the root via `..` components.
"""
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

"""
    resolve_repo_path(root_dir, repo_path) -> String

Resolve a path relative to the repository root, normalizing separators.
Throws `ArgumentError` if the resolved path escapes the workspace root.
"""
function resolve_repo_path(root_dir::AbstractString, repo_path::AbstractString)::String
    normalized = replace(String(repo_path), '\\' => '/')
    root = abspath(String(root_dir))
    candidate = isabspath(normalized) ? abspath(normalized) : abspath(joinpath(root, split(normalized, '/')...))
    path_within_root(root, candidate) || throw(ArgumentError("Path escapes workspace root: $repo_path"))
    return candidate
end

"""
    release_namespace_dir(checkpoint_dir, architecture) -> String

Return the architecture-scoped checkpoint directory under `checkpoint_dir`.
"""
function release_namespace_dir(checkpoint_dir::AbstractString, architecture::AbstractString)::String
    return joinpath(String(checkpoint_dir), architecture_slug(architecture))
end

"""
    release_summary_path(checkpoint_dir, architecture, release_id) -> String

Return the expected path of a release summary file within the architecture-scoped checkpoint tree.
"""
function release_summary_path(checkpoint_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(release_namespace_dir(checkpoint_dir, architecture), RELEASE_SUBDIR, String(release_id), RELEASE_SUMMARY_FILE)
end

"""
    release_bundle_summary_path(checkpoint_dir, release_id) -> String

Return the expected path of a release summary within the flat (non-architecture-scoped) bundle tree.
"""
function release_bundle_summary_path(checkpoint_dir::AbstractString, release_id::AbstractString)::String
    return joinpath(String(checkpoint_dir), RELEASE_SUBDIR, String(release_id), RELEASE_SUMMARY_FILE)
end

"""
    release_model_card_path(bundle_dir) -> String

Return the README.md path inside a release bundle directory.
"""
function release_model_card_path(bundle_dir::AbstractString)::String
    return joinpath(String(bundle_dir), MODEL_CARD_FILE)
end

"""
    write_model_card_front_matter(io::IO, summary::Dict{String, Any})

Write YAML front-matter for a Hugging Face model card to `io`, extracting
metrics and metadata from the release summary.
"""
function write_model_card_front_matter(io::IO, summary::Dict{String, Any})
    sections = release_summary_sections(summary)
    release_id = String(sections.run["release_id"])
    best_selection_score = sections.metrics["best_selection_score"]
    baseline_win_rate = sections.metrics["baseline_win_rate"]
    final_loss = sections.metrics["final_loss"]
    selection_current_best_rate = get(sections.metrics, "selection_current_best_rate", nothing)

    println(io, "---")
    println(io, "license: mit")
    println(io, "library_name: flux")
    println(io, "tags:")
    for tag in ("julia", "flux", "awale", "reinforcement-learning", "mcts")
        println(io, "  - $tag")
    end
    println(io, "model-index:")
    println(io, "  - name: Awale release $release_id")
    println(io, "    results:")
    println(io, "      - task:")
    println(io, "          type: custom")
    println(io, "          name: Awale self-play evaluation")
    println(io, "        dataset:")
    println(io, "          name: Awale release summary")
    println(io, "        metrics:")
    println(io, "          - name: Best selection score")
    println(io, "            type: best_selection_score")
    println(io, "            value: $best_selection_score")
    println(io, "          - name: Baseline win rate")
    println(io, "            type: baseline_win_rate")
    println(io, "            value: $baseline_win_rate")
    println(io, "          - name: Final loss")
    println(io, "            type: final_loss")
    println(io, "            value: $final_loss")
    if selection_current_best_rate !== nothing
        println(io, "          - name: Selection current best rate")
        println(io, "            type: selection_current_best_rate")
        println(io, "            value: $selection_current_best_rate")
    end
    println(io, "---")
    println(io)
end

"""
    public_release_bundle_dir(checkpoint_dir, architecture, release_id) -> String

Return the directory for a public (Float32-exported) release bundle, nested under
the release bundle dir.
"""
function public_release_bundle_dir(checkpoint_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(release_bundle_dir(checkpoint_dir, architecture, release_id), "public")
end

"""
    artifact_label(bundle_relpath) -> String

Derive a human-readable label from a bundle-relative artifact path
for use in the bundle manifest.
"""
function artifact_label(bundle_relpath::AbstractString)::String
    artifact_file = basename(bundle_relpath)
    artifact_file == RELEASE_SUMMARY_FILE && return "release_summary"
    stem, _ = splitext(artifact_file)
    stem == "training_state" && return "training_state"
    stem == "training_config" && return "runtime_config_snapshot"
    stem == "model_config" && return "model_config_snapshot"
    return stem
end

"""
    artifact_checksum(path::AbstractString) -> Dict{String, Any}

Compute SHA-256 and byte-size for a release artifact file.
Throws `ArgumentError` if the file does not exist.
"""
function artifact_checksum(path::AbstractString)::Dict{String, Any}
    isfile(path) || throw(ArgumentError("Missing release artifact: $path"))
    return Dict{String, Any}(
        "sha256" => bytes2hex(sha256(read(path))),
        "bytes" => filesize(path),
    )
end

"""
    artifact_destination_name(artifact_file; public=false) -> String

Return the destination filename for an artifact. When `public=true`, model checkpoint
`.bin` files are renamed with the `.f32` extension for Hugging Face releases.
"""
function artifact_destination_name(artifact_file::AbstractString; public::Bool=false)::String
    public || return artifact_file
    artifact_file == "model_final.bin" && return "model_final$(PUBLIC_MODEL_FILE_EXT)"
    artifact_file == "model_best.bin" && return "model_best$(PUBLIC_MODEL_FILE_EXT)"
    artifact_file == "model_last.bin" && return "model_last$(PUBLIC_MODEL_FILE_EXT)"
    return artifact_file
end

"""
    bundle_artifact_path(bundle_dir, bundle_relpath) -> String

Resolve a bundle-relative POSIX path to an absolute filesystem path.
Splits the relative path on `/` to handle cross-platform join correctly.
"""
function bundle_artifact_path(bundle_dir::AbstractString, bundle_relpath::AbstractString)::String
    return joinpath(String(bundle_dir), split(bundle_relpath, '/')...)
end

"""
    bundle_file_relpath(bundle_dir, file_path) -> String

Compute a bundle-relative POSIX path for an absolute `file_path` inside `bundle_dir`.
"""
function bundle_file_relpath(bundle_dir::AbstractString, file_path::AbstractString)::String
    return replace(relpath(String(file_path), String(bundle_dir)), '\\' => '/')
end

"""
    bundle_file_paths(bundle_dir) -> Set{String}

Return the set of all bundle-relative file paths present in `bundle_dir`.
Returns an empty set if the directory does not exist.
"""
function bundle_file_paths(bundle_dir::AbstractString)::Set{String}
    files = Set{String}()
    isdir(bundle_dir) || return files

    for (root, _, filenames) in walkdir(bundle_dir)
        for filename in filenames
            push!(files, bundle_file_relpath(bundle_dir, joinpath(root, filename)))
        end
    end

    return files
end

"""
    expected_bundle_file_paths(artifact_specs) -> Set{String}

Return the complete set of files that a valid bundle should contain:
the manifest, model card, and all artifact relpaths from the spec.
"""
function expected_bundle_file_paths(artifact_specs::Dict{String, String})::Set{String}
    expected = Set{String}((MANIFEST_FILE, MODEL_CARD_FILE))
    union!(expected, keys(artifact_specs))
    return expected
end

"""
    expected_bundle_manifest_artifacts(artifact_specs) -> Dict{String, String}

Build the `[artifacts]` section for the bundle manifest, mapping
human-readable labels to bundle-relative paths.
"""
function expected_bundle_manifest_artifacts(artifact_specs::Dict{String, String})::Dict{String, String}
    artifact_entries = Dict{String, String}()
    for bundle_relpath in keys(artifact_specs)
        artifact_entries[artifact_label(bundle_relpath)] = bundle_relpath
    end
    artifact_entries["model_card"] = MODEL_CARD_FILE
    return artifact_entries
end

"""
    expected_bundle_integrity_paths(artifact_specs) -> Set{String}

Return the set of paths that must have integrity checksums in the manifest.
"""
function expected_bundle_integrity_paths(artifact_specs::Dict{String, String})::Set{String}
    integrity_paths = Set{String}(keys(artifact_specs))
    push!(integrity_paths, MODEL_CARD_FILE)
    return integrity_paths
end

"""
    dict_entries_match(actual, expected) -> Bool

Check whether `actual` dictionary contains all key–value pairs from `expected`
and has the same length (no extra keys).
"""
function dict_entries_match(actual::AbstractDict, expected::AbstractDict)::Bool
    length(actual) == length(expected) || return false
    for (key, value) in expected
        get(actual, key, nothing) == value || return false
    end
    return true
end

"""
    release_summary_sections(summary::Dict{String, Any}) -> NamedTuple

Split a parsed release summary into `(run, paths, metrics)` named sections
for convenient keyed access.
"""
function release_summary_sections(summary::Dict{String, Any})
    return (run=summary["run"], paths=summary["paths"], metrics=summary["metrics"])
end

"""
    latest_release_summary_path(checkpoint_dir, architecture) -> Union{String, Nothing}

Find the most recent release summary file under the architecture-scoped checkpoint tree.
Returns `nothing` if no releases exist.
"""
function latest_release_summary_path(checkpoint_dir::AbstractString, architecture::AbstractString)::Union{String, Nothing}
    release_root = joinpath(release_namespace_dir(checkpoint_dir, architecture), RELEASE_SUBDIR)
    isdir(release_root) || return nothing

    release_ids = filter(release_id -> isdir(joinpath(release_root, release_id)) && isfile(joinpath(release_root, release_id, RELEASE_SUMMARY_FILE)), readdir(release_root))
    isempty(release_ids) && return nothing

    sort!(release_ids)
    return joinpath(release_root, release_ids[end], RELEASE_SUMMARY_FILE)
end

"""
    release_bundle_dir(checkpoint_dir, architecture, release_id) -> String

Return the directory for a release bundle, validating that the checkpoint dir
is already architecture-scoped.
Throws `ArgumentError` if the basename of `checkpoint_dir` does not match the architecture slug.
"""
function release_bundle_dir(checkpoint_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    namespace = architecture_slug(architecture)
    basename(String(checkpoint_dir)) == namespace || throw(ArgumentError("Release bundle checkpoint dir must already be architecture-scoped: $checkpoint_dir"))
    return joinpath(String(checkpoint_dir), RELEASE_SUBDIR, String(release_id))
end

"""
    expected_release_checkpoint_dir(root_dir, architecture) -> String

Return the canonical checkpoint directory for a given architecture.
"""
function expected_release_checkpoint_dir(root_dir::AbstractString, architecture::AbstractString)::String
    return resolve_repo_path(root_dir, joinpath("checkpoints", architecture_slug(architecture)))
end

"""
    expected_release_artifact_paths(checkpoint_dir, architecture, release_id) -> Dict{String, String}

Build a dictionary of expected absolute paths for all release artifacts
(config snapshots, training state, model checkpoints).
"""
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

"""
    runtime_config_snapshot_path(log_dir, architecture, release_id) -> String

Return the expected path for a runtime configuration snapshot file.
"""
function runtime_config_snapshot_path(log_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(String(log_dir), "training_config_$(architecture_slug(architecture))_$(release_id).toml")
end

"""
    model_config_snapshot_path(log_dir, architecture, release_id) -> String

Return the expected path for a model configuration snapshot file.
"""
function model_config_snapshot_path(log_dir::AbstractString, architecture::AbstractString, release_id::AbstractString)::String
    return joinpath(String(log_dir), "model_config_$(architecture_slug(architecture))_$(release_id).toml")
end

"""
    write_release_summary(path; commit_sha, architecture, release_id, timestamp, checkpoint_dir, kwargs...) -> String

Write a release summary TOML file at `path` with run metadata, artifact paths, and training metrics.
"""
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

"""
    read_release_summary(path::AbstractString) -> Dict{String, Any}

Parse a release summary TOML file and return its contents.
Throws `ArgumentError` if the file does not exist.
"""
function read_release_summary(path::AbstractString)::Dict{String, Any}
    isfile(path) || throw(ArgumentError("Missing release summary: $path"))
    return TOML.parsefile(path)
end

"""
    release_model_card(summary, artifact_specs; bundle_kind, model_export_format) -> String

Generate the full text of a Hugging Face model card (README.md) from a release summary
and artifact specifications.
"""
function release_model_card(summary::Dict{String, Any}, artifact_specs::Dict{String, String}; bundle_kind::AbstractString, model_export_format::AbstractString)::String
    sections = release_summary_sections(summary)
    release_id = String(sections.run["release_id"])
    architecture = String(sections.run["architecture"])
    commit_sha = String(sections.run["commit_sha"])
    timestamp = String(sections.run["timestamp"])
    checkpoint_dir = String(sections.run["checkpoint_dir"])
    runtime_config_snapshot = String(sections.paths["runtime_config_snapshot"])
    model_config_snapshot = String(sections.paths["model_config_snapshot"])
    training_state_path = String(sections.paths["training_state_path"])
    last_checkpoint_path = String(sections.paths["last_checkpoint_path"])
    best_checkpoint_path = String(sections.paths["best_checkpoint_path"])
    final_checkpoint_path = String(sections.paths["final_checkpoint_path"])
    last_iter = sections.metrics["last_iter"]
    best_selection_score = sections.metrics["best_selection_score"]
    baseline_win_rate = sections.metrics["baseline_win_rate"]
    final_loss = sections.metrics["final_loss"]
    selection_current_best_rate = get(sections.metrics, "selection_current_best_rate", nothing)
    selection_promoted = get(sections.metrics, "selection_promoted", nothing)

    io = IOBuffer()
    write_model_card_front_matter(io, summary)
    println(io, "# Awale release $release_id model card")
    println(io)
    println(io, "This model card documents an Awale policy/value network implemented in Julia with Flux.jl. The YAML metadata above comes from the release summary and should be treated as the source of truth for this bundle.")
    println(io)
    println(io, "## Release metadata")
    println(io, "- Architecture: $architecture")
    println(io, "- Release ID: $release_id")
    println(io, "- Commit SHA: $commit_sha")
    println(io, "- Timestamp: $timestamp")
    println(io, "- Checkpoint dir: $checkpoint_dir")
    println(io, "- Bundle kind: $(bundle_kind)")
    println(io, "- Model export format: $(model_export_format)")
    println(io)
    println(io, "## Metrics")
    println(io, "- Last iteration: $last_iter")
    println(io, "- Best selection score: $best_selection_score")
    println(io, "- Baseline win rate: $baseline_win_rate")
    println(io, "- Final loss: $final_loss")
    if selection_current_best_rate !== nothing
        println(io, "- Selection current best rate: $selection_current_best_rate")
    end
    if selection_promoted !== nothing
        println(io, "- Selection promoted: $selection_promoted")
    end
    println(io)
    println(io, "## Source paths")
    println(io, "- Runtime config snapshot: $runtime_config_snapshot")
    println(io, "- Model config snapshot: $model_config_snapshot")
    println(io, "- Training state: $training_state_path")
    println(io, "- Last checkpoint: $last_checkpoint_path")
    println(io, "- Best checkpoint: $best_checkpoint_path")
    println(io, "- Final checkpoint: $final_checkpoint_path")
    println(io)
    println(io, "## Bundle contents")
    println(io, "- `$(RELEASE_SUMMARY_FILE)`")
    println(io, "- `$(MANIFEST_FILE)`")
    println(io, "- `$(MODEL_CARD_FILE)`")
    for bundle_relpath in sort!(collect(keys(artifact_specs)))
        println(io, "- `$(bundle_relpath)`")
    end

    return String(take!(io))
end

"""
    write_release_model_card(bundle_dir, summary, artifact_specs; bundle_kind, model_export_format) -> String

Atomically write a model card README.md to `bundle_dir` using the release summary.
"""
function write_release_model_card(bundle_dir::AbstractString, summary::Dict{String, Any}, artifact_specs::Dict{String, String}; bundle_kind::AbstractString, model_export_format::AbstractString)::String
    path = release_model_card_path(bundle_dir)
    atomic_write(path) do io
        write(io, release_model_card(summary, artifact_specs; bundle_kind=bundle_kind, model_export_format=model_export_format))
    end
    return path
end

"""
    required_release_keys(summary::Dict{String, Any})

Validate that a parsed release summary contains all required sections and keys
([run], [paths], [metrics] and their mandatory fields).
Throws `ArgumentError` for any missing key.
"""
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

"""
    bundle_artifact_specs(summary, root_dir, summary_path, checkpoint_dir; public=false) -> Dict{String, String}

Build a dictionary mapping bundle-relative artifact paths to their source (absolute) paths.
Validates that the summary paths match the expected layout.
When `public=true`, model checkpoints are renamed with `.f32` extension.
"""
function bundle_artifact_specs(summary::Dict{String, Any}, root_dir::AbstractString, summary_path::AbstractString, checkpoint_dir::AbstractString; public::Bool=false)
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

    model_final = artifact_destination_name("model_final.bin"; public=public)
    model_best = artifact_destination_name("model_best.bin"; public=public)
    model_last = artifact_destination_name("model_last.bin"; public=public)

    return Dict{String, String}(
        repo_relpath("release_summary.toml") => summary_source,
        repo_relpath(ARTIFACT_SUBDIR, model_final) => expected_paths["final_checkpoint_path"],
        repo_relpath(ARTIFACT_SUBDIR, model_best) => expected_paths["best_checkpoint_path"],
        repo_relpath(ARTIFACT_SUBDIR, model_last) => expected_paths["last_checkpoint_path"],
        repo_relpath(ARTIFACT_SUBDIR, "training_state.toml") => expected_paths["training_state_path"],
        repo_relpath(ARTIFACT_SUBDIR, "training_config.toml") => expected_paths["runtime_config_snapshot"],
        repo_relpath(ARTIFACT_SUBDIR, "model_config.toml") => expected_paths["model_config_snapshot"],
    )
end

"""
    bundle_manifest(summary, artifact_specs, bundle_dir; bundle_kind, model_export_format) -> Dict{String, Any}

Build the full bundle manifest dictionary including run metadata, artifact labels,
and integrity checksums (SHA-256 + byte size) for every artifact.
"""
function bundle_manifest(summary::Dict{String, Any}, artifact_specs::Dict{String, String}, bundle_dir::AbstractString; bundle_kind::AbstractString, model_export_format::AbstractString)
    sections = release_summary_sections(summary)

    artifact_entries = expected_bundle_manifest_artifacts(artifact_specs)
    integrity_entries = Dict{String, Any}()
    for bundle_relpath in expected_bundle_integrity_paths(artifact_specs)
        integrity_entries[bundle_relpath] = artifact_checksum(bundle_artifact_path(bundle_dir, bundle_relpath))
    end

    return Dict{String, Any}(
        "manifest_version" => 1,
        "model_card_generator_version" => MODEL_CARD_GENERATOR_VERSION,
        "bundle_kind" => String(bundle_kind),
        "model_export_format" => String(model_export_format),
        "run" => Dict{String, Any}(
            "commit_sha" => String(sections.run["commit_sha"]),
            "architecture" => String(sections.run["architecture"]),
            "release_id" => String(sections.run["release_id"]),
            "timestamp" => String(sections.run["timestamp"]),
            "checkpoint_dir" => String(sections.run["checkpoint_dir"]),
        ),
        "source_paths" => Dict{String, Any}(
            "runtime_config_snapshot" => String(sections.paths["runtime_config_snapshot"]),
            "model_config_snapshot" => String(sections.paths["model_config_snapshot"]),
            "training_state_path" => String(sections.paths["training_state_path"]),
            "last_checkpoint_path" => String(sections.paths["last_checkpoint_path"]),
            "best_checkpoint_path" => String(sections.paths["best_checkpoint_path"]),
            "final_checkpoint_path" => String(sections.paths["final_checkpoint_path"]),
        ),
        "metrics" => Dict{String, Any}(pairs(sections.metrics)...),
        "artifacts" => artifact_entries,
        "integrity" => integrity_entries,
    )
end

"""
    copy_artifact!(source, destination) -> destination

Copy a release artifact file from `source` to `destination`, creating parent directories.
Throws `ArgumentError` if the source file does not exist.
Returns immediately (no-op) if source and destination are the same file.
"""
function copy_artifact!(source::AbstractString, destination::AbstractString)
    isfile(source) || throw(ArgumentError("Missing release artifact: $source"))
    abspath(source) == abspath(destination) && return destination
    mkpath(dirname(destination))
    cp(source, destination; force=true)
    return destination
end

"""
    stage_release_artifact!(source_path, destination_path; public=false) -> destination

Stage a release artifact: when `public=true` and the destination ends with `.f32`,
the model is exported via `save_public_model` (Float32 weights only). Otherwise,
the file is copied as-is.
"""
function stage_release_artifact!(source_path::AbstractString, destination_path::AbstractString; public::Bool=false)
    public && endswith(destination_path, PUBLIC_MODEL_FILE_EXT) && return save_public_model(load_model(source_path), destination_path)
    return copy_artifact!(source_path, destination_path)
end

"""
    bundle_is_valid(bundle_dir, artifact_specs; bundle_kind, model_export_format) -> Bool

Check whether an existing bundle directory is valid: contains the expected files,
has a parseable manifest with matching bundle_kind and model_export_format, correct artifact
labels, and integrity checksums that match the on-disk files.
"""
function bundle_is_valid(bundle_dir::AbstractString, artifact_specs::Dict{String, String}; bundle_kind::AbstractString, model_export_format::AbstractString)::Bool
    manifest_path = joinpath(String(bundle_dir), MANIFEST_FILE)
    isfile(manifest_path) || return false
    isfile(release_model_card_path(bundle_dir)) || return false
    bundle_file_paths(bundle_dir) == expected_bundle_file_paths(artifact_specs) || return false

    manifest = TOML.parsefile(manifest_path)
    bundle_kind_entry = get(manifest, "bundle_kind", nothing)
    bundle_kind_entry isa AbstractString || return false
    String(bundle_kind_entry) == bundle_kind || return false

    export_format_entry = get(manifest, "model_export_format", nothing)
    export_format_entry isa AbstractString || return false
    String(export_format_entry) == model_export_format || return false

    generator_version_entry = get(manifest, "model_card_generator_version", nothing)
    generator_version_entry isa Integer || return false
    Int(generator_version_entry) == MODEL_CARD_GENERATOR_VERSION || return false

    artifacts = get(manifest, "artifacts", nothing)
    artifacts isa Dict{String, Any} || return false

    dict_entries_match(artifacts, expected_bundle_manifest_artifacts(artifact_specs)) || return false

    integrity = get(manifest, "integrity", nothing)
    integrity isa Dict{String, Any} || return false

    expected_integrity_paths = expected_bundle_integrity_paths(artifact_specs)
    length(integrity) == length(expected_integrity_paths) || return false

    for bundle_relpath in expected_integrity_paths
        entry = get(integrity, bundle_relpath, nothing)
        entry isa Dict{String, Any} || return false
        artifact_file = bundle_artifact_path(bundle_dir, bundle_relpath)
        get(entry, "sha256", nothing) == bytes2hex(sha256(read(artifact_file))) || return false
        get(entry, "bytes", nothing) == filesize(artifact_file) || return false
    end

    return true
end

"""
    reset_bundle_dir!(bundle_dir) -> bundle_dir

Remove and recreate a bundle directory, ensuring a clean slate for staging.
"""
function reset_bundle_dir!(bundle_dir::AbstractString)
    ispath(bundle_dir) && rm(bundle_dir; force=true, recursive=true)
    mkpath(bundle_dir)
    return bundle_dir
end

"""
    stage_bundle_artifacts(bundle_dir, artifact_specs; public=false) -> bundle_dir

Copy all artifacts from their source paths into the bundle directory.
When `public=true`, model files are exported as Float32 weight vectors.
"""
function stage_bundle_artifacts(bundle_dir::AbstractString, artifact_specs::Dict{String, String}; public::Bool=false)
    for (bundle_relpath, source_path) in artifact_specs
        destination = bundle_artifact_path(bundle_dir, bundle_relpath)
        stage_release_artifact!(source_path, destination; public=public)
    end
    return bundle_dir
end

"""
    write_release_bundle(bundle_dir, summary, artifact_specs; bundle_kind, model_export_format) -> bundle_dir

Write the model card and manifest TOML into a prepared bundle directory.
"""
function write_release_bundle(bundle_dir::AbstractString, summary::Dict{String, Any}, artifact_specs::Dict{String, String}; bundle_kind::AbstractString, model_export_format::AbstractString)
    write_release_model_card(bundle_dir, summary, artifact_specs; bundle_kind=bundle_kind, model_export_format=model_export_format)
    atomic_write(joinpath(bundle_dir, MANIFEST_FILE)) do io
        TOML.print(io, bundle_manifest(summary, artifact_specs, bundle_dir; bundle_kind=bundle_kind, model_export_format=model_export_format))
    end
    return bundle_dir
end

"""
    plan_release_bundle(summary_path; root_dir) -> (bundle_dir, artifact_specs, summary)

Read and validate a release summary, resolve all artifact paths against the repo root,
and return the planned bundle layout without writing anything. All source artifacts
are verified to exist on disk.
"""
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

"""
    stage_release_bundle(summary_path; root_dir) -> bundle_dir

Build (or return cached) a local-trusted release bundle: reads the summary,
validates artifact paths, resets the bundle dir, stages all artifacts, and
writes the model card + manifest. Skips rebuilding if the bundle is already valid.
"""
function stage_release_bundle(summary_path::AbstractString; root_dir::AbstractString=DEFAULT_ROOT_DIR)
    planned = plan_release_bundle(summary_path; root_dir=root_dir)
    bundle_dir = planned.bundle_dir
    summary = planned.summary

    bundle_is_valid(bundle_dir, planned.artifact_specs; bundle_kind="local_trusted", model_export_format="serialization") && return bundle_dir

    reset_bundle_dir!(bundle_dir)
    run = summary["run"]
    paths = summary["paths"]
    metrics = summary["metrics"]
    write_release_summary(
        joinpath(bundle_dir, RELEASE_SUMMARY_FILE);
        commit_sha=String(run["commit_sha"]),
        architecture=String(run["architecture"]),
        release_id=String(run["release_id"]),
        timestamp=String(run["timestamp"]),
        checkpoint_dir=String(run["checkpoint_dir"]),
        runtime_config_snapshot=String(paths["runtime_config_snapshot"]),
        model_config_snapshot=String(paths["model_config_snapshot"]),
        training_state_path=String(paths["training_state_path"]),
        last_checkpoint_path=String(paths["last_checkpoint_path"]),
        best_checkpoint_path=String(paths["best_checkpoint_path"]),
        final_checkpoint_path=String(paths["final_checkpoint_path"]),
        last_iter=Int(metrics["last_iter"]),
        best_selection_score=metrics["best_selection_score"],
        baseline_win_rate=metrics["baseline_win_rate"],
        final_loss=metrics["final_loss"],
        selection_current_best_rate=get(metrics, "selection_current_best_rate", nothing),
        selection_promoted=get(metrics, "selection_promoted", nothing),
    )
    stage_bundle_artifacts(bundle_dir, planned.artifact_specs; public=false)
    return write_release_bundle(bundle_dir, summary, planned.artifact_specs; bundle_kind="local_trusted", model_export_format="serialization")
end

"""
    stage_public_release_bundle(summary_path; root_dir) -> bundle_dir

Build (or return cached) a public-safe release bundle with Float32-exported weights.
"""
function stage_public_release_bundle(summary_path::AbstractString; root_dir::AbstractString=DEFAULT_ROOT_DIR)
    planned = plan_release_bundle(summary_path; root_dir=root_dir)
    run = planned.summary["run"]
    checkpoint_dir = resolve_repo_path(root_dir, String(run["checkpoint_dir"]))
    bundle_dir = public_release_bundle_dir(checkpoint_dir, String(run["architecture"]), String(run["release_id"]))
    artifact_specs = bundle_artifact_specs(planned.summary, root_dir, summary_path, checkpoint_dir; public=true)

    bundle_is_valid(bundle_dir, artifact_specs; bundle_kind="public_safe", model_export_format="float32") && return bundle_dir

    reset_bundle_dir!(bundle_dir)
    stage_bundle_artifacts(bundle_dir, artifact_specs; public=true)
    return write_release_bundle(bundle_dir, planned.summary, artifact_specs; bundle_kind="public_safe", model_export_format="float32")
end

"""
    default_repo_path(architecture, release_id) -> String

Return the default remote repository path for a release on Hugging Face.
"""
function default_repo_path(architecture::AbstractString, release_id::AbstractString)::String
    return repo_relpath("releases", architecture_slug(architecture), release_id)
end

"""
    hf_upload_command(repo_id, local_path, repo_path) -> Cmd

Build the `hf upload` command for uploading a file or directory to Hugging Face.
"""
function hf_upload_command(repo_id::AbstractString, local_path::AbstractString, repo_path::AbstractString)
    return `hf upload $repo_id $local_path $repo_path`
end

"""
    publish_model_card_upload_target(bundle_dir) -> NamedTuple

Return the local path and remote repo path for the model card README.md upload.
"""
function publish_model_card_upload_target(bundle_dir::AbstractString)
    return (local_path=joinpath(String(bundle_dir), MODEL_CARD_FILE), repo_path=MODEL_CARD_FILE)
end

"""
    publish_model_card_command(repo_id, bundle_dir) -> Cmd

Build the `hf upload` command to publish only the model card README.md.
"""
function publish_model_card_command(repo_id::AbstractString, bundle_dir::AbstractString)
    target = publish_model_card_upload_target(bundle_dir)
    return hf_upload_command(repo_id, target.local_path, target.repo_path)
end

"""
    publish_release_bundle(summary_path, repo_id; repo_path, root_dir, upload_runner) -> bundle_dir

Full release pipeline: stage the public release bundle, upload the model card,
then upload the full bundle to a Hugging Face repo.
`upload_runner` defaults to `run` and can be replaced for dry-run or testing.
"""
function publish_release_bundle(summary_path::AbstractString, repo_id::AbstractString; repo_path::Union{Nothing, AbstractString}=nothing, root_dir::AbstractString=DEFAULT_ROOT_DIR, upload_runner::Function=run)
    summary = read_release_summary(summary_path)
    required_release_keys(summary)
    bundle_dir = stage_public_release_bundle(summary_path; root_dir=root_dir)
    run_info = summary["run"]
    remote_path = repo_path === nothing ? default_repo_path(String(run_info["architecture"]), String(run_info["release_id"])) : String(repo_path)
    upload_runner(publish_model_card_command(repo_id, bundle_dir))
    upload_runner(hf_upload_command(repo_id, bundle_dir, remote_path))
    return bundle_dir
end

end # module
