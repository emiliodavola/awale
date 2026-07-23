"""
    Utils

Deterministic helpers: hashing, formatting, and checkpoint path resolution.
"""
module Utils

export fnv1a64, bytes_to_hex, architecture_slug, checkpoint_fragment, architecture_scoped_path, architecture_scoped_candidates, first_existing_path

using Printf

const FNV_offset = UInt64(0xcbf29ce484222325)
const FNV_prime = UInt64(0x00000100000001B3)

"""
    fnv1a64(bytes::Vector{UInt8}) -> UInt64

FNV-1a 64-bit non-cryptographic hash for deterministic hashing across platforms.
"""
function fnv1a64(bytes::Vector{UInt8})::UInt64
    h = FNV_offset
    for b in bytes
        h = xor(h, UInt64(b))
        h = UInt64((UInt128(h) * UInt128(FNV_prime)) & UInt128(0xffffffffffffffff))
    end
    return h
end

"""
    bytes_to_hex(bytes::Vector{UInt8}) -> String

Format bytes as a lowercase hex string (e.g. "a1b2c3").
"""
function bytes_to_hex(bytes::Vector{UInt8})::String
    join(map(x->@sprintf("%02x", x), bytes), "")
end

"""
    architecture_slug(value::AbstractString) -> String

Convert an architecture name to a filesystem-safe slug (lowercase, underscores).
"""
function architecture_slug(value::AbstractString)::String
    slug = lowercase(strip(String(value)))
    slug = replace(slug, r"[^a-z0-9._-]+" => "_")
    slug = replace(slug, r"_+" => "_")
    slug = strip(slug, '_')
    return isempty(slug) ? "unknown" : slug
end

"""
    checkpoint_fragment(configured_path::AbstractString, default_filename::AbstractString) -> String

Extract the filename portion from a configured checkpoint path,
falling back to `default_filename` if empty.
"""
function checkpoint_fragment(configured_path::AbstractString, default_filename::AbstractString)::String
    configured = strip(String(configured_path))
    default_fragment = basename(String(default_filename))

    isempty(configured) && return default_fragment
    return basename(configured)
end

"""
    architecture_scoped_path(base_dir, architecture, configured_path, default_filename) -> String

Resolve a checkpoint path under an architecture-scoped subdirectory.
"""
function architecture_scoped_path(base_dir::AbstractString, architecture::AbstractString, configured_path::AbstractString, default_filename::AbstractString)::String
    configured = strip(String(configured_path))
    if isabspath(configured)
        return configured
    end

    fragment = checkpoint_fragment(configured, default_filename)
    return joinpath(String(base_dir), architecture_slug(architecture), fragment)
end

"""
    architecture_scoped_candidates(base_dir, architecture, configured_path, default_filename) -> Vector{String}

Return ordered candidate paths for checkpoint lookup, checking the
architecture-scoped path first, then the legacy flat path.
"""
function architecture_scoped_candidates(base_dir::AbstractString, architecture::AbstractString, configured_path::AbstractString, default_filename::AbstractString)::Vector{String}
    configured = strip(String(configured_path))
    if isabspath(configured)
        return String[configured]
    end

    fragment = checkpoint_fragment(configured, default_filename)
    namespaced = joinpath(String(base_dir), architecture_slug(architecture), fragment)
    legacy = joinpath(String(base_dir), fragment)
    return namespaced == legacy ? String[namespaced] : String[namespaced, legacy]
end

"""
    first_existing_path(candidates) -> Union{String, Nothing}

Return the first path from `candidates` that exists on disk, or `nothing`.
"""
function first_existing_path(candidates)::Union{String, Nothing}
    for candidate in candidates
        isfile(candidate) && return String(candidate)
    end

    return nothing
end

end # module