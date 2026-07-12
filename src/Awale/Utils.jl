module Utils

export fnv1a64, bytes_to_hex

using Printf

const FNV_offset = UInt64(0xcbf29ce484222325)
const FNV_prime = UInt64(0x00000100000001B3)

# FNV-1a 64-bit hash for deterministic hashing across platforms
function fnv1a64(bytes::Vector{UInt8})::UInt64
    h = FNV_offset
    for b in bytes
        h = xor(h, UInt64(b))
        h = UInt64((UInt128(h) * UInt128(FNV_prime)) & UInt128(0xffffffffffffffff))
    end
    return h
end

function bytes_to_hex(bytes::Vector{UInt8})::String
    join(map(x->@sprintf("%02x", x), bytes), "")
end

function architecture_slug(value::AbstractString)::String
    slug = lowercase(strip(String(value)))
    slug = replace(slug, r"[^a-z0-9._-]+" => "_")
    slug = replace(slug, r"_+" => "_")
    slug = strip(slug, '_')
    return isempty(slug) ? "unknown" : slug
end

function checkpoint_fragment(configured_path::AbstractString, default_filename::AbstractString)::String
    configured = strip(String(configured_path))
    default_fragment = basename(String(default_filename))

    isempty(configured) && return default_fragment
    return basename(configured)
end

function architecture_scoped_path(base_dir::AbstractString, architecture::AbstractString, configured_path::AbstractString, default_filename::AbstractString)::String
    configured = strip(String(configured_path))
    if isabspath(configured)
        return configured
    end

    fragment = checkpoint_fragment(configured, default_filename)
    return joinpath(String(base_dir), architecture_slug(architecture), fragment)
end

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

function first_existing_path(candidates)::Union{String, Nothing}
    for candidate in candidates
        isfile(candidate) && return String(candidate)
    end

    return nothing
end

end # module
