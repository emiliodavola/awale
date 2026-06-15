<<<<<<< HEAD
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

=======
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

>>>>>>> origin/dev
end # module