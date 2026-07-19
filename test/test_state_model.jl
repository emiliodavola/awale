using Test
using .Awale

@testset "state model" begin
    s = Awale.initial_state()
    @test Awale.validate_invariants(s)

    s_can = Awale.canonicalize(s)
    encoded = Awale.encode_state(s_can)
    s_mirror = Awale.GameState(s.board, Int8(2), s.captured, s.history_hash, s.config, Set{UInt64}())

    @test s_can.to_move == 1
    @test Awale.hash_state(s_can) == Awale.hash_state(Awale.canonicalize(s_can))
    @test size(encoded) == (4, 12)
    @test all(0.0f0 .<= encoded) && all(encoded .<= 1.0f0)
    @test encoded == Awale.encode_state(Awale.canonicalize(s_mirror))

    bytes = Awale.serialize_state(s_can)
    s2 = Awale.deserialize_state(bytes)
    @test Awale.serialize_state(s2) == bytes
    @test Awale.validate_invariants(s2)
    @test !Awale.is_terminal(s2)

    legacy = Awale.initial_state(Awale.GameConfig(forced_feeding=:allow_move_feeding))
    legacy_bytes = Awale.serialize_state(legacy)
    legacy_s2 = Awale.deserialize_state(legacy_bytes)
    @test legacy_s2.config.forced_feeding == :allow_move_feeding
end
