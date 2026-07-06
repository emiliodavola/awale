using Test
using .Awale

@testset "state model" begin
    s = Awale.initial_state()
    @test Awale.validate_invariants(s)

    s_can = Awale.canonicalize(s)
    @test s_can.to_move == 1
    @test Awale.hash_state(s_can) == Awale.hash_state(Awale.canonicalize(s_can))

    bytes = Awale.serialize_state(s_can)
    s2 = Awale.deserialize_state(bytes)
    @test Awale.serialize_state(s2) == bytes
    @test Awale.validate_invariants(s2)
    @test !Awale.is_terminal(s2)
end
