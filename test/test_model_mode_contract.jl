using Test
using Flux
using .Awale

@testset "prediction mode contract" begin
    mktempdir() do tmpdir
        config_path = joinpath(tmpdir, "config.toml")
        write(config_path, """
        [model]
        architecture = "mlp"

        [[model.layers.shared]]
        type = "Dense"
        in = 48
        out = 48
        activation = "relu"

        [[model.layers.shared]]
        type = "Dropout"
        rate = 0.5

        [[model.layers.shared]]
        type = "Dense"
        in = 48
        out = 16
        activation = "relu"

        [[model.layers.policy]]
        type = "Dense"
        in = 16
        out = 6
        activation = "identity"

        [[model.layers.value]]
        type = "Dense"
        in = 16
        out = 1
        activation = "tanh"
        """)

        model = Awale.create_model(config_path)
        state = Awale.initial_state()
        next_state = Awale.transition(state, 1)
        x_single = reshape(vec(Awale.encode_state(Awale.canonicalize(state))), :, 1)
        x_batch = hcat(
            vec(Awale.encode_state(Awale.canonicalize(state))),
            vec(Awale.encode_state(Awale.canonicalize(next_state))),
        )

        Flux.trainmode!(model)
        @test model.shared.layers[2].active
        Flux.testmode!(model)
        expected_logits, expected_value = Awale.predict_raw(model, x_single)
        expected_batch_logits, expected_batch_values = Awale.predict_raw(model, x_batch)
        @test !model.shared.layers[2].active
        Flux.trainmode!(model)

        helper_logits, helper_value = Awale.predict_inference(model, state)
        helper_batch_logits, helper_batch_values = Awale.predict_batch_inference(model, [state, next_state])

        @test model.shared.layers[2].active
        @test helper_logits ≈ vec(expected_logits)
        @test helper_value ≈ expected_value[1]
        @test helper_batch_logits ≈ expected_batch_logits
        @test helper_batch_values ≈ expected_batch_values

        Flux.trainmode!(model)
        train_logits, train_value = Awale.predict(model, state)
        train_batch_logits, train_batch_values = Awale.predict_batch(model, [state, next_state])

        @test model.shared.layers[2].active
        @test length(train_logits) == 6
        @test isfinite(train_value)
        @test size(train_batch_logits) == (6, 2)
        @test size(train_batch_values) == (1, 2)

        Flux.testmode!(model)
        test_logits, test_value = Awale.predict(model, state)
        test_batch_logits, test_batch_values = Awale.predict_batch(model, [state, next_state])

        @test !model.shared.layers[2].active
        @test length(test_logits) == 6
        @test isfinite(test_value)
        @test size(test_batch_logits) == (6, 2)
        @test size(test_batch_values) == (1, 2)
    end
end
