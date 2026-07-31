using Test
using TOML
using .Awale

function seed_release_inputs(root_dir::AbstractString; checkpoint_root_relpath::AbstractString="checkpoints", release_id::AbstractString="20260719_120000")
    checkpoint_dir = joinpath(root_dir, checkpoint_root_relpath)
    arch_dir = joinpath(checkpoint_dir, "mlp")
    log_dir = joinpath(arch_dir, "log")
    release_dir = joinpath(arch_dir, "release", release_id)
    mkpath(release_dir)
    mkpath(log_dir)

    artifact_paths = Dict(
        "model_final.bin" => joinpath(arch_dir, "model_final.bin"),
        "model_best.bin" => joinpath(arch_dir, "model_best.bin"),
        "model_last.bin" => joinpath(arch_dir, "model_last.bin"),
        "training_state.toml" => joinpath(arch_dir, "training_state.toml"),
    )

    model = Awale.create_model()
    for (label, path) in artifact_paths
        endswith(label, ".bin") && Awale.Model.save_model(model, path)
    end
    write(artifact_paths["training_state.toml"], "resume_contract = \"weights-only\"\nlast_iter = 300\n")

    runtime_snapshot = joinpath(log_dir, "training_config_mlp_$(release_id).toml")
    model_snapshot = joinpath(log_dir, "model_config_mlp_$(release_id).toml")
    write(runtime_snapshot, "training = true\n")
    write(model_snapshot, read(joinpath(@__DIR__, "..", "src", "Awale", "config.toml"), String))

    summary_path = Awale.Publication.release_summary_path(checkpoint_dir, "mlp", release_id)
    Awale.Publication.write_release_summary(
        summary_path;
        commit_sha="abc123",
        architecture="mlp",
        release_id=release_id,
        timestamp="2026-07-19T12:00:00",
        checkpoint_dir=joinpath(checkpoint_root_relpath, "mlp"),
        runtime_config_snapshot=joinpath(checkpoint_root_relpath, "mlp", "log", "training_config_mlp_$(release_id).toml"),
        model_config_snapshot=joinpath(checkpoint_root_relpath, "mlp", "log", "model_config_mlp_$(release_id).toml"),
        training_state_path=joinpath(checkpoint_root_relpath, "mlp", "training_state.toml"),
        last_checkpoint_path=joinpath(checkpoint_root_relpath, "mlp", "model_last.bin"),
        best_checkpoint_path=joinpath(checkpoint_root_relpath, "mlp", "model_best.bin"),
        final_checkpoint_path=joinpath(checkpoint_root_relpath, "mlp", "model_final.bin"),
        last_iter=300,
        best_selection_score=62.5,
        baseline_win_rate=71.0,
        final_loss=0.42,
        selection_current_best_rate=64.0,
        selection_promoted=true,
    )

    return summary_path
end

@testset "Hugging Face publication flow" begin
    @testset "release summary round-trips and bundles cleanly" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)

            summary = Awale.Publication.read_release_summary(summary_path)
            @test summary["run"]["architecture"] == "mlp"
            @test summary["metrics"]["best_selection_score"] == 62.5
            @test Awale.Publication.latest_release_summary_path(joinpath(root_dir, "checkpoints"), "mlp") == summary_path

            bundle_dir = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            manifest_path = joinpath(bundle_dir, "manifest.toml")
            manifest = TOML.parsefile(manifest_path)

            @test bundle_dir == joinpath(root_dir, "checkpoints", "mlp", "release", "20260719_120000")
            @test manifest["bundle_kind"] == "local_trusted"
            @test manifest["model_export_format"] == "serialization"
            @test isfile(joinpath(bundle_dir, "release_summary.toml"))
            @test isfile(joinpath(bundle_dir, "README.md"))
            @test isfile(joinpath(bundle_dir, "artifacts", "model_final.bin"))
            @test isfile(joinpath(bundle_dir, "artifacts", "training_state.toml"))
            model_card = read(joinpath(bundle_dir, "README.md"), String)
            @test manifest["run"]["release_id"] == "20260719_120000"
            @test manifest["artifacts"]["release_summary"] == "release_summary.toml"
            @test manifest["artifacts"]["model_final"] == "artifacts/model_final.bin"
            @test manifest["artifacts"]["training_state"] == "artifacts/training_state.toml"
            @test manifest["artifacts"]["model_card"] == "README.md"
            @test manifest["model_card_generator_version"] == Awale.Publication.MODEL_CARD_GENERATOR_VERSION
            @test haskey(manifest, "integrity")
            @test haskey(manifest["integrity"], "artifacts/model_final.bin")
            target = Awale.Publication.publish_model_card_upload_target(bundle_dir)
            @test target.local_path == joinpath(bundle_dir, "README.md")
            @test target.repo_path == "README.md"
            @test manifest["metrics"]["baseline_win_rate"] == 71.0
            @test startswith(model_card, "---\n")
            @test occursin("license: mit", model_card)
            @test occursin("library_name: flux", model_card)
            @test occursin("  - julia", model_card)
            @test occursin("  - awale", model_card)
            @test occursin("model-index:", model_card)
            @test occursin("Awale self-play evaluation", model_card)
            @test occursin("Awale release summary", model_card)
            @test occursin("This model card documents an Awale policy/value network implemented in Julia with Flux.jl.", model_card)
            @test occursin("# Awale release 20260719_120000 model card", model_card)
            @test occursin("Architecture: mlp", model_card)
            @test occursin("Bundle kind: local_trusted", model_card)
            @test occursin("Model export format: serialization", model_card)
            @test occursin("Commit SHA: abc123", model_card)
            @test occursin("Best selection score: 62.5", model_card)
            @test !occursin("Transformers", model_card)
            @test occursin("## Code", model_card)
            @test occursin("github.com/emiliodavola/awale", model_card)
            @test Awale.Publication.default_repo_path("mlp", "20260719_120000") == "releases/mlp/20260719_120000"
        end
    end

    @testset "staged bundles are rebuilt when expected artifacts disappear" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            bundle_dir = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            rm(joinpath(bundle_dir, "artifacts", "model_final.bin"))

            restaged = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            @test restaged == bundle_dir
            @test isfile(joinpath(bundle_dir, "artifacts", "model_final.bin"))
        end
    end

    @testset "staged bundles are rebuilt when the model-card generator version changes" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            planned = Awale.Publication.plan_release_bundle(summary_path; root_dir=root_dir)
            bundle_dir = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            manifest_path = joinpath(bundle_dir, "manifest.toml")
            manifest = TOML.parsefile(manifest_path)
            manifest["model_card_generator_version"] = Awale.Publication.MODEL_CARD_GENERATOR_VERSION + 1
            open(manifest_path, "w") do io
                TOML.print(io, manifest)
            end

            @test !Awale.Publication.bundle_is_valid(bundle_dir, planned.artifact_specs; bundle_kind="local_trusted", model_export_format="serialization")
            restaged = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            @test restaged == bundle_dir
            @test TOML.parsefile(manifest_path)["model_card_generator_version"] == Awale.Publication.MODEL_CARD_GENERATOR_VERSION
        end
    end

    @testset "staged bundles are rebuilt when the model card changes" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            planned = Awale.Publication.plan_release_bundle(summary_path; root_dir=root_dir)
            bundle_dir = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            readme_path = joinpath(bundle_dir, "README.md")
            original_readme = read(readme_path, String)

            write(readme_path, "stale model card\n")

            @test !Awale.Publication.bundle_is_valid(bundle_dir, planned.artifact_specs; bundle_kind="local_trusted", model_export_format="serialization")
            restaged = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            @test restaged == bundle_dir
            @test read(readme_path, String) == original_readme
        end
    end

    @testset "local trusted bundles restage cleanly when a stray file appears" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            bundle_dir = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            stray_path = joinpath(bundle_dir, "artifacts", "stray.txt")
            write(stray_path, "leftover")

            restaged = Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
            @test restaged == bundle_dir
            @test !isfile(stray_path)
            @test isfile(joinpath(bundle_dir, "artifacts", "model_final.bin"))
            @test isfile(joinpath(bundle_dir, "manifest.toml"))
        end
    end

    @testset "public release bundle exports safe float payloads" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            public_bundle_dir = Awale.Publication.stage_public_release_bundle(summary_path; root_dir=root_dir)
            manifest = TOML.parsefile(joinpath(public_bundle_dir, "manifest.toml"))

            @test public_bundle_dir == joinpath(root_dir, "checkpoints", "mlp", "release", "20260719_120000", "public")
            @test manifest["bundle_kind"] == "public_safe"
            @test manifest["model_export_format"] == "float32"
            @test isfile(joinpath(public_bundle_dir, "README.md"))
            @test isfile(joinpath(public_bundle_dir, "artifacts", "model_final.f32"))
            @test !isfile(joinpath(public_bundle_dir, "artifacts", "model_final.bin"))
            @test manifest["artifacts"]["model_final"] == "artifacts/model_final.f32"
            @test manifest["artifacts"]["model_card"] == "README.md"
            @test all(!endswith(String(path), ".bin") for path in values(manifest["artifacts"]))

            local_model = Awale.Model.load_model(joinpath(root_dir, "checkpoints", "mlp", "model_final.bin"))
            public_model = Awale.Model.load_public_model(joinpath(public_bundle_dir, "artifacts", "model_final.f32"))
            @test Awale.predict(public_model, Awale.initial_state()) == Awale.predict(local_model, Awale.initial_state())
            @test occursin("Bundle kind: public_safe", read(joinpath(public_bundle_dir, "README.md"), String))
            @test occursin("Model export format: float32", read(joinpath(public_bundle_dir, "README.md"), String))
        end
    end

    @testset "public bundles restage cleanly when a stray file appears" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            public_bundle_dir = Awale.Publication.stage_public_release_bundle(summary_path; root_dir=root_dir)
            stray_path = joinpath(public_bundle_dir, "artifacts", "stray.txt")
            write(stray_path, "leftover")

            restaged = Awale.Publication.stage_public_release_bundle(summary_path; root_dir=root_dir)
            @test restaged == public_bundle_dir
            @test !isfile(stray_path)
            @test isfile(joinpath(public_bundle_dir, "artifacts", "model_final.f32"))
            @test isfile(joinpath(public_bundle_dir, "manifest.toml"))
        end
    end

    @testset "publish flow uploads the model card before the bundle" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            commands = Cmd[]
            upload_runner(cmd) = (push!(commands, cmd); nothing)

            bundle_dir = Awale.Publication.publish_release_bundle(summary_path, "user/repo"; root_dir=root_dir, upload_runner=upload_runner)
            expected_model_card = Awale.Publication.publish_model_card_command("user/repo", bundle_dir)
            expected_bundle = Awale.Publication.hf_upload_command("user/repo", bundle_dir, Awale.Publication.default_repo_path("mlp", "20260719_120000"))

            @test commands == [expected_model_card, expected_bundle]
        end
    end

    @testset "latest release summary wins when multiple runs exist" begin
        mktempdir() do root_dir
            older = seed_release_inputs(root_dir; release_id="20260719_120000")
            newer = seed_release_inputs(root_dir; release_id="20260720_090000")

            @test older != newer
            @test Awale.Publication.latest_release_summary_path(joinpath(root_dir, "checkpoints"), "mlp") == newer
            @test Awale.Publication.stage_release_bundle(newer; root_dir=root_dir) == joinpath(root_dir, "checkpoints", "mlp", "release", "20260720_090000")
        end
    end

    @testset "missing artifacts fail fast" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir)
            rm(joinpath(root_dir, "checkpoints", "mlp", "model_final.bin"))

            @test_throws ArgumentError Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
        end
    end

    @testset "summary paths stay rooted under checkpoints" begin
        mktempdir() do root_dir
            summary_path = seed_release_inputs(root_dir; checkpoint_root_relpath="notes")

            @test_throws ArgumentError Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)

            summary_path = seed_release_inputs(root_dir)
            Awale.Publication.write_release_summary(
                summary_path;
                commit_sha="abc123",
                architecture="mlp",
                release_id="20260719_120000",
                timestamp="2026-07-19T12:00:00",
                checkpoint_dir=joinpath("checkpoints", "mlp"),
                runtime_config_snapshot=joinpath("checkpoints", "mlp", "log", "training_config_mlp_20260719_120000.toml"),
                model_config_snapshot=joinpath("checkpoints", "mlp", "log", "model_config_mlp_20260719_120000.toml"),
                training_state_path=joinpath("..", "escape.toml"),
                last_checkpoint_path=joinpath("checkpoints", "mlp", "model_last.bin"),
                best_checkpoint_path=joinpath("checkpoints", "mlp", "model_best.bin"),
                final_checkpoint_path=joinpath("checkpoints", "mlp", "model_final.bin"),
                last_iter=300,
                best_selection_score=62.5,
                baseline_win_rate=71.0,
                final_loss=0.42,
                selection_current_best_rate=64.0,
                selection_promoted=true,
            )

            @test_throws ArgumentError Awale.Publication.stage_release_bundle(summary_path; root_dir=root_dir)
        end
    end
end

@testset "model card helper functions" begin
    @testset "format_metric rounds to 2-4 significant digits and strips trailing zeros" begin
        @test Awale.Publication.format_metric(61.702127659574465) == "61.7"
        @test Awale.Publication.format_metric(71.0) == "71"
        @test Awale.Publication.format_metric(0.42) == "0.42"
        @test Awale.Publication.format_metric(62.5) == "62.5"
        @test Awale.Publication.format_metric(64.0) == "64"
        @test Awale.Publication.format_metric(1.23456) == "1.235"
        @test Awale.Publication.format_metric(0.5) == "0.5"
        @test Awale.Publication.format_metric(300) == "300"
    end

    @testset "read_bundle_configs parses bundle config TOMLs defensively" begin
        mktempdir() do root_dir
            empty_bundle = joinpath(root_dir, "empty")
            mkpath(empty_bundle)
            training_cfg, model_cfg = Awale.Publication.read_bundle_configs(empty_bundle)
            @test training_cfg == Dict{String,Any}()
            @test model_cfg == Dict{String,Any}()
        end

        mktempdir() do root_dir
            bad_bundle = joinpath(root_dir, "bad")
            mkpath(joinpath(bad_bundle, "artifacts"))
            write(joinpath(bad_bundle, "artifacts", "training_config.toml"), "not = [valid")
            training_cfg, model_cfg = Awale.Publication.read_bundle_configs(bad_bundle)
            @test training_cfg == Dict{String,Any}()
            @test model_cfg == Dict{String,Any}()
        end

        mktempdir() do root_dir
            bundle_dir = Awale.Publication.stage_release_bundle(seed_release_inputs(root_dir); root_dir=root_dir)
            training_cfg, model_cfg = Awale.Publication.read_bundle_configs(bundle_dir)
            @test training_cfg == Dict{String,Any}("training" => true)
            @test model_cfg["model"]["architecture"] == "mlp"
            @test length(model_cfg["model"]["variants"]["mlp"]["layers"]["shared"]) == 2
        end
    end

    @testset "model_parameter_count sums Dense and Conv parameters" begin
        config = TOML.parsefile(joinpath(@__DIR__, "..", "src", "Awale", "config.toml"))
        @test Awale.Publication.model_parameter_count(config) == 31559

        flat = Dict{String,Any}(
            "layers" => Dict{String,Any}(
                "shared" => [Dict{String,Any}("type" => "Dense", "in" => 48, "out" => 128)],
                "policy" => [Dict{String,Any}("type" => "Dense", "in" => 128, "out" => 6)],
                "value" => [Dict{String,Any}("type" => "Dense", "in" => 128, "out" => 1)],
            ),
        )
        @test Awale.Publication.model_parameter_count(flat) == 48 * 128 + 128 + 128 * 6 + 6 + 128 + 1

        conv = Dict{String,Any}(
            "layers" => Dict{String,Any}(
                "shared" => [
                    Dict{String,Any}("type" => "Reshape", "shape" => [4, 12, 1]),
                    Dict{String,Any}("type" => "Conv", "kernel" => [3, 3], "in" => 1, "out" => 8),
                    Dict{String,Any}("type" => "MaxPool", "size" => [2, 2]),
                    Dict{String,Any}("type" => "Flatten"),
                ],
                "policy" => [Dict{String,Any}("type" => "Dense", "in" => 16, "out" => 6)],
                "value" => [Dict{String,Any}("type" => "Dense", "in" => 16, "out" => 1)],
            ),
        )
        @test Awale.Publication.model_parameter_count(conv) == 3 * 3 * 1 * 8 + 8 + 16 * 6 + 6 + 16 + 1

        @test Awale.Publication.model_parameter_count(Dict{String,Any}()) == nothing
        @test Awale.Publication.model_parameter_count(
            Dict{String,Any}(
                "model" => Dict{String,Any}(
                    "variants" => Dict{String,Any}("mlp" => Dict{String,Any}()),
                ),
            ),
        ) == nothing
    end

    @testset "public_model_parameter_count derives parameter counts from the bundle" begin
        mktempdir() do root_dir
            bundle_dir = joinpath(root_dir, "bundle")
            mkpath(joinpath(bundle_dir, "artifacts"))
            write(joinpath(bundle_dir, "artifacts", "model_best.f32"), zeros(UInt8, 126236))
            @test Awale.Publication.public_model_parameter_count(bundle_dir; model_export_format="float32") == 31559

            write(joinpath(bundle_dir, "artifacts", "model_best.f32"), zeros(UInt8, 10))
            @test Awale.Publication.public_model_parameter_count(bundle_dir; model_export_format="float32") == 2

            cp(
                joinpath(@__DIR__, "..", "src", "Awale", "config.toml"),
                joinpath(bundle_dir, "artifacts", "model_config.toml"),
            )
            @test Awale.Publication.public_model_parameter_count(bundle_dir; model_export_format="serialization") == 31559

            rm(joinpath(bundle_dir, "artifacts", "model_best.f32"))
            @test Awale.Publication.public_model_parameter_count(bundle_dir; model_export_format="float32") == nothing

            rm(joinpath(bundle_dir, "artifacts", "model_config.toml"))
            @test Awale.Publication.public_model_parameter_count(bundle_dir; model_export_format="serialization") == nothing
        end
    end
end
