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

    for (label, path) in artifact_paths
        write(path, "artifact:$label")
    end

    runtime_snapshot = joinpath(log_dir, "training_config_mlp_$(release_id).toml")
    model_snapshot = joinpath(log_dir, "model_config_mlp_$(release_id).toml")
    write(runtime_snapshot, "training = true\n")
    write(model_snapshot, "model = true\n")

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
            @test Awale.Publication.publish_model_card_upload_target(bundle_dir) == (joinpath(bundle_dir, "README.md"), "README.md")
            @test manifest["metrics"]["baseline_win_rate"] == 71.0
            @test occursin("# Awale release 20260719_120000 model card", model_card)
            @test occursin("Architecture: mlp", model_card)
            @test occursin("Commit SHA: abc123", model_card)
            @test occursin("Best selection score: 62.5", model_card)
            @test Awale.Publication.default_repo_path("mlp", "20260719_120000") == "releases/mlp/20260719_120000"
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
