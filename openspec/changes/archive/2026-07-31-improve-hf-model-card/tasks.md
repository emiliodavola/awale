# Tasks: Improve Hugging Face Model Card Quality

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~550-650 (src ~330-380, tests ~220-270) |
| 400-line budget risk | High (default guard); Low vs session budget 1200 |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 → PR 2 → PR 3 |
| Delivery strategy | auto-forecast |
| Chain strategy | pending |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: pending
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Pure helpers + unit tests (T1-T3) | PR 1 | Base: feature branch from dev; additive, no behavior change |
| 2 | Template rewrite + wiring + version bump + assertion updates (T4-T7) | PR 2 | Base: PR 1 branch; heaviest slice (~380-420 lines) |
| 3 | Extra test coverage (T8-T9) | PR 3 | Base: PR 2 branch |

## Phase 1: Foundation — pure helpers (test-first)

- [x] T1 (TDD) `format_metric(x::Real)::String`: round sigdigits=4, strip trailing zeros. Files: `src/Awale/Publication.jl`, `test/test_publication_flow.jl`. Deps: none. AC: 61.702127659574465→"61.7"; 71.0→"71"; 0.42→"0.42".
- [x] T2 (TDD) `read_bundle_configs(bundle_dir)::Tuple{Dict,Dict}`: parse `artifacts/training_config.toml` + `model_config.toml`; try/catch → empty dicts. Deps: T1. AC: missing files → empty dicts, no throw; present files parsed.
- [x] T3 (TDD) `model_parameter_count(model_config)` (Dense `in*out+out`, Conv `prod(kernel)*in*out+out`, resolve variant) + `public_model_parameter_count(bundle_dir; model_export_format)` (float32 → `filesize(model_best.f32)÷4`; else config; else nothing). Deps: T2. AC: `src/Awale/config.toml` MLP → 31559; float32 → bytes÷4.

## Phase 2: Core — template rewrite

- [x] T4 Caller-side config extraction: `write_release_bundle` reads configs + param count, passes `training_config`/`model_config`/`model_params` kwargs through `write_release_model_card` → `release_model_card` (pure render; defaults keep old releases working). Files: Publication.jl. Deps: T3. AC: no IO in render; old-render fallbacks intact.
- [x] T5 Front matter rewrite: stable model name in `model-index`; tags + `alphazero`, `self-play`, `board-game`; values via `format_metric`; per-metric `description`. Deps: T1, T4. AC: YAML tags/rounded values/descriptions present.
- [x] T6 Body rewrite + version bump: title pinned `# Awale AlphaZero-like`; section helpers `card_model_details`, `card_usage`, `card_training_details`, `card_evaluation` (RandomAgent baseline, 400 sims, 100 games, 56%-over-200 gate), `card_limitations`, `card_bundle_contents` (dedup: sorted `artifact_specs` keys + `manifest.toml`/`README.md` if absent, per-file descriptions), `card_code_section` (repo links last), `card_citation`; per-checkpoint promotion narrative from optional `selection_promoted` (never a global flag); `## Release` section (ID, SHA, timestamp, bundle kind, export format; no `checkpoint_dir`); drop `## Source paths`; `MODEL_CARD_GENERATOR_VERSION` 2→3. Deps: T4, T5. AC: spec sections; `!occursin("Source paths")`; no global flag; no local paths; dedup; version 3.

## Phase 3: Tests — coupling + coverage

- [x] T7 Update `test/test_publication_flow.jl` L90-107 (stable title, all sections, `!occursin("Source paths")`, `!occursin("Selection promoted")`, no `checkpoints\` path, `## Code` links with repo last, `Release ID:` present) and L194-195 (Bundle kind / Model export format via `## Release`). Deps: T6. AC: full flow testset green.
- [x] T8 New integration tests: dedup `count("release_summary.toml", card) == 1`; body/YAML rounding agreement (61.702… → "61.7" both); promotion `true`/`false` variants (per-checkpoint wording, no global flag); defensive render (empty config dicts, missing optional metrics → fallbacks). Deps: T6. AC: all pass.
- [x] T9 E2E restage: manifest missing `model_card_generator_version` → `bundle_is_valid == false` → restaged records version 3. Deps: T6. AC: testset green.

## Phase 4: Verify + commits

- [x] T10 Verify: `julia --project=. test/test_publication_flow.jl` + JuliaFormatter; create branch from `dev`; commit by work unit per PR slices (work-unit-commits skill). Deps: all. AC: tests green; commits map to PR slices. *(Reconciled at archive-time: verified inline per orchestrator — 780/780 tests green, JuliaFormatter clean, all 3 PRs merged.)*
