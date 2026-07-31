# Exploration: improve-hf-model-card

Issue: GitHub #63 — the published HF model card (`emiliodavola/awale-alphazero-like`) reads as a raw release-summary dump. The card is **generated**; the fix point is the generator, not the Hub README (next publish overwrites it).

## Current State

### Card generation (Q1)

The entire card is produced by **two functions** in `src/Awale/Publication.jl`:

- `release_model_card(summary, artifact_specs; bundle_kind, model_export_format)` (L587–668) — builds the body in one `IOBuffer`: title `# Awale release $release_id model card` (L615), intro sentence (L617–620), `## Release metadata` (L622–630), `## Metrics` (L631–641, raw values), `## Source paths` (L643–650, leaked internal paths), `## Bundle contents` (L651–657), `## Code` (L660–665, dead bullets).
- `write_model_card_front_matter(io, summary)` (L177–219) — YAML: `license: mit`, `library_name: flux`, 5 tags, `model-index` with unrounded metric values and context-free custom metric types.

Data flow: `train.jl` → `write_release_summary` (TOML) → `plan_release_bundle` (validates via `required_release_keys`) → `stage_public_release_bundle` → `stage_bundle_artifacts` (exports `.f32`, copies config TOMLs into the bundle **before** the card is written) → `write_release_bundle` → `write_release_model_card` → `release_model_card`. Publish uploads `README.md` to the repo **root**, then the bundle to `releases/<arch>/<release_id>/`.

**Duplicated `release_summary.toml` root cause**: L652 prints it explicitly, then the L655 loop iterates `artifact_specs` keys, which already contain `release_summary.toml` (added in `bundle_artifact_specs`, L776).

**Minimal change set**: `release_model_card` + `write_model_card_front_matter` (plus optionally a rounding helper and a config-extraction helper). Bump `MODEL_CARD_GENERATOR_VERSION` 2→3 to force restage of cached bundles (gate mechanism already exists and is spec'd).

### Data available at render time (Q2)

- **Release summary** (the `summary` arg): `run{commit_sha, architecture, release_id, timestamp, checkpoint_dir}`, `paths{...}`, `metrics{last_iter, best_selection_score, baseline_win_rate, final_loss, selection_current_best_rate?, selection_promoted?}`.
- **`artifacts/training_config.toml`** (staged before card write; readable via `artifact_specs` source paths): iterations (550), games/iteration (200), sims/move (300), learning_rate, batch_size, updates/iteration, replay settings, temperature_moves, seeds; `[evaluation]` eval_games (100), sims_per_eval (400); `[selection]` target_sims (400), promotion_games (200), promotion_threshold (56.0), opening suite, anchor flags; `[mcts]` c_puct (2.5), dirichlet_alpha (0.3), dirichlet_epsilon (0.25).
- **`artifacts/model_config.toml`**: full layer spec → topology description. Parameter count also derivable as `filesize(.f32)/4` — verified: 126236 B = 31559 params (MLP), matches hand-computed layer math.
- **Static but verifiable**: optimizer is `Flux.Adam(lr)` (hardcoded, `train.jl:680`); baseline is `RandomAgent` (hardcoded, `train.jl:805` — `baseline_win_rate` = win rate vs Random over `eval_games` at `sims_per_eval`); selection gate = promote if decided win rate vs current best ≥ `promotion_threshold` over `promotion_games` at `target_sims` (`train.jl:812–825`); framework = Julia ≥1.8 + Flux 0.16.10 (`Project.toml`); license MIT; encoding = `encode_state` → 4×12 Float32 (48 features), outputs = 6 policy logits + tanh value (`State.jl:249`, `Model.jl`).

**Feasible with existing data**: all sections proposed in #63 — title/description, Model Details (incl. param count), Usage, Training Details, Evaluation table (baseline identity + game count + MCTS budget + gate semantics), Limitations, deduplicated Bundle Contents with descriptions, Source/Reproducibility links, Citation/License, rounded metrics, extra YAML tags.

**Requires new plumbing (out-of-scope candidates)**: embedded learning-curve/Elo charts (CSVs live in `checkpoints/mlp/log/`, are not staged, and the project has no plotting dependency); hardware info (captured nowhere); Elo ratings in the card (EloTracker exists but final Elo never reaches the release summary; `promotion_history.toml` is not bundled); which iteration `model_best` came from; win rate vs heuristic anchor (printed only); optimizer name in machine-readable form (hardcoded in `train.jl`, not in config).

### Test coupling and blast radius (Q3)

`test/test_publication_flow.jl` (10 testsets). Card-text assertions are concentrated in:

- Testset 1 (L90–107): front-matter strings (`license: mit`, `library_name: flux`, tag bullets, `model-index:`, `Awale self-play evaluation`, `Awale release summary`), exact intro sentence, exact title `# Awale release 20260719_120000 model card`, `Architecture: mlp`, `Bundle kind: local_trusted`, `Model export format: serialization`, `Commit SHA: abc123`, `Best selection score: 62.5`, `!occursin("Transformers")`, `## Code`, repo URL.
- Public-bundle testset (L194–195): `Bundle kind: public_safe`, `Model export format: float32`.

Everything else is structural (manifest, integrity checksums, restage semantics) and is **unaffected** by a card rewrite as long as `README.md` is still written and checksummed. `release_model_card` is internal (single caller: `write_release_model_card`); no other source file consumes card text. **Blast radius: 1 source file + 1 test file + 1 spec delta.**

### Spec constraints (Q4)

- `spec/` (00–11) has **no** model-card requirements — only release-flow mentions in `07_training` and `08_evaluation`. No constraint from there.
- `openspec/specs/release-model-card/spec.md` (merged delta from archived change `add-code-section-to-release-readme`) **mandates the current state**: a `## Code` section with exactly the five dead bullets (link last), and explicitly requires all pre-existing sections (`## Bundle contents`, `## Metrics`, `## Source paths`, YAML) to remain. The redesign conflicts directly → this change **MUST ship a `MODIFIED` Requirements delta** for domain `release-model-card`, not just `ADDED`.
- The same spec defines the optional `MODEL_CARD_GENERATOR_VERSION` gate — bumping to 3 is the supported way to invalidate stale bundles.
- Note: `openspec/config.yaml` does not exist in this repo (only `specs/` + `changes/`); later phases should not assume it.

### Usage snippet source (Q5)

The repo already ships everything the snippet needs:

- `Awale.Model.load_public_model(path)` (`Model.jl:631`, exported) reconstructs the network from raw Float32 weights + a **sibling** `model_config.toml` — which is staged next to the `.f32` files in the bundle (verified on the live Hub repo tree).
- `predict_inference(model, state)` (`Model.jl:482`, exported) → `(logits, value)`; `initial_state()` and `encode_state` exported from `State`.
- The roundtrip is already proven by `test_publication_flow.jl:191–193`.
- Caveat the card must state: Awale is **not a registered Julia package** — consumers clone the GitHub repo (`julia --project=.`) and download artifacts, e.g. `hf download emiliodavola/awale-alphazero-like --include "releases/mlp/<id>/artifacts/*"`.

## Affected Areas

- `src/Awale/Publication.jl` — `release_model_card` (L587–668), `write_model_card_front_matter` (L177–219), `MODEL_CARD_GENERATOR_VERSION` (L38); possibly a metric-rounding helper and a training/model-config extraction helper.
- `test/test_publication_flow.jl` — update card-text assertions (L90–107, L194–195); keep structural tests untouched.
- `openspec/specs/release-model-card/spec.md` — via a MODIFIED delta in this change.
- `openspec/changes/improve-hf-model-card/` — this change's artifacts.

## Approaches

1. **Single template rewrite** — restructure `release_model_card` + front matter in one change, data scope limited to (summary + bundled config TOMLs + static text), update tests + spec delta, bump generator version.
   - Pros: fixes the framing holistically; one spec delta; one test update; generator-version bump gives a clean restage; the next publish produces the final card in one shot.
   - Cons: single larger diff (~150–250 lines in Publication.jl); all card assertions change at once.
   - Effort: Medium

2. **Incremental section additions** — land rounding first, then sections one PR at a time.
   - Pros: smaller individual PRs.
   - Cons: the Hub card passes through several half-fixed public states (every publish overwrites it); multiple spec deltas against `release-model-card`; more reviewer overhead for the same end state.
   - Effort: Low per step, Higher total

## Recommendation

**Approach 1 — one cohesive rewrite.** The card is generated by exactly two functions and guarded by one test file, so the rewrite is tightly bounded; incrementalism only buys smaller diffs at the cost of publishing intermediate card states. Strict data scope: release summary + bundled `training_config.toml`/`model_config.toml` + static text. Keep internal paths only in `release_summary.toml`/`manifest.toml` (option: keep `## Source paths` for `bundle_kind == "local_trusted"` cards — a decision for the design phase). Parse configs defensively (`get` with fallbacks) — bundle staging validates file existence, not config schema, and older releases must still render.

Rollout note: fixing the generator does **not** fix the live card — the root `README.md` is only rewritten on publish. A card-only re-upload for the current release is possible via the existing `publish_model_card_command` path (or `hf upload` of a restaged bundle README).

## Risks

- Adding file IO (config parsing) to a currently pure render path — mitigate by extracting configs in `write_release_bundle`/caller and passing dictionaries in, or by accepting IO with graceful degradation.
- Exact-string test assertions (title, intro) must change in the same PR — the spec's "existing sections unmodified" scenario is superseded by this change's MODIFIED delta.
- Rounding changes metric formatting in both body and YAML `model-index` — keep one rounding helper used by both to avoid drift.
- HF "Eval results" panel inherits whatever `model-index` carries — adding `description` fields to metrics is supported and improves context.
- Card-only re-upload of an old release must restage with generator version 3 (automatic via `bundle_is_valid`).

## Ready for Proposal

Yes. Scope is bounded (2 functions + 1 test file + 1 spec delta), data availability is verified against the real bundle, and out-of-scope items (charts, hardware, Elo plumbing) are clearly identified. The proposal should state the MODIFIED delta against `release-model-card` and the generator-version bump explicitly.
