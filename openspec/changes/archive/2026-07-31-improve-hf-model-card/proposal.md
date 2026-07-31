# Proposal: Improve Hugging Face Model Card Quality

## Intent

The published card at `emiliodavola/awale-alphazero-like` reads as a raw release-summary dump (issue #63): build-ID title, leaked internal paths, raw-precision context-free metrics, a confusing global `Selection promoted: false` flag, dead Code bullets, duplicated bundle entries, and no usage or methodology. The card is generated and every publish overwrites the Hub README, so the generator template is the fix point. Audience is mixed: RL researchers and model users — usage/quickstart high up, complete methodology below.

## Scope

### In Scope
- Full template rewrite: stable title/description; Model Details (architecture, parameter count, framework, encoding, outputs, license, release date); Usage snippet (`load_public_model` + `predict_inference`); Training Details; Evaluation with complete methodology (baseline `RandomAgent`, 100 games, 400 MCTS sims, 56% decided-wins gate over 200 promotion games); Limitations; deduplicated Bundle Contents with per-file descriptions; concrete Source/Reproducibility links; Citation + License.
- Per-checkpoint promotion semantics: `model_best` was promoted when selected (passed the gate); `model_last`/`model_final` reflect the run's final state and may not have passed. Remove the single global "Selection promoted" flag from the card narrative.
- Metric rounding (2-4 significant decimals) via one helper shared by body and YAML `model-index`.
- YAML: extra tags (`alphazero`, `self-play`, `board-game`), rounded values, metric descriptions.
- Drop `## Source paths` from the public card (paths stay in `release_summary.toml`); replace the dead-bullet `## Code` section with real repo links.
- `MODEL_CARD_GENERATOR_VERSION` 2→3; update `test/test_publication_flow.jl` card assertions in the same PR.
- Fully generated: all narrative lives in the template; zero hand-curated Hub text.

### Out of Scope
- Charts/images, hardware info, Elo in the card, new telemetry plumbing.
- Changes to training, MCTS, or bundle artifact contents.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `release-model-card`: the current spec mandates the dead-bullet Code section and retention of Source paths; the delta replaces these with the restructured card (MODIFIED + ADDED requirements in this domain).

## Approach

Single cohesive rewrite (exploration recommendation): restructure `release_model_card` + `write_model_card_front_matter` in one change. Data scope: release summary + bundled `training_config.toml`/`model_config.toml` (parsed defensively with fallbacks so older releases still render) + static-verifiable facts (optimizer, baseline identity, encoding). One rounding helper serves body and YAML. Generator version 3 forces restage of cached bundles. The live Hub card updates on next publish, or via a card-only re-upload through `publish_model_card_command`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/Awale/Publication.jl` | Modified | Card body + front matter rewrite; generator version 3; rounding/config helpers |
| `test/test_publication_flow.jl` | Modified | Card-text assertions (L90-107, L194-195) updated to new template |
| `openspec/specs/release-model-card/` | Modified | Delta spec: MODIFIED + ADDED requirements |
| HF repo README | Modified | Regenerated card uploaded on next publish |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Config parsing adds IO to a pure render path | Med | Extract configs in caller; defensive `get` with graceful degradation |
| Body/YAML metric drift | Low | Single shared rounding helper |
| Older releases fail to render | Low | Fallback parsing + version gate restages bundles |
| Exact-string tests break | High (expected) | Update assertions in same PR under the MODIFIED delta |

## Rollback Plan

`git revert` the PR; bundles restage to the previous template on next run. The live Hub card only changes on publish, so rolling back an already-published card means re-uploading the prior README via `publish_model_card_command` (or `git revert` on the Hub repo).

## Dependencies

- Issue #63 (approved); exploration artifact `sdd/improve-hf-model-card/explore`.

## Success Criteria

- [ ] Regenerated card contains all #63 sections with rounded metrics and per-checkpoint promotion semantics; no Source paths, no dead Code bullets, no duplicated bundle entries.
- [ ] `julia --project=. test/test_publication_flow.jl` passes with updated assertions.
- [ ] Older release summaries render without error (defensive parsing).
- [ ] Merged spec delta leaves `release-model-card` reflecting the new template.
