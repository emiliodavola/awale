# Tasks: Add Code Section to Release README

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~10–12 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | single-pr |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: Low

## Phase 1: Core Implementation (Publication.jl)

- [x] 1.1 Inserted 7 `println` lines in `src/Awale/Publication.jl` (lines 570–576): blank line + `## Code` header + 5 bullet items (`- Training scripts`, `- Inference code`, `- Evaluation scripts`, `- Configuration files`, `- <https://github.com/emiliodavola/awale>`)
- [x] 1.2 _(Optional — skipped)_ `MODEL_CARD_GENERATOR_VERSION` not bumped; pre-existing staged bundles remain valid

## Phase 2: Test Assertions (test_publication_flow.jl)

- [x] 2.1 Added `@test occursin("## Code", model_card)` after existing assertions (line 106)
- [x] 2.2 Added `@test occursin("github.com/emiliodavola/awale", model_card)` after new assertion from 2.1 (line 107)

## Phase 3: Verification

- [x] 3.1 Ran `julia --project=. test/test_publication_flow.jl` — all 75/75 tests pass (existing + new)
- [x] 3.2 _(Skipped — version not bumped, no regression risk)_

## Implementation Order

Phase 1 → Phase 2 → Phase 3. All tasks are linear: you can't test before the code section exists. The version-bump (1.2) is optional and independent of the test additions.
