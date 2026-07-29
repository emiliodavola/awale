# Proposal: Add Code Section to Release README

## Intent

The model card published to Hugging Face (`release_model_card()`) documents the model and bundle contents but provides no link back to the source code. Users browsing the model on Hugging Face have no way to find the training scripts, inference code, or repository. A brief "Code" section with links closes this gap.

## Scope

### In Scope
- Add 7 `println` lines in `src/Awale/Publication.jl` → `release_model_card()` to insert a "## Code" section before the closing return
- Add 2 test assertions in `test/test_publication_flow.jl` verifying the new section content
- Bump `MODEL_CARD_GENERATOR_VERSION` (optional, same PR)

### Out of Scope
- Changing the publication pipeline, bundle structure, or any other feature
- Modifying any training, MCTS, self-play, or model code
- Adding new files or restructuring the repo

## Capabilities

### New Capabilities
None — this is a cosmetic change to an existing output template.

### Modified Capabilities
- `release-model-card`: The generated README gains a "Code" section with a repository link. No behavior change, only output text content.

## Approach

Insert 7 lines into `release_model_card()` between the end of the "## Bundle contents" loop (line 568) and the `return String(take!(io))` (line 570):

```julia
    println(io)
    println(io, "## Code")
    println(io, "- Training scripts")
    println(io, "- Inference code")
    println(io, "- Evaluation scripts")
    println(io, "- Configuration files")
    println(io, "- <https://github.com/emiliodavola/awale>")
```

Add 2 assertions in `test_publication_flow.jl`:
- `@test occursin("## Code", card)` — section header present
- `@test occursin("github.com/emiliodavola/awale", card)` — repo URL present

Optionally bump `MODEL_CARD_GENERATOR_VERSION` in `Publication.jl`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/Awale/Publication.jl` | Modified | +7 `println` in `release_model_card()` |
| `test/test_publication_flow.jl` | Modified | +2 `@test` assertions in the model card block |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Snapshot test fails due to assertion mismatch | Low | Only adding new assertions; existing ones unchanged |
| Bump forgotten or version mismatch | Low | Optional step, tests pass without it |

## Rollback Plan

Revert the 2 files: remove the 7 `println` lines from `Publication.jl` and the 2 `@test` lines from `test_publication_flow.jl`. Single-commit revert, zero impact on functionality.

## Dependencies

None.

## Success Criteria

- [ ] Running `julia --project=. test/test_publication_flow.jl` passes all assertions (existing + new)
- [ ] The generated model card at `model-card/README.md` contains a "## Code" section linking to the repository
- [ ] Zero changes outside the two specified files
