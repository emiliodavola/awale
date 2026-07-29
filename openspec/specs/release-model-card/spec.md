# Delta for release-model-card

## ADDED Requirements

### Requirement: Code Section in Model Card

The `release_model_card()` function MUST emit a `## Code` section in the generated README output. The section MUST appear after the `## Bundle contents` section and before the closing return of the function.

The `## Code` section MUST contain a bullet list with exactly five items:
- `Training scripts`
- `Inference code`
- `Evaluation scripts`
- `Configuration files`
- A hyperlink to `<https://github.com/emiliodavola/awale>`

The link item MUST appear last in the bullet list.

#### Scenario: Code section appears in generated model card

- GIVEN a valid release summary with all required keys
- WHEN `release_model_card()` is called
- THEN the returned string MUST contain `"## Code"`
- AND the string MUST contain `"github.com/emiliodavola/awale"`
- AND all five list items appear in the specified order

#### Scenario: Existing sections are unmodified

- GIVEN the same release summary
- WHEN `release_model_card()` is called
- THEN the returned string MUST still contain all pre-existing sections (`"## Bundle contents"`, `"## Metrics"`, `"## Source paths"`, YAML front matter, etc.)
- AND all existing `@test occursin(...)` assertions in `test_publication_flow.jl` MUST pass unchanged

### Requirement: Version Gate for Bundle Validity (Optional)

Implementations MAY bump `MODEL_CARD_GENERATOR_VERSION` from `1` to `2`. If bumped, `bundle_is_valid()` MUST reject bundles with `model_card_generator_version = 1`, forcing a rebuild of previously staged bundles to include the new section. If NOT bumped, `bundle_is_valid()` MUST accept bundles at version 1, and pre-existing staged bundles remain valid without rebuilding.

#### Scenario: Version bumped → old bundles are invalid

- GIVEN a staged bundle with `model_card_generator_version = 1`
- WHEN `MODEL_CARD_GENERATOR_VERSION` is `2`
- THEN `bundle_is_valid()` MUST return `false` for that bundle
- AND `stage_release_bundle()` MUST rebuild the bundle

#### Scenario: Version not bumped → old bundles stay valid

- GIVEN a staged bundle with `model_card_generator_version = 1`
- WHEN `MODEL_CARD_GENERATOR_VERSION` is `1`
- THEN `bundle_is_valid()` MUST return `true` for that bundle
- AND `stage_release_bundle()` MUST NOT rebuild the bundle

## MODIFIED Requirements

No existing requirements are modified. The spec change is purely additive.

## REMOVED Requirements

None.

## RENAMED Requirements

None.
