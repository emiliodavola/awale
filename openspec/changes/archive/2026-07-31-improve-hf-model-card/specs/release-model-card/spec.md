# Delta for release-model-card

## ADDED Requirements

### Requirement: Stable Card Structure

The card MUST use a stable title and description naming the model, not a build/release ID, and MUST render: Model Details (architecture, parameter count, framework, input encoding, outputs, license, release date), Usage, Training Details, Evaluation, Limitations, Bundle Contents, Source/Reproducibility, and Citation + License.

#### Scenario: Modern release renders the full card

- GIVEN a valid release summary and bundled configs
- WHEN the card is rendered
- THEN the title is the stable model name, not "# Awale release <id> model card"
- AND all listed sections appear, release ID moved into metadata

### Requirement: Per-Checkpoint Promotion Semantics

The card MUST explain promotion per artifact: `model_best` passed the gate when promoted; `model_last`/`model_final` reflect the final run state and MAY NOT have passed. The narrative MUST NOT use a global "Selection promoted" flag.

#### Scenario: Unpromoted final checkpoint described honestly

- GIVEN a release whose gate did not promote the final checkpoint
- WHEN the card is rendered
- THEN `model_best` is stated to have passed the gate at selection
- AND `model_last`/`model_final` are stated as final state without gate claims
- AND no global "Selection promoted: <bool>" line appears

### Requirement: Complete Evaluation Methodology

Evaluation MUST state the baseline identity (RandomAgent), MCTS simulations (400), evaluation game count (100), and the gate: promotion needs a decided win rate of at least 56% over 200 promotion games against the current best.

#### Scenario: Methodology fully exposed

- GIVEN a rendered card
- WHEN the Evaluation section is read
- THEN it names RandomAgent as baseline
- AND states 400 MCTS simulations and 100 evaluation games
- AND states the 56%-over-200-games promotion threshold

### Requirement: Shared Metric Rounding

One shared rounding helper MUST format all metrics to 2-4 significant decimals, used by BOTH the card body and the YAML `model-index` so values never drift.

#### Scenario: Body and YAML values agree

- GIVEN a raw value such as 61.702127659574465
- WHEN card and front matter are rendered
- THEN the value appears identically rounded (e.g. 61.7) in body and `model-index`

### Requirement: Model-Index YAML Metadata

YAML front matter MUST include tags `alphazero`, `self-play`, `board-game`, MUST emit `model-index` values via the shared rounding helper, and MUST describe each custom metric.

#### Scenario: Enriched front matter

- GIVEN a rendered card
- THEN YAML tags include `alphazero`, `self-play`, `board-game`
- AND `model-index` values are rounded
- AND each custom metric has a description

### Requirement: No Internal Paths in Public Card

The card MUST NOT contain a `## Source paths` section or any local filesystem path (e.g. `checkpoints\mlp\log\...`). Internal paths MAY remain in `release_summary.toml`.

#### Scenario: No local paths leak

- GIVEN a rendered card
- THEN it contains neither `## Source paths` nor any local path string

### Requirement: Deduplicated Bundle Contents

Bundle Contents MUST list each bundled file exactly once, each with a one-line description of its role (e.g. how `model_best`, `model_last`, `model_final` differ).

#### Scenario: release_summary.toml listed once

- GIVEN a bundle whose artifact list already contains `release_summary.toml`
- WHEN the card is rendered
- THEN every filename appears exactly once
- AND each entry carries a description

### Requirement: Defensive Rendering of Older Releases

The card MUST render without error for releases lacking the new config files (`training_config.toml`/`model_config.toml`) or optional metric keys; missing data MUST degrade to fallback values, never crash.

#### Scenario: Old release still renders

- GIVEN an older staged bundle without the config TOMLs
- WHEN the card is rendered
- THEN README content is produced without error
- AND missing-config sections show fallback values

### Requirement: Test Coupling

`test/test_publication_flow.jl` MUST be updated in the same change so every card-text assertion matches the new template; no assertion MAY keep referencing the old template.

#### Scenario: Card assertions pass

- GIVEN the generator change is applied
- WHEN `test_publication_flow.jl` runs
- THEN all card-text assertions pass
- AND none reference the build-ID title or dead-bullet Code list

## MODIFIED Requirements

### Requirement: Code Section in Model Card

The `release_model_card()` function MUST emit a `## Code` section after `## Bundle contents` and before the closing return. Every list item MUST be a hyperlink into the GitHub repository; plain-text bullets are forbidden. Items MUST cover training scripts, inference code, evaluation scripts, configuration files, and the repository, with the repository link last.

(Previously: exactly five items — four plain-text bullets plus one bare repo URL. The "Existing sections are unmodified" scenario is superseded: sections are restructured and `## Source paths` is dropped.)

#### Scenario: Code section uses real links

- GIVEN a valid release summary with all required keys
- WHEN `release_model_card()` is called
- THEN the output contains `"## Code"`
- AND every item links to `github.com/emiliodavola/awale`
- AND the repository link appears last

#### Scenario: No dead bullets remain

- GIVEN a rendered card
- WHEN the `## Code` section is inspected
- THEN no item is plain text without a hyperlink

### Requirement: Version Gate for Bundle Validity

`MODEL_CARD_GENERATOR_VERSION` MUST be `3`. `bundle_is_valid()` MUST reject any bundle whose `model_card_generator_version` differs from `3`, forcing a restage of cached bundles.

(Previously: optional `1`/`2`; `bundle_is_valid()` accepted the current version.)

#### Scenario: Cached bundles are restaged

- GIVEN a staged bundle with `model_card_generator_version = 2`
- WHEN the version is `3`
- THEN `bundle_is_valid()` returns `false`
- AND `stage_release_bundle()` rebuilds it

#### Scenario: New bundles record version 3

- GIVEN the version is `3`
- WHEN a bundle is staged
- THEN the manifest records `model_card_generator_version = 3`
- AND `bundle_is_valid()` returns `true`

## RENAMED Requirements

### Requirement: Version Gate for Bundle Validity (Optional) → Version Gate for Bundle Validity

(Reason: the version is fixed and mandatory; "(Optional)" is misleading.)
(Migration: "None" — the MODIFIED block above replaces the content.)

## REMOVED Requirements

None.
