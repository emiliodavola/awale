# release-automation

## Requirements

### Requirement: Manual Dispatch Trigger

The release workflow MUST be triggered by `workflow_dispatch` only, MUST run against the `main` branch only, and MUST NOT accept a semver input. The workflow performs no version bump, no commit, and no push.

#### Scenario: Dispatch on main

- GIVEN a `workflow_dispatch` event on `main`
- WHEN a maintainer triggers the workflow
- THEN the workflow starts on `main`
- AND no semver or bump input is required

#### Scenario: No automatic trigger

- GIVEN a push, pull request, or schedule event
- WHEN the event is evaluated
- THEN no release workflow starts

### Requirement: Version Sourced from Project.toml

The workflow MUST read the current version from `Project.toml` (the single source of truth) at dispatch time. The tag and Release MUST be named from that version.

#### Scenario: Version read at dispatch

- GIVEN `Project.toml` on `main` has `version = "0.2.0"`
- WHEN the workflow reads the version
- THEN the workflow uses `0.2.0` for the tag and Release

### Requirement: Version Bump Is a Human-Reviewed PR

The version bump MUST happen in a normal, human-reviewed pull request (dev→main) that edits `Project.toml` `version` and `CITATION.cff` `version` + `date-released`. The release workflow MUST NOT bump versions, MUST NOT create commits, and MUST NOT push.

#### Scenario: Bump lands via PR

- GIVEN a human edits `Project.toml` and `CITATION.cff`
- WHEN the change merges to `main` through the existing PR flow
- THEN the new version is on `main`
- AND the release workflow performs no commit or push

### Requirement: Test Gate Before Release

The workflow MUST run the full test suite before creating a tag or Release. If the suite fails, the workflow MUST abort and MUST NOT create a tag or Release.

#### Scenario: Failing suite aborts the release

- GIVEN the test suite fails
- WHEN the test gate runs
- THEN the workflow aborts
- AND no tag or Release is created

#### Scenario: Passing suite proceeds

- GIVEN the test suite passes
- WHEN the test gate completes
- THEN the workflow continues to tag/Release creation

### Requirement: Annotated Tag Creation

The workflow MUST create an annotated git tag named `v<version>` (where `<version>` is read from `Project.toml`) pointing at the current `main` commit.

#### Scenario: Annotated tag on main

- GIVEN `Project.toml` version is `0.2.0` and tests pass
- WHEN the tag is created
- THEN an annotated tag `v0.2.0` points at the current `main` commit

### Requirement: GitHub Release Creation

The workflow MUST create a GitHub Release for the tag with notes auto-generated from the PRs and commits between the previous tag and this one.

#### Scenario: Release with auto-generated notes

- GIVEN tag `v0.2.0` exists
- WHEN the Release is created
- THEN a GitHub Release for `v0.2.0` exists
- AND its body contains "What's Changed" notes auto-generated since the previous tag

### Requirement: Double-Create Guard

Before creating a tag or Release, the workflow MUST check for an existing tag or Release for the target version. If either already exists, the workflow MUST abort with a clear error and MUST NOT overwrite it.

#### Scenario: Tag already exists

- GIVEN tag `v0.2.0` already exists
- WHEN the workflow attempts to release `0.2.0`
- THEN the workflow aborts with a clear error
- AND the existing tag is left unchanged

#### Scenario: Release already exists

- GIVEN a Release for `v0.2.0` already exists
- WHEN the workflow attempts to create it
- THEN the workflow aborts with a clear error
- AND no Release is overwritten

### Requirement: Permissions and Token

The workflow MUST declare `permissions: contents: write` to create the tag and Release. Because the workflow performs no push, the default `GITHUB_TOKEN` MUST suffice and no PAT fallback is required. The documentation MUST state that no push is performed.

#### Scenario: Default token is sufficient

- GIVEN `permissions: contents: write` is set
- WHEN the workflow creates the tag and Release
- THEN those steps succeed with the default `GITHUB_TOKEN`

#### Scenario: No push step

- GIVEN the workflow runs
- WHEN the workflow completes
- THEN no commit is created and no push to `main` is performed

### Requirement: Explicit Non-Goals

The workflow MUST NOT publish to Hugging Face, MUST NOT perform hotfix-direct-to-main releases (all releases enter via a PR dev→main), MUST NOT maintain a `CHANGELOG.md`, MUST NOT migrate or back-fill existing tags `v0.1.0`..`v0.1.8`, and MUST NOT perform automated version bumping (semver decisions are human-reviewed via PR).

#### Scenario: HF publish stays out of the release path

- GIVEN a release run completes
- WHEN the workflow finishes
- THEN no Hugging Face publication is attempted

#### Scenario: Existing tags untouched

- GIVEN tags `v0.1.0`..`v0.1.8` already exist
- WHEN a new release is created
- THEN none of the historical tags are modified or migrated

#### Scenario: No automated version bumping

- GIVEN a release run completes
- WHEN the workflow finishes
- THEN no version file is modified by the workflow
