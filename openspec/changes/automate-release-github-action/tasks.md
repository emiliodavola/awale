# Tasks: Automate Release with a GitHub Action (Modelo B)

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~115-160 (release.yml ~90-120 new, CONTRIBUTING.md ~25-40 added) |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested PR boundary | Single PR (2 files, ~150 lines — no slice needed) |
| Delivery strategy | single-pr |
| Chain strategy | size-exception |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | release.yml + CONTRIBUTING.md "Releases" | PR 1 | Single PR; commits split by work unit (workflow scaffold → gate/version/guards → docs → validation) |

## Phase 1: Release Workflow (`.github/workflows/release.yml`)

- [ ] 1.1 Create workflow scaffold — `name: Release`; `on: workflow_dispatch` restricted to `branches: [main]` with NO inputs; `permissions: contents: write`; job `release` on `ubuntu-latest`; `actions/checkout@v4` with `fetch-depth: 0`.
  - Files: `.github/workflows/release.yml` (new)
  - Verify: YAML parses; dispatch UI offers only `main`; no semver input shown.
- [ ] 1.2 Inline test gate — copy the exact steps from `.github/workflows/ci.yml`: `julia-actions/setup-julia@v2` (version `'1.11'`), `julia-actions/cache@v2`, config copies (`cp config.toml.example config.toml`, `cp src/Awale/config.toml.example src/Awale/config.toml`), `julia --project=. -e "using Pkg; Pkg.instantiate()"`, then `julia --project=. -e "using Pkg; Pkg.test()"` as the gate (fails ⇒ job aborts).
  - Files: `.github/workflows/release.yml`
  - Verify: steps match ci.yml verbatim (spec requires `Pkg.instantiate` even though the design's illustrative YAML omits it); any test failure fails this step.
- [ ] 1.3 Add version extraction — step `Read version from Project.toml` with `id: version`: `julia -e 'using TOML; print(TOML.parsefile("Project.toml")["version"])'` → `$GITHUB_OUTPUT` as `version`.
  - Files: `.github/workflows/release.yml`
  - Verify: step output equals Project.toml `version` (currently `0.1.8`).
- [ ] 1.4 Add guards + create steps — tag guard via `git ls-remote --tags origin refs/tags/v${{ steps.version.outputs.version }}` (exists ⇒ `::error::Tag ... already exists — aborting`, exit 1); release guard via `gh release view v<version>` (exists ⇒ `::error::Release ... already exists — aborting`, exit 1); then `git tag -a` + `git push origin v<version>` (tag ref ONLY); final step `gh release create v<version> --generate-notes` with `GH_TOKEN: ${{ github.token }}`.
  - Files: `.github/workflows/release.yml`
  - Verify: no branch push exists anywhere; guards run before any create; `gh release create` is the last step.

## Phase 2: Documentation

- [ ] 2.1 Add "Releases" section to `CONTRIBUTING.md` — human PR-bump convention: edit `Project.toml` `version` + `CITATION.cff` `version`/`date-released` in a normal dev→main PR, merge via existing review flow, then dispatch `Release` manually. Document the double-create guard (abort if tag/Release exists) and that no PAT is needed (workflow only pushes the tag ref; default `GITHUB_TOKEN` suffices; no branch push).
  - Files: `CONTRIBUTING.md`
  - Verify: section covers bump files, PR flow, dispatch steps, guard behavior, no-PAT statement; consistent with spec scenarios.

## Phase 3: Verification

- [ ] 3.1 Validate `release.yml` with actionlint — not installed locally; use Docker (available): `docker run --rm -v "${PWD}:/repo" -w /repo rhysd/actionlint:latest -color`; fallback: manual review against GitHub Actions schema.
  - Files: `.github/workflows/release.yml`
  - Verify: actionlint exits 0 with no errors/warnings.
- [ ] 3.2 E2E smoke — after merge to `main`, manually dispatch `Release`. Expect the double-create guard to abort cleanly (current version `0.1.8` already tagged/released) unless a bump PR landed first; on the success path verify annotated tag `v<version>` and Release with "What's Changed" notes.
  - Files: none (runtime on GitHub)
  - Verify: duplicate dispatch aborts with clear `::error::`; success path creates tag + Release once.