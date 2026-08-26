# Apply Progress: automate-release-github-action

## Status

- All 7 tasks complete except 3.2 (E2E smoke — requires GitHub runtime after merge to `main`; deferred to verify phase).
- Mode: Standard workflow (this change introduces no Julia-logic units — only a workflow YAML and docs; a Strict-TDD evidence table is included below for the verify gate).

## Tasks Completed

- [x] 1.1 Workflow scaffold — `workflow_dispatch` (no inputs), main-only enforcement, `permissions: contents: write`, `ubuntu-latest`, `actions/checkout@v4` with `fetch-depth: 0`.
- [x] 1.2 Inline test gate — exact `ci.yml` steps: `setup-julia@v2` (`1.11`), `cache@v2`, config copies, `Pkg.instantiate()`, `Pkg.test()` (fails ⇒ job aborts).
- [x] 1.3 Version extraction — `julia -e 'using TOML; ...'` → `$GITHUB_OUTPUT` as `version` (verified locally: `0.1.8`).
- [x] 1.4 Guards + create — tag guard (`git ls-remote`), release guard (`gh release view`), annotated tag + tag-only push, `gh release create --generate-notes` with `GH_TOKEN: ${{ github.token }}`.
- [x] 2.1 CONTRIBUTING.md "Releases" section — PR-bump convention, dispatch steps, double-create guard, no-PAT statement.
- [x] 3.1 actionlint validation via Docker (`rhysd/actionlint:latest`) — exit 0, no errors/warnings.
- [ ] 3.2 E2E smoke — manual dispatch after merge to `main` (runtime on GitHub; verify phase).

## Deviations from Spec/Design/Tasks

1. **`on.workflow_dispatch.branches: [main]` (tasks.md task 1.1 / orchestrator binding constraint 1) replaced by an in-workflow ref guard.** `branches` is NOT valid under `workflow_dispatch`:
   - GitHub staff (actions/runner#1858): "The fields `branches`, `name`, `description`, and `secrets` have never been supported under `workflow_dispatch`... The only valid key under `workflow_dispatch` is `inputs`." Workflows using it fail with "Workflow is not valid".
   - Official docs document `branches` filters only for `push`/`pull_request`/`pull_request_target`.
   - actionlint (latest, mirrors GitHub's real schema) rejects it: `expected "inputs" key for "workflow_dispatch" section but got "branches"`.
   - Preserved the spec INTENT (manual trigger, main-only, no semver input): `workflow_dispatch` with no inputs + first step `Guard dispatch branch is main` (`github.ref == 'refs/heads/main'` → clear `::error::` + exit 1). Design.md is consistent (its illustrative YAML omits `branches`).

## Verification Results

- actionlint (`rhysd/actionlint:latest`, Docker): **exit 0**, no errors/warnings.
- YAML parse: validated by actionlint's YAML parser (`ConvertFrom-Yaml` not available in pwsh).
- Version-extraction command tested locally end-to-end under `bash` (runner shell): writes `version=0.1.8` to `$GITHUB_OUTPUT`.
- Guard logic sanity: `git ls-remote --tags origin refs/tags/v0.1.8` returns the tag and `gh release view v0.1.8` exits 0 ⇒ double-create guard would abort for the current version (expected).
- `git diff origin/dev -- .github/workflows/ci.yml` is **empty** (ci.yml untouched).
- Working tree clean after all commits.

## Commits (work units)

| Commit | Message |
|--------|---------|
| `ba5e404` | docs(sdd): add SDD artifacts for automate-release-github-action change |
| `a6ab75e` | ci(release): scaffold manual release workflow restricted to main |
| `8fdc984` | ci(release): inline CI test gate into release workflow |
| `7364596` | ci(release): guard against existing tag/release and create annotated tag + GitHub release |
| `512b5c9` | docs(contributing): document manual release process with PR-based version bumps |
| apply commit | docs(sdd): add apply-progress report and mark tasks complete |

## Files Changed

| File | Action | What Was Done |
|------|--------|---------------|
| `.github/workflows/release.yml` | Created | Dispatch-only release pipeline (67 lines): main guard, inlined CI test gate, version extraction, double-create guards, annotated tag + tag-only push, GitHub Release. |
| `CONTRIBUTING.md` | Modified | Added "Releases" section (+44 lines): PR-bump convention, dispatch steps, double-create guard, no-PAT note. |
| `openspec/changes/automate-release-github-action/*` | Added | SDD artifacts (proposal, exploration, design, tasks, spec) + this apply-progress. |

## TDD Cycle Evidence (verify-phase gate)

| Task | RED (validation defined first) | GREEN (implementation) | REFACTOR |
|------|--------------------------------|------------------------|----------|
| 1.1 | actionlint/parse accepts trigger block; no inputs shown | `workflow_dispatch` (no inputs) + main guard step | N/A — no code unit |
| 1.2 | Steps match ci.yml verbatim incl. `Pkg.instantiate` (task verify) | 6 steps inlined | N/A |
| 1.3 | Output equals Project.toml version (tested: `0.1.8`) | TOML read step | N/A |
| 1.4 | Guards precede create; no branch push; `gh release create` last | 2 guards + tag push + release | N/A |
| 2.1 | Spec scenarios: bump files, PR flow, dispatch, guard, no-PAT | CONTRIBUTING.md Releases section | N/A |
| 3.1 | actionlint exits 0 | Docker actionlint run passed | N/A |
| 3.2 | E2E smoke on GitHub (post-merge) | Deferred — requires runtime | N/A |

Note: no Julia test units exist for this change (workflow YAML + docs only). The executable validation for YAML is actionlint (run in this phase; sdd-verify re-runs it), and 3.2 is a runtime smoke only executable post-merge.

## Risks

- `.github/.gitignore` (`*`) blocks new workflow files; `release.yml` was added with `git add -f` (tracked files bypass gitignore). Future workflow files need `-f` or a `.gitignore` negation (out of scope here).
- The `branches: [main]` constraint from the orchestrator is invalid GitHub syntax; replaced with the ref guard (documented above). Verify phase should re-confirm intent.
- E2E smoke (3.2) requires merging to `main`; the double-create guard will abort for `0.1.8` (already tagged/released) unless a bump PR lands first — expected behavior.

## Next

- `sdd-verify`: re-run actionlint, spec-scenario checklist, post-merge smoke plan.