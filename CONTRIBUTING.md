# Contributing to Awale

Thanks for helping improve Awale.

## Quick path

1. Work from a branch off `dev`.
2. Link your change to an approved issue.
3. Make the smallest useful Julia change.
4. Run the test suite.
5. Open a PR with a clear summary.

## Repository conventions

- This is a **Julia** repository. Use `julia --project=.` for local commands.
- Keep code and docs in English.
- Keep `README_ES.md` as the Spanish mirror of `README.md`.
- Prefer specification-first changes: contracts/specs → tests → implementation.
- Use conventional commits, for example `feat(scope): description`.

## Local setup

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Testing

Run the full suite before opening a PR:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

If you only changed docs, still make sure the repo stays consistent and the affected examples or snippets remain valid.

## Code and docs

- Keep terminal output, comments, and user-facing strings in English.
- Prefer small, reviewable changes.
- Update tests when behavior or output changes.
- For publication changes, keep Hugging Face model-card text in English.

## Pull requests

- Open PRs against `dev` unless the maintainer says otherwise.
- Link the approved issue in the PR body.
- Add a clear summary of what changed and how you verified it.

## License

By contributing, you agree that your changes will be distributed under the repository’s MIT License.
