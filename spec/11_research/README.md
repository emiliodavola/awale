# 11_research: Lightweight research notes and decision trail

This repository currently keeps research history in a lightweight form.

## Current sources of truth

- `README.md` for the current experimental workflow
- `config.toml` for runtime experiment settings
- commit history for concrete implementation decisions
- arena outputs and checkpoint comparisons for training signal interpretation

## Recommended notes to maintain when needed

These files are recommended, but not mandatory unless the workflow starts depending on them:
- `decision_log.md`
- `experiments.md`
- `known_issues.md`
- `literature.md`

## Minimum expectation

When a change affects training methodology, evaluation methodology, or model/data contracts, update:
1. the relevant spec file,
2. the README usage notes if the workflow changes,
3. tests that prove the behavior.

## Open research questions currently relevant

- How stable is the checkpoint arena signal as training scales beyond the first milestones?
- When should the repo move from baseline-vs-random selection to stronger selection criteria?
- At what point does the current MLP become the bottleneck rather than data/search budget?
