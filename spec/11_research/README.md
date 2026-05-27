11_research: Decision log, experiments, and open questions

Purpose
- Capture research notes, design trade-offs, experiment results, and decisions so future contributors understand rationale.

Contents (files to maintain)
- decision_log.md  -- every architectural or interface decision with timestamp and author
- experiments.md   -- experiment configs and results (reproducible via checkpoint metadata)
- known_issues.md  -- current limitations and TODOs
- literature.md    -- references to Awale/Oware rule variants, AlphaZero, and RL search papers

Reproducibility
- Each experiment must record: code commit hash, Project.toml, Julia version, hardware, RNG seeds, and dataset/checkpoint IDs

Initial open questions
- Best canonicalization: rotation-only vs mirroring? Trade-offs documented
- Preferred property-based testing library for Julia (implement small in-repo generator as fallback)

Test artifacts
- Attach canonical test positions and their expected outcomes here for reference in regression tests