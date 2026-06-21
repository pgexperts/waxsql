"""Internal package for the data generator.

Public entry point is `waxsql.data.generate_data`. Submodules:
- strategies: per-type value generators
- columns:    column-name override registry (hook for future plausibility)
- rows:       topological row materialization + FK resolution
- emit:       COPY block formatting

Role: every module in this package is internal. Callers should import
through `waxsql.data` (or the top-level `waxsql.generate_data` re-export);
the layout here may change without notice. The split exists so each
concern (type-keyed values, name-keyed overrides, FK ordering, COPY
encoding) lives in one file with one set of invariants.
"""
