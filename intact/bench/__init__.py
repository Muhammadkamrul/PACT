"""Paired multi-method benchmark support for INTACT (v10).

Modules
-------
lean_rundir  drop-in replacement for intact.runlog.RunDir that keeps only a
             compact per-epoch decision trace (no checkpoints, no per-decision
             JSON), so thousands of runs fit on disk.
registry     every method, baseline, ladder rung, ablation and sensitivity
             variant, with the exact configuration overrides it uses, plus
             validation that rejects typos and forbidden MILD overrides.
stats        paired scenario-level statistics (bootstrap CI, Wilcoxon, sign
             test, Holm correction, TOST equivalence, power).
fingerprint  hash of the code that can change a simulation result, used to
             invalidate cached results safely.
"""
