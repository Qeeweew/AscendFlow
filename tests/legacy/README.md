# Historical test oracles

This directory retains the retired M2 local synchronization implementation as
an explicit historical oracle. Its modules are imported through `tests.legacy`;
the former `xdsl_npu` module paths have been removed without compatibility
wrappers. Setuptools includes only `src`, so these modules are outside the
compiler product package.

The retained tests preserve M2 contracts, counterexamples, and comparisons.
Passing them does not qualify the current source compiler, its notification
selection, event allocation, progress analysis, final IR, or ASC emitter. Those
paths must satisfy their own current synchronization and validation contracts.

These seven modules still share product IR, native types, and historical
planner/formatting helpers. Such imports use explicit `xdsl_npu` paths; imports
between the retired M2 modules use `tests.legacy` paths. Further retirement of
those shared dependencies requires a separate boundary change.
