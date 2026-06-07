# Source inventory summary

The uploaded CuteDSL source tree was scanned as the reference corpus.

Approximate reference size:

- 140 total files
- 134 Python modules
- about 82k total Python lines
- about 65k non-comment Python LOC

The current Zig project is an integrated partial port. It implements a cohesive foundation for layout algebra, MLIR text construction, tensor/SSA descriptors, atoms/copy/MMA descriptors, runtime/export planning, CUTLASS parser bridge, and examples.

This file intentionally avoids treating implementation checkpoints as API boundaries. Remaining parity is tracked by source area in `docs/REMAINING_FULL_PORT.md`.
