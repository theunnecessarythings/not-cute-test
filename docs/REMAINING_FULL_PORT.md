# Remaining full-port work

This project is not yet a full production-equivalent replacement for the uploaded CuteDSL tree.

## Highest-priority gaps

1. Make all generated tensor/copy/MMA/tiled examples pass not only CUTLASS parsing, but the real `cute-to-nvvm` lowering pipeline.
2. Implement CUDA runtime execution: compile, load module, resolve kernel symbol, pack arguments, launch, synchronize, and report driver errors.
3. Complete source-wide API parity for `cutlass/cute/core.py`, `tensor.py`, `algorithm.py`, `arch/*`, `nvgpu/*`, runtime/export/JIT support, and utility modules.
4. Replace descriptor-only architecture catalogs with exact SM80/SM90/SM100/tcgen05 validation and lowering behavior.
5. Add real numerical examples and regression tests: copy, tiled copy, MMA microkernel, GEMM skeleton, and architecture-specific kernels.
6. Split full-project optimized builds into smaller targets so ReleaseSafe validation is practical in constrained environments.
7. Add upstream-scale golden tests and negative tests for invalid layouts, memory spaces, atom configs, dtype combinations, and malformed MLIR.

## Current non-goals

- Recreating Python AST/decorator behavior. The Zig port should remain Zig-native.
- Linking MLIR/Python into the core Zig library by default.
- Pretending descriptor-level emitters are equivalent to CUDA execution.
