# Porting notes

The project is now organized as a Zig library with stable domain modules. Earlier checkpoint files have been consolidated; transient source names were removed from `src/` and replaced with cohesive module names.

## Design constraints

- Zig-native public API.
- Zero default dependency on MLIR/CUDA/Python.
- Textual MLIR generation by default.
- Optional CUTLASS parser verification through `tools/cutlass_mlir_bridge.py` when `nvidia-cutlass-dsl` is installed.
- No Python AST/decorator compatibility layer is required for the Zig port.

## Current translation scope

Implemented foundations include layout algebra, representation-agnostic MLIR construction, type/tensor/SSA descriptors, atom/copy/MMA descriptors, parser-aligned Cute MLIR fixtures, a CUTLASS DSL parser bridge, runtime/export planning metadata, and examples. The current backend encodes IR textually, but callers use IR values, types, operations, attributes, and module builders rather than a text-oriented API.

This is still not full source parity with the uploaded CuteDSL tree. See `docs/REMAINING_FULL_PORT.md` for the remaining production work.


## Runtime execution wiring

The integrated library now includes `cuda_driver.zig` and `execution.zig`. These provide real CUDA Driver API dynamic loading and launch wiring, while leaving actual GPU execution to environments that have CUDA, a generated cubin/fatbin, and the expected symbols available.

## Latest integrated pipeline/API/architecture pass

The library includes `compile_pipeline.zig`, `semantics.zig`, and
`arch.zig`. These add CUTLASS bridge artifact planning, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.
