# Parity status

This Zig project is an integrated work-in-progress, not a full production-equivalent CuteDSL implementation.

## Meaningful parity already present

- Static tuple/layout algebra foundation.
- Higher-level layout product/divide/composition helpers.
- Textual MLIR construction and deterministic golden generation.
- CUTLASS DSL parser bridge using installed `nvidia-cutlass-dsl`.
- Parser-accepted Cute/CuteNVGPU fixtures for layout, tensor memrefs, copy atoms, MMA atoms, tiled copy, and tiled MMA skeletons.
- Tensor/SSA descriptor layer with vector/scalar load-store emission hooks.
- Atom/trait descriptors with runtime-field validation.
- Runtime/export planning and C-wrapper text generation.
- Examples and build targets.

## Non-parity remaining

- Full API coverage across the uploaded CuteDSL source tree.
- Real pass-pipeline validation through `cute-to-nvvm` for generated kernels.
- Cubin/object generation and CUDA launch.
- Exhaustive SM80/SM90/SM100/tcgen05 op validation and exact lowering.
- Full tensor arithmetic/indexing/reduction behavior.
- Upstream-scale regression tests and numerical kernel validation.


## Core/API and architecture constructor status

The integrated library includes `core.zig`, `arch_atoms.zig`, and
`arch_catalog.zig` as executable modules. Architecture support is represented
by typed descriptors, constructors, and validation tests rather than generated
source inventories.

Remaining parity work is still significant: many runtime, JIT, full tensor, utility, pipeline, scheduler, and architecture-specific edge cases are not yet source-complete.
## Runtime execution wiring update

The runtime layer now contains a concrete CUDA Driver API bridge (`cuda_driver.zig`) and launch orchestration layer (`execution.zig`). This closes the code-generation-side wiring gap for module loading, symbol resolution, argument packing, memory copy helpers, stream synchronization, and `cuLaunchKernel` invocation. Actual GPU launch remains unverified in this sandbox.

## Latest integrated pipeline/API/architecture pass

The library includes `compile_pipeline.zig`, `semantics.zig`, and
`arch.zig`. These add CUTLASS bridge artifact planning, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.
