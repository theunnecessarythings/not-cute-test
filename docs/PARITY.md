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


## Source-derived parity manifests

The current integrated artifact compiles the uploaded CuteDSL API surface into `src/api_manifest.zig` and `docs/api_surface_manifest.json`:

- public top-level records: 1327
- classes: 456
- functions: 871
- public Zig name matches: 155

It also compiles the `cute.arch` / `cute.nvgpu` surface into `src/arch_manifest.zig` and `docs/arch_nvgpu_manifest.json`:

- arch/nvgpu public records: 380
- copy records: 96
- MMA records: 51
- records with extracted validation rules: 23
- records with extracted MLIR type factories: 57

These manifests are now part of the Zig test suite. They are used to prevent accidental hiding of missing API/architecture coverage.


## Core/API and architecture constructor status

The integrated library includes `core.zig` and `arch_atoms.zig` as executable modules, separate from generated audit manifests.

Remaining parity work is still significant: many runtime, JIT, full tensor, utility, pipeline, scheduler, and architecture-specific edge cases are not yet source-complete.
## API closure continuation

The public name audit now reports 841 / 1,327 source-name matches overall and 721 / 721 matches for `cutlass.cute.*`. This is still not full semantic parity: runtime CUDA execution, exact dynamic behavior, non-Cute integrations, and systematic `cute-to-nvvm` lowering remain open.


## Runtime execution wiring update

The runtime layer now contains a concrete CUDA Driver API bridge (`cuda_driver.zig`) and launch orchestration layer (`execution.zig`). This closes the code-generation-side wiring gap for module loading, symbol resolution, argument packing, memory copy helpers, stream synchronization, and `cuLaunchKernel` invocation. Actual GPU launch remains unverified in this sandbox.

## Latest integrated pipeline/API/architecture pass

The library now includes `compile_pipeline.zig`, `pipeline_verify.zig`,
`semantics.zig`, and `arch_validation.zig`. These add CUTLASS bridge artifact
planning/extraction commands, sharded parser/pipeline verification, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors. See
`docs/PIPELINE_API_ARCH_IMPLEMENTATION.md`.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.


## Upstream CuTeDSL example parity

Implemented records: 9/9 packaged CuTeDSL notebook/FFI examples available in the installed CUTLASS examples tree.

The upstream notebook mappings are parity fixtures, not executable Zig
examples. Their inventory and limitations are documented in
`docs/UPSTREAM_EXAMPLE_PARITY.md`; generated MLIR lives under
`testdata/golden/upstream/`.

Each record has a golden MLIR file under `testdata/golden/upstream/` and shape/op-structure expectations in `src/upstream_parity.zig`.
