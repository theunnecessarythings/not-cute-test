# Core API and Architecture Implementation Pass

This integrated pass converts the previous API/architecture audit into additional runnable library code without introducing phase-named modules.

## Added modules

- `src/core.zig`
  - Zig-native, source-named compatibility layer for the core CuteDSL static API.
  - Exposes snake_case public names matching `cutlass.cute.core` where practical.
  - Delegates to the existing semantic modules instead of regenerating code:
    - `tuple.zig`
    - `core_static.zig`
    - `layout_algebra.zig`
    - `layout_core.zig`
    - `basis.zig`
    - `typing.zig`

- `src/arch_ops.zig`
  - Typed, source-named constructor layer for common `cute.arch` / `cute.nvgpu` operations.
  - Provides dispatch by upstream source class name for copy and MMA operation families.
  - Wraps the existing `nvgpu.zig` and `atom.zig` descriptor implementation.

## Core API coverage added

The new `core.zig` module exposes implementations/wrappers for high-value CuteDSL names including:

- tuple/static predicates: `is_tuple`, `is_static`, `is_valid_leaf`, `has_underscore`, `has_scaled_basis`
- shape/tree operations: `get_leaves`, `depth`, `rank`, `get`, `select`, `group_modes`, `slice_`, `dice`
- tuple algebra: `repeat_as_tuple`, `repeat`, `repeat_like`, `flatten`, `product`, `inner_product`, `prefix_product`
- elementwise helpers: `elem_less`, `elem_max`, `elem_min`
- shape math: `shape_div`, `ceil_div`, `round_up`
- layout construction: `make_layout`, `make_identity_layout`, `make_ordered_layout`, `make_layout_like`, `compact_col_major`, `compact_row_major`, `make_composed_layout`
- layout access/transforms: `cosize`, `size_in_bytes`, `coalesce`, `crd2idx`, `idx2crd`, `increment_coord`, `slice_and_offset`, `shape`, `stride`, `composition`, `complement`, `right_inverse`, `left_inverse`
- product/divide transforms: `logical_product`, `zipped_product`, `tiled_product`, `flat_product`, `raked_product`, `blocked_product`, `logical_divide`, `zipped_divide`, `tiled_divide`, `flat_divide`
- locality/common-layout helpers: `local_partition`, `local_tile`, `max_common_layout`, `max_common_vector`, `tile_to_shape`, `make_layout_image_mask`, `leading_dim`, `make_layout_tv`, `nullspace`
- basis/swizzle helpers: `E`, `get_divisibility`, `basis_value`, `basis_get`, `Swizzle`, `Ratio`, `ScaledBasis`
- fast-divmod descriptor: `FastDivmodDivisor`, `fast_divmod_create_divisor`

The implementation is static/Zig-native. Dynamic Python AST behavior is not part of this port target.

## Architecture/nvgpu constructors added

`arch_ops.zig` adds explicit constructors and dispatch for commonly used upstream families:

- MMA:
  - `MmaUniversalOp`
  - `MmaAtomSM80`
  - `MmaAtomSM90`
  - `MmaAtomSM100`
  - `MmaF16BF16Op`
  - `MmaFP8Op`
  - `MmaI8Op`
  - `MmaMXF4Op`
  - generic `makeMmaBySourceName`

- Copy:
  - `CopyUniversalOp`
  - `CopyG2ROp`
  - `CopyR2GOp`
  - `CopyS2ROp`
  - `CopyR2SOp`
  - `CopyG2SOp`
  - `CopyAtomCpAsync`
  - `CopyAtomTmaLoad`
  - `CopyAtomTmaStore`
  - `CopyAtomLdMatrix`
  - `CopyAtomStMatrix`
  - exact class-name wrappers for ldmatrix/stmatrix and tcgen05 load/store/copy families
  - generic `makeCopyBySourceName`

## Updated audit numbers

The source API manifest is regenerated after this pass.

- CuteDSL public source records: `1327`
- previous Zig name matches: `155`
- current Zig name matches: `297`
- implemented `cutlass.cute.*` matches: `254 / 721`

This is real API closure progress, but it is **not full source-wide parity yet**.

## Validation

Debug validation passed:

```sh
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan --summary all
python3 tools/cutlass_mlir_bridge.py discover --json
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/cutlass_routed_tensor_vector.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_copy.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir
```

`src/core.zig` also passed direct ReleaseSafe testing. The full-project ReleaseSafe target still exceeds this sandbox's timeout budget.
