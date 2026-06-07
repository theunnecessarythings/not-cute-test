# API and architecture parity integration

This file is generated from the uploaded CuteDSL source tree and the current Zig sources.

## Source API inventory

- public top-level records: 1327
- classes: 456
- functions: 871
- public Zig name matches: 155

The complete machine-readable manifest is in `docs/api_surface_manifest.json` and is compiled into `src/api_manifest.zig`. A name match is not a semantic implementation claim; it is a guardrail that prevents the port from hiding missing public APIs.

## Architecture/nvgpu inventory

- arch/nvgpu public records: 380
- copy records: 96
- MMA records: 51
- trait records: 0
- enum records: 23
- records with extracted validation rules: 23
- records with extracted MLIR type factory names: 57

The complete machine-readable manifest is in `docs/arch_nvgpu_manifest.json` and is compiled into `src/arch_manifest.zig`.

## Status

This integration pass makes the missing API and architecture operation surface explicit and testable inside the Zig library. It does not magically implement every semantic path from the original Python code. Behavior must still be closed symbol-by-symbol against this manifest.
