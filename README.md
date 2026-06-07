# not-cute

`not-cute` is a Zig-native, zero-default-dependency port of NVIDIA
CuteDSL/CuTe concepts. It provides static layout algebra, tensor and atom
descriptors, deterministic textual MLIR generation, compile and launch
planning, CUDA Driver API wiring, and CUTLASS parser fixtures.

The project targets Zig 0.16.0. It is an active port, not a production-complete
CuteDSL replacement.

## Quick start

```sh
zig fmt --check build.zig src/*.zig examples/*.zig
zig build test --summary all
zig build examples --summary all
```

The `examples` step installs standalone programs that call the public APIs.

## Library layout

The public library is exposed from `src/root.zig` and grouped into:

- layout and tuple algebra
- MLIR text construction and verification
- tensor, atom, copy, and MMA descriptors
- architecture operation catalogs and validation
- kernel builders and memory models
- compilation, runtime, export, and CUDA execution planning
- CUTLASS bridge, fixture, and parity tooling

See [docs/MODULE_MAP.md](docs/MODULE_MAP.md) for the module inventory and
[docs/API_SURFACE.md](docs/API_SURFACE.md) for the compatibility surface.

## Build targets

List all available targets with:

```sh
zig build --help
```

Common targets include:

```sh
zig build test
zig build examples
zig build launch
```

## Optional CUTLASS validation

CUTLASS validation requires Python and `nvidia-cutlass-dsl`:

```sh
python3 -m pip install nvidia-cutlass-dsl
python3 tools/cutlass_mlir_bridge.py discover --json
zig build verify-cutlass-parse -Dcutlass-python=python3
zig build verify-cutlass-pipeline -Dcutlass-python=python3
```

Artifact compilation additionally depends on a compatible local CUDA/CUTLASS
toolchain:

```sh
zig build verify-cutlass-kernel-cubin -Dcutlass-python=python3
```

## Runtime boundary

`src/cuda_driver.zig` dynamically loads the CUDA Driver API.
`src/execution.zig` handles module loading, function lookup, argument packing,
and `cuLaunchKernel` orchestration. These paths compile and have unit coverage,
but actual GPU execution depends on the host CUDA installation and hardware.

See [docs/RUNTIME_EXECUTION_WIRING.md](docs/RUNTIME_EXECUTION_WIRING.md).

## Port status

Current gaps and compatibility limits are tracked in
[docs/REMAINING_FULL_PORT.md](docs/REMAINING_FULL_PORT.md). Upstream example
coverage is documented in
[docs/UPSTREAM_EXAMPLE_PARITY.md](docs/UPSTREAM_EXAMPLE_PARITY.md).

For the validation commands used on the current tree, see
[VALIDATION.md](VALIDATION.md).
