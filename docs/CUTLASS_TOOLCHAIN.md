# CUTLASS DSL toolchain bridge

The optional bridge uses the Python package `nvidia-cutlass-dsl`.

After installation, the package provides:

- `cutlass._mlir.ir`
- `cutlass._mlir.passmanager`
- `cutlass._mlir.execution_engine`
- `_mlir/_mlir_libs/_cutlass_ir.cpython*.so`
- `libcute_dsl_runtime.so`

The bridge script is `tools/cutlass_mlir_bridge.py`. It is intentionally an external helper; the Zig library does not link Python or MLIR by default.

Useful commands:

```sh
python3 -m pip install nvidia-cutlass-dsl
python3 tools/cutlass_mlir_bridge.py discover --json
python3 tools/cutlass_mlir_bridge.py ops --dialect cute
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir
```

`pip install nvidia-cutlass` alone is not enough for this bridge; in this sandbox it installed `cutlass_cppgen`/`cutlass_library` but not the importable `cutlass._mlir` stack.
