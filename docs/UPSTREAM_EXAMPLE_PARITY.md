# Upstream CuTeDSL fixture parity

The uploaded CuTeDSL library tree did not contain a tests/ or examples/
directory. The packaged notebooks and FFI source are represented as golden
MLIR fixtures and parity metadata, not standalone Zig examples.

| Upstream source | Golden MLIR | Status | Notes |
|---|---|---|---|
| `notebooks/hello_world.ipynb` | `testdata/golden/upstream/hello_world.mlir` | parser_checked | Kernel and launch metadata fixture; runtime launch remains deferred. |
| `notebooks/print.ipynb` | `testdata/golden/upstream/print_values.mlir` | parser_checked | Arithmetic/metadata fixture; device printf is not implemented. |
| `notebooks/data_types.ipynb` | `testdata/golden/upstream/data_types.mlir` | parser_checked | Numeric conversion and operator fixture. |
| `notebooks/cute_layout_algebra.ipynb` | `testdata/golden/upstream/layout_algebra.mlir` | parser_checked | Layout operation fixture accepted by CUTLASS DSL. |
| `notebooks/tensor.ipynb` | `testdata/golden/upstream/tensor.mlir` | parser_checked | Typed memref load/store fixture. |
| `notebooks/tensorssa.ipynb` | `testdata/golden/upstream/tensorssa.mlir` | parser_checked | Vector load/arithmetic/store fixture. |
| `notebooks/elementwise_add.ipynb` | `testdata/golden/upstream/elementwise_add.mlir` | parser_checked | Vector-add GPU module fixture. |
| `notebooks/cuda_graphs.ipynb` | `testdata/golden/upstream/cuda_graphs.mlir` | dry_run | Launch metadata only; no CUDA graph capture. |
| `cute/ffi/tensor.cpp` | `testdata/golden/upstream/ffi_tensor.mlir` | parser_checked | External pointer and typed memref fixture. |
