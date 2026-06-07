# Upstream CuTeDSL example parity

The uploaded CuTeDSL library tree did not contain a tests/ or examples/ directory. The upstream packaged CUTLASS examples used for parity are the CuTeDSL notebooks and FFI tensor example.

| Upstream example | Zig port | Golden MLIR | Status | Notes |
|---|---|---|---|---|
| `examples/python/CuTeDSL/notebooks/hello_world.ipynb` | `examples/upstream/hello_world.zig` | `testdata/golden/upstream/hello_world.mlir` | ported_parser_checked | GPU kernel/host launch tutorial mapped to Zig kernel module plus launch-plan metadata; runtime launch remains deferred. |
| `examples/python/CuTeDSL/notebooks/print.ipynb` | `examples/upstream/print_values.zig` | `testdata/golden/upstream/print_values.mlir` | ported_parser_checked | Static/dynamic printing is represented by deterministic arithmetic/metadata MLIR; actual device printf formatting is deferred to runtime execution. |
| `examples/python/CuTeDSL/notebooks/data_types.ipynb` | `examples/upstream/data_types.zig` | `testdata/golden/upstream/data_types.mlir` | ported_parser_checked | Numeric type conversion/operator tutorial mapped to explicit arith operations and type markers. |
| `examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb` | `examples/upstream/layout_algebra.zig` | `testdata/golden/upstream/layout_algebra.mlir` | ported_parser_checked | Layout algebra tutorial mapped to real cute.make_shape/stride/layout forms accepted by CUTLASS DSL. |
| `examples/python/CuTeDSL/notebooks/tensor.ipynb` | `examples/upstream/tensor.zig` | `testdata/golden/upstream/tensor.mlir` | ported_parser_checked | Pointer-backed tensor construction/fill/load-store mapped to memref scalar load/store kernel. |
| `examples/python/CuTeDSL/notebooks/tensorssa.ipynb` | `examples/upstream/tensorssa.zig` | `testdata/golden/upstream/tensorssa.mlir` | ported_parser_checked | TensorSSA load/arithmetic/store tutorial mapped to vector load/add/store MLIR. |
| `examples/python/CuTeDSL/notebooks/elementwise_add.ipynb` | `examples/upstream/elementwise_add.zig` | `testdata/golden/upstream/elementwise_add.mlir` | ported_parser_checked | Naive elementwise add tutorial mapped to a full vector add GPU module. |
| `examples/python/CuTeDSL/notebooks/cuda_graphs.ipynb` | `examples/upstream/cuda_graphs.zig` | `testdata/golden/upstream/cuda_graphs.mlir` | ported_dry_run | CUDA graph capture/replay requires PyTorch CUDA graph runtime; Zig port records launch-plan/dry-run structure, not actual capture. |
| `examples/python/CuTeDSL/cute/ffi/tensor.cpp` | `examples/upstream/ffi_tensor.zig` | `testdata/golden/upstream/ffi_tensor.mlir` | ported_parser_checked | FFI tensor pointer view mapped to typed memref load/store and memory-model external pointer descriptors. |
