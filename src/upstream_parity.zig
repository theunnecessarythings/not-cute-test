const std = @import("std");
const mlir = @import("mlir_text.zig");
const kernel_builders = @import("kernel_builders.zig");

pub const Error = mlir.Error || kernel_builders.Error || error{
    UnknownUpstreamExample,
    MissingExpectedOperation,
    UnsupportedExample,
};

pub const ParityStatus = enum {
    ported_parser_checked,
    ported_dry_run,
    inventory_only,
    unsupported_external_runtime,

    pub fn text(self: ParityStatus) []const u8 {
        return switch (self) {
            .ported_parser_checked => "ported_parser_checked",
            .ported_dry_run => "ported_dry_run",
            .inventory_only => "inventory_only",
            .unsupported_external_runtime => "unsupported_external_runtime",
        };
    }
};

pub const ExampleKind = enum {
    hello_world,
    print_values,
    data_types,
    layout_algebra,
    tensor,
    tensorssa,
    elementwise_add,
    cuda_graphs,
    ffi_tensor,

    pub fn name(self: ExampleKind) []const u8 {
        return switch (self) {
            .hello_world => "hello_world",
            .print_values => "print",
            .data_types => "data_types",
            .layout_algebra => "cute_layout_algebra",
            .tensor => "tensor",
            .tensorssa => "tensorssa",
            .elementwise_add => "elementwise_add",
            .cuda_graphs => "cuda_graphs",
            .ffi_tensor => "ffi_tensor",
        };
    }
};

pub const OperationExpectation = struct {
    text: []const u8,
    min_count: usize = 1,
};

pub const UpstreamExample = struct {
    kind: ExampleKind,
    upstream_path: []const u8,
    zig_path: []const u8,
    golden_path: []const u8,
    status: ParityStatus,
    notes: []const u8,
    expectations: []const OperationExpectation,
};

const hello_expect = [_]OperationExpectation{
    .{ .text = "gpu.module" },
    .{ .text = "gpu.func @hello_world" },
    .{ .text = "gpu.return" },
};
const print_expect = [_]OperationExpectation{
    .{ .text = "arith.constant" },
    .{ .text = "arith.addi" },
    .{ .text = "gpu.func @print_values" },
};
const data_expect = [_]OperationExpectation{
    .{ .text = "arith.constant" },
    .{ .text = "arith.addi" },
    .{ .text = "arith.muli" },
    .{ .text = "gpu.func @data_types" },
};
const layout_expect = [_]OperationExpectation{
    .{ .text = "cute.make_shape" },
    .{ .text = "cute.make_stride" },
    .{ .text = "cute.make_layout" },
};
const tensor_expect = [_]OperationExpectation{
    .{ .text = "cute.memref.load" },
    .{ .text = "cute.memref.store" },
};
const tensorssa_expect = [_]OperationExpectation{
    .{ .text = "cute.memref.load_vec", .min_count = 2 },
    .{ .text = "arith.addf" },
    .{ .text = "cute.memref.store_vec" },
};
const elementwise_expect = [_]OperationExpectation{
    .{ .text = "cute.memref.load_vec", .min_count = 2 },
    .{ .text = "arith.addf" },
    .{ .text = "cute.memref.store_vec" },
};
const graph_expect = [_]OperationExpectation{
    .{ .text = "gpu.module" },
    .{ .text = "gpu.func @cuda_graphs" },
    .{ .text = "gpu.return" },
};
const ffi_expect = [_]OperationExpectation{
    .{ .text = "cute.memref.load" },
    .{ .text = "cute.memref.store" },
};

pub const examples = [_]UpstreamExample{
    .{ .kind = .hello_world, .upstream_path = "examples/python/CuTeDSL/notebooks/hello_world.ipynb", .zig_path = "examples/upstream/hello_world.zig", .golden_path = "testdata/golden/upstream/hello_world.mlir", .status = .ported_parser_checked, .notes = "GPU kernel/host launch tutorial mapped to Zig kernel module plus launch-plan metadata; runtime launch remains deferred.", .expectations = hello_expect[0..] },
    .{ .kind = .print_values, .upstream_path = "examples/python/CuTeDSL/notebooks/print.ipynb", .zig_path = "examples/upstream/print_values.zig", .golden_path = "testdata/golden/upstream/print_values.mlir", .status = .ported_parser_checked, .notes = "Static/dynamic printing is represented by deterministic arithmetic/metadata MLIR; actual device printf formatting is deferred to runtime execution.", .expectations = print_expect[0..] },
    .{ .kind = .data_types, .upstream_path = "examples/python/CuTeDSL/notebooks/data_types.ipynb", .zig_path = "examples/upstream/data_types.zig", .golden_path = "testdata/golden/upstream/data_types.mlir", .status = .ported_parser_checked, .notes = "Numeric type conversion/operator tutorial mapped to explicit arith operations and type markers.", .expectations = data_expect[0..] },
    .{ .kind = .layout_algebra, .upstream_path = "examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb", .zig_path = "examples/upstream/layout_algebra.zig", .golden_path = "testdata/golden/upstream/layout_algebra.mlir", .status = .ported_parser_checked, .notes = "Layout algebra tutorial mapped to real cute.make_shape/stride/layout forms accepted by CUTLASS DSL.", .expectations = layout_expect[0..] },
    .{ .kind = .tensor, .upstream_path = "examples/python/CuTeDSL/notebooks/tensor.ipynb", .zig_path = "examples/upstream/tensor.zig", .golden_path = "testdata/golden/upstream/tensor.mlir", .status = .ported_parser_checked, .notes = "Pointer-backed tensor construction/fill/load-store mapped to memref scalar load/store kernel.", .expectations = tensor_expect[0..] },
    .{ .kind = .tensorssa, .upstream_path = "examples/python/CuTeDSL/notebooks/tensorssa.ipynb", .zig_path = "examples/upstream/tensorssa.zig", .golden_path = "testdata/golden/upstream/tensorssa.mlir", .status = .ported_parser_checked, .notes = "TensorSSA load/arithmetic/store tutorial mapped to vector load/add/store MLIR.", .expectations = tensorssa_expect[0..] },
    .{ .kind = .elementwise_add, .upstream_path = "examples/python/CuTeDSL/notebooks/elementwise_add.ipynb", .zig_path = "examples/upstream/elementwise_add.zig", .golden_path = "testdata/golden/upstream/elementwise_add.mlir", .status = .ported_parser_checked, .notes = "Naive elementwise add tutorial mapped to a full vector add GPU module.", .expectations = elementwise_expect[0..] },
    .{ .kind = .cuda_graphs, .upstream_path = "examples/python/CuTeDSL/notebooks/cuda_graphs.ipynb", .zig_path = "examples/upstream/cuda_graphs.zig", .golden_path = "testdata/golden/upstream/cuda_graphs.mlir", .status = .ported_dry_run, .notes = "CUDA graph capture/replay requires PyTorch CUDA graph runtime; Zig port records launch-plan/dry-run structure, not actual capture.", .expectations = graph_expect[0..] },
    .{ .kind = .ffi_tensor, .upstream_path = "examples/python/CuTeDSL/cute/ffi/tensor.cpp", .zig_path = "examples/upstream/ffi_tensor.zig", .golden_path = "testdata/golden/upstream/ffi_tensor.mlir", .status = .ported_parser_checked, .notes = "FFI tensor pointer view mapped to typed memref load/store and memory-model external pointer descriptors.", .expectations = ffi_expect[0..] },
};

pub fn exampleByKind(kind: ExampleKind) ?UpstreamExample {
    for (examples) |ex| if (ex.kind == kind) return ex;
    return null;
}

pub fn writeMlirForExample(out: anytype, kind: ExampleKind) Error!void {
    switch (kind) {
        .hello_world => try writeHelloWorld(out),
        .print_values => try writePrintValues(out),
        .data_types => try writeDataTypes(out),
        .layout_algebra => try writeLayoutAlgebra(out),
        .tensor => try kernel_builders.writeKernelModule(out, .{ .name = "tensor", .kind = .copy }),
        .tensorssa => try writeVectorAddKernel(out, "tensorssa"),
        .elementwise_add => try writeVectorAddKernel(out, "elementwise_add"),
        .cuda_graphs => try writeCudaGraphsDryRun(out),
        .ffi_tensor => try kernel_builders.writeKernelModule(out, .{ .name = "ffi_tensor", .kind = .copy }),
    }
}

pub fn validateMlirForExample(kind: ExampleKind, text: []const u8) Error!void {
    const ex = exampleByKind(kind) orelse return Error.UnknownUpstreamExample;
    if (std.mem.indexOf(u8, text, "!cute.tensor") != null) return Error.InvalidMlirType;
    for (ex.expectations) |expectation| {
        if (countOccurrences(text, expectation.text) < expectation.min_count) return Error.MissingExpectedOperation;
    }
}

pub fn writeInventoryJson(out: anytype) !void {
    try out.append("{\n  \"source\": \"NVIDIA CUTLASS examples/python/CuTeDSL plus uploaded CuTeDSL library tree\",\n");
    try out.append("  \"uploaded_tree_examples\": 0,\n");
    try out.append("  \"records\": [\n");
    for (examples, 0..) |ex, i| {
        try out.append("    {\"name\": \"");
        try out.append(ex.kind.name());
        try out.append("\", \"upstream_path\": \"");
        try out.append(ex.upstream_path);
        try out.append("\", \"zig_path\": \"");
        try out.append(ex.zig_path);
        try out.append("\", \"golden_path\": \"");
        try out.append(ex.golden_path);
        try out.append("\", \"status\": \"");
        try out.append(ex.status.text());
        try out.append("\"}");
        if (i + 1 != examples.len) try out.append(",");
        try out.append("\n");
    }
    try out.append("  ]\n}\n");
}

pub fn writeMarkdownReport(out: anytype) !void {
    try out.append("# Upstream CuTeDSL example parity\n\n");
    try out.append("The uploaded CuTeDSL library tree did not contain a tests/ or examples/ directory. The upstream packaged CUTLASS examples used for parity are the CuTeDSL notebooks and FFI tensor example.\n\n");
    try out.append("| Upstream example | Zig port | Golden MLIR | Status | Notes |\n");
    try out.append("|---|---|---|---|---|\n");
    for (examples) |ex| {
        try out.append("| `");
        try out.append(ex.upstream_path);
        try out.append("` | `");
        try out.append(ex.zig_path);
        try out.append("` | `");
        try out.append(ex.golden_path);
        try out.append("` | ");
        try out.append(ex.status.text());
        try out.append(" | ");
        try out.append(ex.notes);
        try out.append(" |\n");
    }
}

fn writeHelloWorld(out: anytype) Error!void {
    try out.append("// upstream parity: hello_world.ipynb\n");
    try out.append("module {\n");
    try out.append("  gpu.module @notcute_upstream {\n");
    try out.append("    gpu.func @hello_world() kernel {\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

fn writeCudaGraphsDryRun(out: anytype) Error!void {
    try out.append("// upstream parity: cuda_graphs.ipynb dry-run launch body\n");
    try out.append("module {\n");
    try out.append("  gpu.module @notcute_upstream {\n");
    try out.append("    gpu.func @cuda_graphs() kernel {\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

fn writePrintValues(out: anytype) Error!void {
    try out.append("// upstream parity: print.ipynb static/dynamic value flow\n");
    try out.append("module {\n");
    try out.append("  gpu.module @notcute_upstream {\n");
    try out.append("    gpu.func @print_values() kernel {\n");
    try out.append("      %a = arith.constant 8 : i32\n");
    try out.append("      %b = arith.constant 2 : i32\n");
    try out.append("      %c = arith.addi %a, %b : i32\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

fn writeDataTypes(out: anytype) Error!void {
    try out.append("// upstream parity: data_types.ipynb numeric operations\n");
    try out.append("module {\n");
    try out.append("  gpu.module @notcute_upstream {\n");
    try out.append("    gpu.func @data_types() kernel {\n");
    try out.append("      %a = arith.constant 10 : i32\n");
    try out.append("      %b = arith.constant 3 : i32\n");
    try out.append("      %c = arith.addi %a, %b : i32\n");
    try out.append("      %d = arith.muli %a, %b : i32\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

fn writeLayoutAlgebra(out: anytype) Error!void {
    try out.append("// upstream parity: cute_layout_algebra.ipynb layout construction\n");
    try out.append("module {\n");
    try out.append("  func.func @cute_layout_algebra() {\n");
    try out.append("    %0 = cute.make_shape() : () -> !cute.shape<\"(2,3)\">\n");
    try out.append("    %1 = cute.make_stride() : () -> !cute.stride<\"(3,1)\">\n");
    try out.append("    %2 = cute.make_layout(%0, %1) : !cute.layout<\"(2,3):(3,1)\">\n");
    try out.append("    return\n");
    try out.append("  }\n");
    try out.append("}\n");
}

fn writeVectorAddKernel(out: anytype, name: []const u8) Error!void {
    try out.append("// upstream parity: vector TensorSSA/elementwise-add flow\n");
    try out.append("!memref_vec = !cute.memref<f32, gmem, align<16>, \"(4):(1)\">\n");
    try out.append("module {\n");
    try out.append("  gpu.module @notcute_upstream {\n");
    try out.append("    gpu.func @");
    try out.append(name);
    try out.append("(%a: !memref_vec, %b: !memref_vec, %c: !memref_vec) kernel {\n");
    try out.append("      %av = cute.memref.load_vec(%a) : (!memref_vec) -> vector<4xf32>\n");
    try out.append("      %bv = cute.memref.load_vec(%b) : (!memref_vec) -> vector<4xf32>\n");
    try out.append("      %cv = arith.addf %av, %bv : vector<4xf32>\n");
    try out.append("      cute.memref.store_vec(%cv, %c) : (vector<4xf32>, !memref_vec) -> ()\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

fn countOccurrences(text: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    var count: usize = 0;
    var start: usize = 0;
    while (start <= text.len) {
        const found = std.mem.indexOf(u8, text[start..], needle) orelse break;
        count += 1;
        start += found + needle.len;
    }
    return count;
}

test "upstream_parity: every example emits expected MLIR structure" {
    const kinds = [_]ExampleKind{ .hello_world, .print_values, .data_types, .layout_algebra, .tensor, .tensorssa, .elementwise_add, .cuda_graphs, .ffi_tensor };
    for (kinds) |kind| {
        var out: mlir.TextBuffer(40000) = .{};
        try writeMlirForExample(&out, kind);
        try validateMlirForExample(kind, out.slice());
    }
}

test "upstream_parity: inventory covers all packaged CuTeDSL notebooks plus FFI tensor" {
    try std.testing.expectEqual(@as(usize, 9), examples.len);
    var parser_checked: usize = 0;
    for (examples) |ex| {
        if (ex.status == .ported_parser_checked) parser_checked += 1;
    }
    try std.testing.expect(parser_checked >= 8);
}

test "upstream_parity: report and JSON are emit-ready" {
    var json: mlir.TextBuffer(20000) = .{};
    try writeInventoryJson(&json);
    try std.testing.expect(std.mem.indexOf(u8, json.slice(), "hello_world") != null);
    var md: mlir.TextBuffer(40000) = .{};
    try writeMarkdownReport(&md);
    try std.testing.expect(std.mem.indexOf(u8, md.slice(), "Upstream CuTeDSL example parity") != null);
}

test "upstream_parity: emitted MLIR exactly matches checked-in goldens" {
    const pairs = [_]struct { kind: ExampleKind, golden: []const u8 }{
        .{ .kind = .hello_world, .golden = @embedFile("testdata/golden/upstream/hello_world.mlir") },
        .{ .kind = .print_values, .golden = @embedFile("testdata/golden/upstream/print_values.mlir") },
        .{ .kind = .data_types, .golden = @embedFile("testdata/golden/upstream/data_types.mlir") },
        .{ .kind = .layout_algebra, .golden = @embedFile("testdata/golden/upstream/layout_algebra.mlir") },
        .{ .kind = .tensor, .golden = @embedFile("testdata/golden/upstream/tensor.mlir") },
        .{ .kind = .tensorssa, .golden = @embedFile("testdata/golden/upstream/tensorssa.mlir") },
        .{ .kind = .elementwise_add, .golden = @embedFile("testdata/golden/upstream/elementwise_add.mlir") },
        .{ .kind = .cuda_graphs, .golden = @embedFile("testdata/golden/upstream/cuda_graphs.mlir") },
        .{ .kind = .ffi_tensor, .golden = @embedFile("testdata/golden/upstream/ffi_tensor.mlir") },
    };
    for (pairs) |pair| {
        var out: mlir.TextBuffer(40000) = .{};
        try writeMlirForExample(&out, pair.kind);
        try std.testing.expectEqualStrings(pair.golden, out.slice());
    }
}
