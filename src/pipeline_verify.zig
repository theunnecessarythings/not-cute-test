const std = @import("std");
const mlir = @import("mlir_text.zig");
const mlir_harness = @import("mlir_harness.zig");
const cutlass_bridge = @import("cutlass_bridge.zig");
const compile_pipeline = @import("compile_pipeline.zig");

pub const Error = cutlass_bridge.Error || compile_pipeline.Error || mlir_harness.Error || error{
    InvalidVerifierCase,
    TooManyVerifierCases,
};

pub const VerifyMode = enum { parse, canonicalize, cute_to_nvvm, lir_to_cute_to_nvvm, expect_fail };
pub const VerifyShard = enum { layout, tensor, copy, mma, tiled, negative, all };

pub const VerifyCase = struct {
    name: []const u8,
    shard: VerifyShard,
    input: []const u8,
    mode: VerifyMode = .parse,
    expected_diagnostic: ?[]const u8 = null,

    pub fn validate(self: VerifyCase) Error!void {
        if (self.name.len == 0 or self.input.len == 0) return Error.InvalidVerifierCase;
        if (self.mode == .expect_fail and self.expected_diagnostic == null) return Error.InvalidVerifierCase;
    }
};

pub const cutlass_cases = [_]VerifyCase{
    .{ .name = "layout-parse", .shard = .layout, .input = "testdata/cutlass/layout_case.mlir", .mode = .parse },
    .{ .name = "identity-tensor-parse", .shard = .tensor, .input = "testdata/cutlass/identity_tensor_case.mlir", .mode = .parse },
    .{ .name = "memref-load-parse", .shard = .tensor, .input = "testdata/cutlass/memref_load_case.mlir", .mode = .parse },
    .{ .name = "tensor-vector-canonicalize", .shard = .tensor, .input = "testdata/cutlass/cutlass_routed_tensor_vector.mlir", .mode = .canonicalize },
    .{ .name = "copy-atom-canonicalize", .shard = .copy, .input = "testdata/cutlass/cutlass_routed_copy_atom.mlir", .mode = .canonicalize },
    .{ .name = "mma-atom-canonicalize", .shard = .mma, .input = "testdata/cutlass/cutlass_routed_mma_atom.mlir", .mode = .canonicalize },
    .{ .name = "tiled-copy-canonicalize", .shard = .tiled, .input = "testdata/cutlass/tiled_emit_full_tiled_copy.mlir", .mode = .canonicalize },
    .{ .name = "tiled-mma-canonicalize", .shard = .tiled, .input = "testdata/cutlass/tiled_emit_full_tiled_mma.mlir", .mode = .canonicalize },
    .{ .name = "kernel-tiled-copy-cute-to-nvvm", .shard = .tiled, .input = "testdata/cutlass/kernel_tiled_copy.mlir", .mode = .cute_to_nvvm },
    .{ .name = "fake-tensor-negative", .shard = .negative, .input = "testdata/cutlass/negative_fake_tensor.mlir", .mode = .expect_fail, .expected_diagnostic = "unknown  type `tensor` in dialect `cute`" },
};

pub fn pipelineFor(mode: VerifyMode, out: anytype) Error!void {
    switch (mode) {
        .parse, .expect_fail => try out.append("parse-only"),
        .canonicalize => try out.append("builtin.module(canonicalize)"),
        .cute_to_nvvm => {
            const req: compile_pipeline.CompileRequest = .{ .input_mlir = "unused.mlir", .work_dir = "zig-cache/not-cute", .function_name = "kernel", .pipeline_kind = .cute_to_nvvm };
            try req.writePipeline(out);
        },
        .lir_to_cute_to_nvvm => {
            const req: compile_pipeline.CompileRequest = .{ .input_mlir = "unused.mlir", .work_dir = "zig-cache/not-cute", .function_name = "kernel", .pipeline_kind = .lir_to_cute_to_nvvm };
            try req.writePipeline(out);
        },
    }
}

pub fn invocationFor(config: cutlass_bridge.PythonBridgeConfig, case: VerifyCase) Error!mlir_harness.Invocation {
    try case.validate();
    try config.validate();
    var inv = mlir_harness.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    switch (case.mode) {
        .parse => {
            try inv.append("parse");
            try inv.append("--module");
            try inv.append(config.package_module);
            try inv.append("--input");
            try inv.append(case.input);
        },
        .canonicalize, .cute_to_nvvm, .lir_to_cute_to_nvvm => {
            var pipeline: mlir.TextBuffer(4096) = .{};
            try pipelineFor(case.mode, &pipeline);
            try inv.append("verify");
            try inv.append("--module");
            try inv.append(config.package_module);
            try inv.append("--input");
            try inv.append(case.input);
            try inv.append("--pipeline");
            try inv.append(pipeline.slice());
            try inv.append("--enable-verifier");
        },
        .expect_fail => {
            try inv.append("expect-fail");
            try inv.append("--module");
            try inv.append(config.package_module);
            try inv.append("--input");
            try inv.append(case.input);
            try inv.append("--expected");
            try inv.append(case.expected_diagnostic.?);
        },
    }
    return inv;
}

pub fn writeShardScript(config: cutlass_bridge.PythonBridgeConfig, shard: VerifyShard, out: anytype) Error!void {
    var emitted: usize = 0;
    for (cutlass_cases) |case| {
        if (shard != .all and case.shard != shard) continue;
        const inv = try invocationFor(config, case);
        if (emitted != 0) try out.append(" && \\\n  ");
        try inv.writeShell(out);
        emitted += 1;
    }
    if (emitted == 0) return Error.InvalidVerifierCase;
}

pub fn countCases(shard: VerifyShard) usize {
    var n: usize = 0;
    for (cutlass_cases) |case| {
        if (shard == .all or case.shard == shard) n += 1;
    }
    return n;
}

pub fn writeVerifierManifest(out: anytype) Error!void {
    try out.append("{\"total\":");
    try out.appendUnsigned(cutlass_cases.len);
    try out.append(",\"layout\":");
    try out.appendUnsigned(countCases(.layout));
    try out.append(",\"tensor\":");
    try out.appendUnsigned(countCases(.tensor));
    try out.append(",\"copy\":");
    try out.appendUnsigned(countCases(.copy));
    try out.append(",\"mma\":");
    try out.appendUnsigned(countCases(.mma));
    try out.append(",\"tiled\":");
    try out.appendUnsigned(countCases(.tiled));
    try out.append(",\"negative\":");
    try out.appendUnsigned(countCases(.negative));
    try out.append("}");
}

test "pipeline_verify: sharded scripts include parser and pipeline commands" {
    var out: mlir.TextBuffer(8192) = .{};
    try writeShardScript(.{}, .tiled, &out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "tiled_emit_full_tiled_copy") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "builtin.module(canonicalize)") != null);
    try std.testing.expectEqual(@as(usize, 3), countCases(.tiled));
}

test "pipeline_verify: manifest captures all shards" {
    var out: mlir.TextBuffer(512) = .{};
    try writeVerifierManifest(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "\"total\":10") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "\"negative\":1") != null);
}
