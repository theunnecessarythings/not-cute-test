const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const runtime = @import("runtime.zig");
const mlir = @import("mlir_text.zig");
const atom = @import("atom.zig");
const tensor_ssa = @import("tensor_ssa.zig");
const copy_mma = @import("copy_mma.zig");
const options = @import("build_options");

pub const layout_case_mlir =
    \\module {
    \\  func.func @layout_case() {
    \\    %0 = cute.make_shape() : () -> !cute.shape<"(2,3)">
    \\    %1 = cute.make_stride() : () -> !cute.stride<"(3,1)">
    \\    %2 = cute.make_layout(%0, %1) : !cute.layout<"(2,3):(3,1)">
    \\    return
    \\  }
    \\}
    \\
;

pub const tensor_case_mlir =
    \\module {
    \\  func.func @tensor_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: vector<4xf32>) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">) -> vector<4xf32>
    \\    %1 = arith.addf %0, %arg1 : vector<4xf32>
    \\    cute.memref.store_vec(%1, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, align<16>, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const copy_case_mlir =
    \\module {
    \\  func.func @copy_case(%arg0: !cute.memref<f32, gmem, align<16>, "(1):(1)">, %arg1: !cute.memref<f32, gmem, align<16>, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%atom, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, align<16>, "(1):(1)">, !cute.memref<f32, gmem, align<16>, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const mma_case_mlir =
    \\module {
    \\  func.func @mma_case(%arg0: !cute.memref<f32, generic, "(1):(1)">, %arg1: !cute.memref<f32, generic, "(1):(1)">, %arg2: !cute.memref<f32, generic, "(1):(1)">, %arg3: !cute.memref<f32, generic, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%atom, %arg3, %arg0, %arg1, %arg2) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const Error = copy_mma.Error || error{
    EmptyCase,
    GoldenMismatch,
    MissingExpectedDiagnostic,
    NegativeTestUnexpectedSuccess,
    ToolNotConfigured,
    ToolFailed,
    TooManyArguments,
    InvalidToolConfig,
};

pub const ToolKind = enum {
    cute_opt,
    mlir_opt,
    filecheck,
    custom,
};

pub const MlirCaseKind = enum {
    layout,
    tensor,
    copy,
    mma,
    negative,
};

pub const ToolConfig = struct {
    cute_opt: []const u8 = options.cute_opt_path,
    mlir_opt: []const u8 = options.mlir_opt_path,
    filecheck: []const u8 = options.filecheck_path,
    enable_external_tools: bool = options.enable_mlir_tools,
    assume_tools_present: bool = options.assume_mlir_tools_present,
    max_output_bytes: usize = 1 << 20,

    pub fn pathFor(
        self: ToolConfig,
        kind: ToolKind,
        custom_path: ?[]const u8,
    ) Error![]const u8 {
        return switch (kind) {
            .cute_opt => self.cute_opt,
            .mlir_opt => self.mlir_opt,
            .filecheck => self.filecheck,
            .custom => custom_path orelse Error.InvalidToolConfig,
        };
    }

    pub fn shouldRunExternal(self: ToolConfig) bool {
        return self.enable_external_tools or self.assume_tools_present;
    }
};

pub const Invocation = struct {
    argv: [32][]const u8 = undefined,
    argc: usize = 0,

    pub fn init() Invocation {
        return .{};
    }

    pub fn append(self: *Invocation, arg: []const u8) Error!void {
        if (self.argc >= self.argv.len) return Error.TooManyArguments;
        self.argv[self.argc] = arg;
        self.argc += 1;
    }

    pub fn args(self: *const Invocation) []const []const u8 {
        return self.argv[0..self.argc];
    }

    pub fn writeShell(self: *const Invocation, out: anytype) !void {
        for (self.args(), 0..) |arg, i| {
            if (i != 0) try out.append(" ");
            try appendShellQuoted(out, arg);
        }
    }
};

pub const GoldenCase = struct {
    name: []const u8,
    kind: MlirCaseKind,
    mlir_text: []const u8,
    expect_failure: bool = false,
    expected_diagnostic: ?[]const u8 = null,
};

pub fn expectGolden(actual: []const u8, expected: []const u8) Error!void {
    if (!std.mem.eql(u8, actual, expected)) return Error.GoldenMismatch;
}

pub fn expectContains(haystack: []const u8, needle: []const u8) Error!void {
    if (std.mem.indexOf(u8, haystack, needle) == null)
        return Error.MissingExpectedDiagnostic;
}

pub fn validateGeneratedMlir(text: []const u8) Error!void {
    if (text.len == 0) return Error.EmptyCase;
    try mlir.validateBalancedText(text);
    try expectContains(text, "module");
}

pub fn cuteOptVerifyInvocation(
    config: ToolConfig,
    input_path: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.cute_opt, null));
    try inv.append("--verify-diagnostics");
    try inv.append(input_path);
    return inv;
}

pub fn cuteOptPipelineInvocation(
    config: ToolConfig,
    input_path: []const u8,
    output_path: []const u8,
    pipeline: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.cute_opt, null));
    try inv.append(pipeline);
    try inv.append(input_path);
    try inv.append("-o");
    try inv.append(output_path);
    return inv;
}

pub fn mlirOptPipelineInvocation(
    config: ToolConfig,
    input_path: []const u8,
    output_path: []const u8,
    pass_pipeline: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.mlir_opt, null));
    try inv.append(pass_pipeline);
    try inv.append(input_path);
    try inv.append("-o");
    try inv.append(output_path);
    return inv;
}

pub fn fileCheckInvocation(
    config: ToolConfig,
    input_path: []const u8,
    check_file: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.filecheck, null));
    try inv.append(check_file);
    try inv.append("--input-file");
    try inv.append(input_path);
    return inv;
}

pub fn emitLayoutCase(builder: anytype) Error!void {
    try builder.append(layout_case_mlir);
}

pub fn emitTensorCase(builder: anytype) Error!void {
    try builder.append(tensor_case_mlir);
}

pub fn emitCopyCase(builder: anytype) Error!void {
    try builder.append(copy_case_mlir);
}

pub fn emitMmaCase(builder: anytype) Error!void {
    try builder.append(mma_case_mlir);
}

pub fn emitNegativeCase(builder: anytype) Error!void {
    // Deliberately unbalanced and malformed.  This case is for external verifier
    // negative tests and must not be passed through Builder.finish().
    try builder.rawLine("module {");
    try builder.rawLine("  func.func @negative_case(%arg0: i32) {");
    try builder.rawLine("    %0 = arith.addi %arg0, %arg0 : (i32) -> i32");
    try builder.rawLine("    // expected-error {{malformed return}}");
}

fn tensorValue(meta: tensor_ssa.TensorMeta, value: mlir.Value) tensor_ssa.TensorValue {
    return tensor_ssa.TensorValue.init(meta, value, "");
}

fn makeGenericCopyAtom(
    dtype: typing.Numeric,
    src_space: typing.AddressSpace,
    dst_space: typing.AddressSpace,
) Error!atom.CopyAtom {
    const thr = layout.makeCompactLayout(.{4});
    const tv = layout.makeCompactLayout(.{ 4, 1 });
    var tr: atom.Trait = .{ .name = "copy", .thr_id = thr };
    tr = tr.withCopyLayouts(tv, tv);
    return atom.makeCopyAtom(
        atom.OpDescriptor.copyTyped("copy", "generic", "unit", dtype, src_space, dst_space, dtype.width, &.{}),
        tr,
    );
}

fn makeGenericMmaAtom() Error!atom.MmaAtom {
    const thr = layout.makeCompactLayout(.{32});
    const tv = layout.makeCompactLayout(.{ 32, 1 });
    var tr: atom.Trait = .{
        .name = "mma",
        .thr_id = thr,
        .shape_mnk = layout.Tree.fromComptime(.{ 16, 8, 8 }),
    };
    tr = tr.withMmaLayouts(tv, tv, tv);
    return atom.makeMmaAtom(
        atom.OpDescriptor.mmaTyped("mma", "generic", "unit", layout.Tree.fromComptime(.{
            16,
            8,
            8,
        }), typing.Float16, typing.Float16, typing.Float32, &.{.accumulate}),
        tr,
    );
}

fn appendShellQuoted(out: anytype, arg: []const u8) !void {
    if (arg.len == 0) {
        try out.append("''");
        return;
    }
    var needs_quote = false;
    for (arg) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '/' or c == '.' or c == '=' or c == ':' or c == ',')) {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        try out.append(arg);
        return;
    }
    try out.append("'");
    for (arg) |c| {
        if (c == '\'') try out.append("'\\''") else try out.appendByte(c);
    }
    try out.append("'");
}

test "mlir_harness: deterministic layout golden case" {
    var b: mlir.Builder(4096) = .{};
    try emitLayoutCase(&b);
    _ = try b.finish();
    const expected = @embedFile("testdata/golden/layout_case.mlir");
    try std.testing.expectEqualStrings(expected, b.slice());
    try validateGeneratedMlir(b.slice());
}

test "mlir_harness: deterministic tensor golden case" {
    var b: mlir.Builder(8192) = .{};
    try emitTensorCase(&b);
    _ = try b.finish();
    try expectContains(b.slice(), "cute.memref.load_vec");
    try expectContains(b.slice(), "cute.memref.store_vec");
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.memref_load_vec") == null);
}

test "mlir_harness: deterministic copy and mma golden cases" {
    var b: mlir.Builder(16384) = .{};
    try emitCopyCase(&b);
    _ = try b.finish();
    try expectContains(b.slice(), "cute.make_atom()");
    try expectContains(b.slice(), "cute.copy_atom_call(");
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "!cute.tensor") == null);

    b.reset();
    try emitMmaCase(&b);
    _ = try b.finish();
    try expectContains(b.slice(), "cute.make_atom()");
    try expectContains(b.slice(), "cute.mma_atom_call(");
}

test "mlir_harness: negative golden case intentionally fails local structural validation" {
    var b: mlir.Builder(2048) = .{};
    try emitNegativeCase(&b);
    const expected = @embedFile("testdata/golden/negative_case.mlir");
    try std.testing.expectEqualStrings(expected, b.slice());
    try std.testing.expectError(
        mlir.Error.UnbalancedRegion,
        mlir.validateBalancedText(b.slice()),
    );
}

test "mlir_harness: tool invocation builders are deterministic" {
    const config: ToolConfig = .{
        .cute_opt = "/opt/cute/bin/cute-opt",
        .mlir_opt = "/opt/llvm/bin/mlir-opt",
        .filecheck = "/opt/llvm/bin/FileCheck",
    };
    const verify = try cuteOptVerifyInvocation(config, "case.mlir");
    try std.testing.expectEqualStrings("/opt/cute/bin/cute-opt", verify.args()[0]);
    try std.testing.expectEqualStrings("--verify-diagnostics", verify.args()[1]);

    const pipe = try mlirOptPipelineInvocation(
        config,
        "in.mlir",
        "out.mlir",
        "--pass-pipeline=builtin.module(canonicalize,cse)",
    );
    var shell: mlir.TextBuffer(512) = .{};
    try pipe.writeShell(&shell);
    try std.testing.expect(std.mem.indexOf(u8, shell.slice(), "mlir-opt") != null);
    try std.testing.expect(std.mem.indexOf(u8, shell.slice(), "--pass-pipeline") != null);
}
