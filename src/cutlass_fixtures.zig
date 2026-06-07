const std = @import("std");
const mlir = @import("mlir_text.zig");
const mlir_harness = @import("mlir_harness.zig");
const cutlass_bridge = @import("cutlass_bridge.zig");
const arch_catalog = @import("arch_catalog.zig");

pub const Error = cutlass_bridge.Error || arch_catalog.Error || mlir_harness.Error || error{
    InvalidCuteTypePayload,
    InvalidCuteMemorySpace,
    InvalidCutlassFixture,
};

/// This module holds bridge-verified dialect spelling fixtures. Older
/// internal examples previously used project-internal placeholder types
/// such as `!cute.tensor`; CUTLASS DSL does not parse those.  This module keeps
/// those examples as pure Zig goldens and introduces real parser-accepted Cute
/// syntax discovered through the installed `nvidia-cutlass-dsl` package.
pub const FixtureKind = enum {
    builtin,
    layout,
    identity_tensor,
    memref_load,
    negative_fake_tensor,
};

pub const CutlassFixture = struct {
    name: []const u8,
    kind: FixtureKind,
    mlir_text: []const u8,
    expect_parse_failure: bool = false,
    expected_diagnostic: ?[]const u8 = null,

    pub fn validate(self: CutlassFixture) Error!void {
        if (self.name.len == 0 or self.mlir_text.len == 0) return Error.InvalidCutlassFixture;
        try mlir_harness.validateGeneratedMlir(self.mlir_text);
        if (self.expect_parse_failure and self.expected_diagnostic == null) return Error.InvalidCutlassFixture;
    }
};

pub const builtin_fixture_mlir =
    \\module {
    \\  func.func @builtin_case() {
    \\    %0 = arith.constant 7 : i32
    \\    return
    \\  }
    \\}
    \\
;

pub const layout_fixture_mlir =
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

pub const identity_tensor_fixture_mlir =
    \\module {
    \\  func.func @identity_tensor_case() {
    \\    %0 = cute.make_shape() : () -> !cute.shape<"(2,3)">
    \\    %1 = cute.make_identity_tensor(%0) : !cute.coord_tensor<"(0,0)", "(2,3):(1@0,1@1)">
    \\    return
    \\  }
    \\}
    \\
;

pub const memref_load_fixture_mlir =
    \\module {
    \\  func.func @memref_load_case(%arg0: !cute.memref<f32, gmem, align<16>, "(2,3):(3,1)">, %arg1: !cute.coord<"(2,3)">) -> f32 {
    \\    %0 = cute.memref.load(%arg0, %arg1) : (!cute.memref<f32, gmem, align<16>, "(2,3):(3,1)">, !cute.coord<"(2,3)">) -> f32
    \\    return %0 : f32
    \\  }
    \\}
    \\
;

pub const negative_fake_tensor_fixture_mlir =
    \\module {
    \\  func.func @negative_fake_tensor(%arg0: !cute.tensor) {
    \\    return
    \\  }
    \\}
    \\
;

pub const fixtures = [_]CutlassFixture{
    .{ .name = "builtin_case", .kind = .builtin, .mlir_text = builtin_fixture_mlir },
    .{ .name = "layout_case", .kind = .layout, .mlir_text = layout_fixture_mlir },
    .{ .name = "identity_tensor_case", .kind = .identity_tensor, .mlir_text = identity_tensor_fixture_mlir },
    .{ .name = "memref_load_case", .kind = .memref_load, .mlir_text = memref_load_fixture_mlir },
    .{
        .name = "negative_fake_tensor",
        .kind = .negative_fake_tensor,
        .mlir_text = negative_fake_tensor_fixture_mlir,
        .expect_parse_failure = true,
        .expected_diagnostic = "unknown  type `tensor` in dialect `cute`",
    },
};

pub fn fixtureByName(name: []const u8) ?CutlassFixture {
    for (fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn writeShapeType(out: anytype, shape: []const u8) Error!void {
    try validateCutePayload(shape);
    try out.append("!cute.shape<");
    try out.appendQuotedString(shape);
    try out.append(">");
}

pub fn writeStrideType(out: anytype, stride: []const u8) Error!void {
    try validateCutePayload(stride);
    try out.append("!cute.stride<");
    try out.appendQuotedString(stride);
    try out.append(">");
}

pub fn writeLayoutType(out: anytype, shape: []const u8, stride: []const u8) Error!void {
    try validateCutePayload(shape);
    try validateCutePayload(stride);
    try out.append("!cute.layout<");
    try out.appendByte('\"');
    try out.append(shape);
    try out.append(":");
    try out.append(stride);
    try out.appendByte('\"');
    try out.append(">");
}

pub fn writeCoordType(out: anytype, coord: []const u8) Error!void {
    try validateCutePayload(coord);
    try out.append("!cute.coord<");
    try out.appendQuotedString(coord);
    try out.append(">");
}

pub fn writeCoordTensorType(out: anytype, origin: []const u8, layout_text: []const u8) Error!void {
    try validateCutePayload(origin);
    try validateCutePayload(layout_text);
    try out.append("!cute.coord_tensor<");
    try out.appendQuotedString(origin);
    try out.append(", ");
    try out.appendQuotedString(layout_text);
    try out.append(">");
}

pub fn writePtrType(out: anytype, elem: []const u8, memory_space: []const u8, alignment: usize) Error!void {
    try validateElementType(elem);
    try validateMemorySpace(memory_space);
    if (alignment == 0) return Error.InvalidCuteTypePayload;
    try out.append("!cute.ptr<");
    try out.append(elem);
    try out.append(", ");
    try out.append(memory_space);
    if (alignment != 4) {
        try out.append(", align<");
        try out.appendUnsigned(alignment);
        try out.append(">");
    }
    try out.append(">");
}

pub fn writeMemRefType(out: anytype, elem: []const u8, memory_space: []const u8, alignment: usize, layout_text: []const u8) Error!void {
    try validateElementType(elem);
    try validateMemorySpace(memory_space);
    try validateCutePayload(layout_text);
    if (alignment == 0) return Error.InvalidCuteTypePayload;
    try out.append("!cute.memref<");
    try out.append(elem);
    try out.append(", ");
    try out.append(memory_space);
    if (alignment != 4) {
        try out.append(", align<");
        try out.appendUnsigned(alignment);
        try out.append(">");
    }
    try out.append(", ");
    try out.appendQuotedString(layout_text);
    try out.append(">");
}

pub fn writeAllFixtures(out: anytype) Error!void {
    for (fixtures) |fixture| {
        try fixture.validate();
        try out.append("// ----- ");
        try out.append(fixture.name);
        try out.append(" -----\n");
        try out.append(fixture.mlir_text);
    }
}

pub fn cutlassParseInvocation(config: cutlass_bridge.PythonBridgeConfig, input_mlir: []const u8) Error!mlir_harness.Invocation {
    try config.validate();
    if (input_mlir.len == 0) return Error.InvalidBridgeConfig;
    var inv = mlir_harness.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("parse");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--input");
    try inv.append(input_mlir);
    return inv;
}

pub fn cutlassOpsInvocation(config: cutlass_bridge.PythonBridgeConfig, dialect: []const u8) Error!mlir_harness.Invocation {
    try config.validate();
    if (!std.mem.eql(u8, dialect, "cute") and !std.mem.eql(u8, dialect, "cute_nvgpu")) return Error.InvalidBridgeConfig;
    var inv = mlir_harness.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("ops");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--dialect");
    try inv.append(dialect);
    return inv;
}

pub fn validateCutePayload(payload: []const u8) Error!void {
    if (payload.len == 0) return Error.InvalidCuteTypePayload;
    for (payload) |c| switch (c) {
        '0'...'9', 'a'...'z', 'A'...'Z', '(', ')', ',', ':', '@', '_', '-', '+', '*', '/', ' ', '.' => {},
        else => return Error.InvalidCuteTypePayload,
    };
    if (std.mem.indexOfScalar(u8, payload, '"') != null) return Error.InvalidCuteTypePayload;
}

pub fn validateElementType(elem: []const u8) Error!void {
    const valid = [_][]const u8{ "i1", "i8", "i16", "i32", "i64", "f16", "bf16", "tf32", "f32", "f64" };
    for (valid) |candidate| {
        if (std.mem.eql(u8, elem, candidate)) return;
    }
    return Error.InvalidMlirType;
}

pub fn validateMemorySpace(memory_space: []const u8) Error!void {
    const valid = [_][]const u8{ "generic", "gmem", "smem", "rmem", "tmem" };
    for (valid) |candidate| {
        if (std.mem.eql(u8, memory_space, candidate)) return;
    }
    return Error.InvalidCuteMemorySpace;
}

pub fn writeStatus(out: anytype) Error!void {
    try out.append("CUTLASS fixtures align the external bridge with parser-accepted MLIR syntax. ");
    try out.append("`verify-cutlass` now targets real Cute dialect fixtures instead of placeholder `!cute.tensor` examples. ");
    try out.append("The old example goldens remain useful for Zig-internal regression tests but are not passed to CUTLASS until their dialect spelling is upgraded.\n");
}

test "cutlass_fixtures validates parser-aligned fixtures structurally" {
    for (fixtures) |fixture| {
        try fixture.validate();
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "module") != null);
        if (!fixture.expect_parse_failure) {
            try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "!cute.tensor") == null);
        }
    }
}

test "cutlass_fixtures writes CUTLASS Cute type strings" {
    var out: mlir.TextBuffer(512) = .{};
    try writeShapeType(&out, "(2,3)");
    try std.testing.expectEqualStrings("!cute.shape<\"(2,3)\">", out.slice());
    out.clear();
    try writeLayoutType(&out, "(2,3)", "(3,1)");
    try std.testing.expectEqualStrings("!cute.layout<\"(2,3):(3,1)\">", out.slice());
    out.clear();
    try writeMemRefType(&out, "f32", "gmem", 16, "(2,3):(3,1)");
    try std.testing.expectEqualStrings("!cute.memref<f32, gmem, align<16>, \"(2,3):(3,1)\">", out.slice());
}

test "cutlass_fixtures rejects fake tensor syntax as negative fixture" {
    const fixture = fixtureByName("negative_fake_tensor") orelse return Error.InvalidCutlassFixture;
    try std.testing.expect(fixture.expect_parse_failure);
    try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "!cute.tensor") != null);
    try std.testing.expect(fixture.expected_diagnostic != null);
}

test "cutlass_fixtures builds parse and op catalog invocations" {
    const cfg: cutlass_bridge.PythonBridgeConfig = .{ .python_exe = "python", .bridge_script = "tools/cutlass_mlir_bridge.py" };
    const parse = try cutlassParseInvocation(cfg, "testdata/cutlass/layout_case.mlir");
    try std.testing.expectEqualStrings("python", parse.args()[0]);
    try std.testing.expectEqualStrings("parse", parse.args()[2]);
    try std.testing.expectEqualStrings("--input", parse.args()[5]);

    const ops = try cutlassOpsInvocation(cfg, "cute");
    try std.testing.expectEqualStrings("ops", ops.args()[2]);
    try std.testing.expectEqualStrings("cute", ops.args()[6]);
}

test "cutlass_fixtures status documents verify-cutlass fix" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "verify-cutlass") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") != null);
}
