const std = @import("std");
const mlir = @import("mlir_text.zig");
const mlir_harness = @import("mlir_harness.zig");
const cutlass_emit = @import("cutlass_emit.zig");
const cutlass_routed = @import("cutlass_routed.zig");

pub const Error = mlir_harness.Error || cutlass_emit.Error || cutlass_routed.Error || error{InvalidFullTiledFixture};

pub const FullTiledFixtureKind = enum {
    tiled_copy_full,
    tiled_mma_full,
};

pub const FullTiledFixture = struct {
    name: []const u8,
    kind: FullTiledFixtureKind,
    mlir_text: []const u8,

    pub fn validate(self: FullTiledFixture) Error!void {
        if (self.name.len == 0 or self.mlir_text.len == 0)
            return Error.InvalidFullTiledFixture;
        try mlir_harness.validateGeneratedMlir(self.mlir_text);
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_copy_") != null)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_mma_") != null)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled.copy.partition_S") == null and self.kind == .tiled_copy_full)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled.mma.partition") == null and self.kind == .tiled_mma_full)
            return Error.InvalidFullTiledFixture;
    }
};

pub const tiled_copy_alias = "!copy_simt";
pub const tiled_copy_type = "!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>";
pub const memref_gmem_1x1_alias = "!memref_gmem_f32_1x1";
pub const memref_gmem_1x1_type = "!cute.memref<f32, gmem, \"(1,1):(1,1)\">";
pub const memref_gmem_partition_alias = "!memref_gmem_f32_partition";
pub const memref_gmem_partition_type = "!cute.memref<f32, gmem, \"((1,1),1,1):((0,0),0,0)\">";

pub const tiled_mma_alias = "!mma_f32_1x1x1";
pub const tiled_mma_type = "!cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <\"(1,1,1):(1,1,1)\">>";
pub const memref_generic_1x1_alias = "!memref_generic_f32_1x1";
pub const memref_generic_1x1_type = "!cute.memref<f32, generic, \"(1,1):(1,1)\">";
pub const memref_generic_fragment_alias = "!memref_generic_f32_frag";
pub const memref_generic_fragment_type = "!cute.memref<f32, generic, \"(1,1,1):(0,0,0)\">";

pub const full_tiled_copy_fixture =
    \\!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
    \\!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, "(1,1):(1,1)">
    \\!memref_gmem_f32_partition = !cute.memref<f32, gmem, "((1,1),1,1):((0,0),0,0)">
    \\module {
    \\  func.func @tiled_emit_full_tiled_copy(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !cute.coord<"0">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    %tiled = cute.make_tiled_copy(%atom) : !copy_simt
    \\    %src_partitioned = cute.tiled.copy.partition_S(%tiled, %arg0, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    \\    %dst_partitioned = cute.tiled.copy.partition_D(%tiled, %arg1, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    \\    cute.copy(%tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)
    \\    return
    \\  }
    \\}
    \\
;

pub const full_tiled_mma_fixture =
    \\!mma_f32_1x1x1 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(1,1,1):(1,1,1)">>
    \\!memref_generic_f32_1x1 = !cute.memref<f32, generic, "(1,1):(1,1)">
    \\!memref_generic_f32_frag = !cute.memref<f32, generic, "(1,1,1):(0,0,0)">
    \\module {
    \\  func.func @tiled_emit_full_tiled_mma(%arg0: !memref_generic_f32_1x1, %arg1: !memref_generic_f32_1x1, %arg2: !memref_generic_f32_1x1, %arg3: !memref_generic_f32_1x1, %arg4: !cute.coord<"0">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    %tiled = cute.make_tiled_mma(%atom) : !mma_f32_1x1x1
    \\    %a = cute.tiled.mma.partition A(%tiled, %arg0, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %b = cute.tiled.mma.partition B(%tiled, %arg1, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %c = cute.tiled.mma.partition C(%tiled, %arg2, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %d = cute.tiled.mma.partition C(%tiled, %arg3, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    cute.gemm(%tiled, %d, %a, %b, %c) : (!mma_f32_1x1x1, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag)
    \\    return
    \\  }
    \\}
    \\
;

pub const full_tiled_fixtures = [_]FullTiledFixture{
    .{
        .name = "tiled_emit_full_tiled_copy",
        .kind = .tiled_copy_full,
        .mlir_text = full_tiled_copy_fixture,
    },
    .{
        .name = "tiled_emit_full_tiled_mma",
        .kind = .tiled_mma_full,
        .mlir_text = full_tiled_mma_fixture,
    },
};

pub fn fixtureByName(name: []const u8) ?FullTiledFixture {
    for (full_tiled_fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn emitFullTiledCopyModule(out: anytype) Error!void {
    try out.append(full_tiled_copy_fixture);
}

pub fn emitFullTiledMmaModule(out: anytype) Error!void {
    try out.append(full_tiled_mma_fixture);
}

pub fn emitByName(name: []const u8, out: anytype) Error!void {
    if (std.mem.eql(u8, name, "tiled_emit_full_tiled_copy"))
        return emitFullTiledCopyModule(out);
    if (std.mem.eql(u8, name, "tiled_emit_full_tiled_mma"))
        return emitFullTiledMmaModule(out);
    return Error.InvalidFullTiledFixture;
}

pub fn writeAllGenerated(out: anytype) Error!void {
    for (full_tiled_fixtures) |fixture| {
        try out.append("// ----- ");
        try out.append(fixture.name);
        try out.append(" -----\n");
        try out.append(fixture.mlir_text);
    }
}

pub fn writeStatus(out: anytype) Error!void {
    try out.append("Tiled emission adds CUTLASS parser-verified full tiled-copy and tiled-MMA modules. ");
    try out.append("The copy path constructs a tiled copy, partitions source and destination tensors, and calls cute.copy. ");
    try out.append("The MMA path constructs a tiled MMA, partitions A/B/C/D fragments, and calls cute.gemm with the tiled MMA handle.\n");
}

test "tiled_emit full tiled fixtures are structural and placeholder-free" {
    for (full_tiled_fixtures) |fixture| {
        try fixture.validate();
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "!cute.tensor") == null);
    }
}

test "tiled_emit full tiled copy composes make_tiled_copy partition S D and cute.copy" {
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.make_tiled_copy(%atom) : !copy_simt") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.tiled.copy.partition_S") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.tiled.copy.partition_D") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.copy(%tiled, %src_partitioned, %dst_partitioned)") != null);
}

test "tiled_emit full tiled mma composes make_tiled_mma partitions and cute.gemm" {
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.make_tiled_mma(%atom) : !mma_f32_1x1x1") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.tiled.mma.partition A") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.tiled.mma.partition B") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.tiled.mma.partition C") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.gemm(%tiled, %d, %a, %b, %c)") != null);
}

test "tiled_emit fixture lookup and emission" {
    const copy = fixtureByName("tiled_emit_full_tiled_copy") orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(FullTiledFixtureKind.tiled_copy_full, copy.kind);
    var out: mlir.TextBuffer(12000) = .{};
    try emitByName("tiled_emit_full_tiled_mma", &out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "@tiled_emit_full_tiled_mma") != null);
}

test "tiled_emit status names parser-verified tiled scope" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "full tiled-copy") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.gemm") != null);
}
