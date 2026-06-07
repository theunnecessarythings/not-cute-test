const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const atom = @import("atom.zig");
const nvgpu = @import("nvgpu.zig");
const tensor_ssa = @import("tensor_ssa.zig");
const copy_mma = @import("copy_mma.zig");
const mlir_harness = @import("mlir_harness.zig");
const cutlass_emit = @import("cutlass_emit.zig");

pub const Error = copy_mma.Error || mlir_harness.Error || cutlass_emit.Error || nvgpu.Error || error{InvalidRoutedFixture};

pub const RoutedFixtureKind = enum {
    tensor_vector,
    copy_atom,
    mma_atom,
};

pub const RoutedFixture = struct {
    name: []const u8,
    kind: RoutedFixtureKind,
    mlir_text: []const u8,

    pub fn validate(self: RoutedFixture) Error!void {
        if (self.name.len == 0 or self.mlir_text.len == 0)
            return Error.InvalidRoutedFixture;
        try mlir_harness.validateGeneratedMlir(self.mlir_text);
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_load") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_store") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_copy_") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_mma_") != null)
            return Error.InvalidRoutedFixture;
    }
};

pub const tensor_vector_fixture =
    \\module {
    \\  func.func @routed_tensor_vector(%arg0: !cute.memref<f32, gmem, "(4):(1)">) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, "(4):(1)">) -> vector<4xf32>
    \\    cute.memref.store_vec(%0, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const copy_atom_fixture =
    \\module {
    \\  func.func @routed_copy_atom(%arg0: !cute.memref<f32, gmem, "(1):(1)">, %arg1: !cute.memref<f32, gmem, "(1):(1)">) {
    \\    %0 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%0, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, "(1):(1)">, !cute.memref<f32, gmem, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const mma_atom_fixture =
    \\module {
    \\  func.func @routed_mma_atom(%arg0: !cute.memref<f32, generic, "(1):(1)">, %arg1: !cute.memref<f32, generic, "(1):(1)">, %arg2: !cute.memref<f32, generic, "(1):(1)">, %arg3: !cute.memref<f32, generic, "(1):(1)">) {
    \\    %0 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%0, %arg3, %arg0, %arg1, %arg2) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const routed_fixtures = [_]RoutedFixture{
    .{
        .name = "cutlass_routed_tensor_vector",
        .kind = .tensor_vector,
        .mlir_text = tensor_vector_fixture,
    },
    .{
        .name = "cutlass_routed_copy_atom",
        .kind = .copy_atom,
        .mlir_text = copy_atom_fixture,
    },
    .{
        .name = "cutlass_routed_mma_atom",
        .kind = .mma_atom,
        .mlir_text = mma_atom_fixture,
    },
};

pub fn fixtureByName(name: []const u8) ?RoutedFixture {
    for (routed_fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn emitTensorVectorModule(out: anytype) Error!void {
    var builder: mlir.Builder(4096) = .{};
    const layout_value = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{4}));
    const meta = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(0) },
        layout_value,
        typing.Float32,
        .gmem,
    );
    var memref_ty_buf: mlir.TextBuffer(512) = .{};
    try meta.cutlassTensorTypeText(&memref_ty_buf);
    try builder.beginModule();
    try builder.beginFunc(
        "routed_tensor_vector",
        &.{mlir.Type.raw(memref_ty_buf.slice())},
        null,
    );
    const tv = tensor_ssa.TensorValue.init(
        meta,
        mlir.Value.arg(0),
        memref_ty_buf.slice(),
    );
    const loaded = try tv.load(&builder, null, null);
    try tv.store(&builder, loaded, null);
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    const text = try builder.finish();
    try out.append(text);
}

fn makeCopyAtom() Error!atom.CopyAtom {
    const thr = layout.makeCompactLayout(.{1});
    const tv = layout.makeCompactLayout(.{ 1, 1 });
    var tr: atom.Trait = .{ .name = "routed_copy_trait", .thr_id = thr };
    tr = tr.withCopyLayouts(tv, tv);
    const desc = atom.OpDescriptor.copyTyped(
        "CopyUniversalOp",
        "generic",
        "simt.sync.copy",
        typing.Float32,
        .gmem,
        .gmem,
        32,
        &.{},
    );
    return atom.makeCopyAtom(desc, tr);
}

pub fn emitCopyAtomModule(out: anytype) Error!void {
    var builder: mlir.Builder(4096) = .{};
    const layout_value = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{1}));
    const src_meta = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(0) },
        layout_value,
        typing.Float32,
        .gmem,
    );
    const dst_meta = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(1) },
        layout_value,
        typing.Float32,
        .gmem,
    );
    var ty_buf: mlir.TextBuffer(512) = .{};
    try src_meta.cutlassTensorTypeText(&ty_buf);
    try builder.beginModule();
    try builder.beginFunc(
        "routed_copy_atom",
        &.{ mlir.Type.raw(ty_buf.slice()), mlir.Type.raw(ty_buf.slice()) },
        null,
    );
    const src = tensor_ssa.TensorValue.init(
        src_meta,
        mlir.Value.arg(0),
        ty_buf.slice(),
    );
    const dst = tensor_ssa.TensorValue.init(
        dst_meta,
        mlir.Value.arg(1),
        ty_buf.slice(),
    );
    _ = try copy_mma.lowerCopyAtom(&builder, try makeCopyAtom(), src, dst, null);
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    const text = try builder.finish();
    try out.append(text);
}

pub fn emitMmaAtomModule(out: anytype) Error!void {
    var builder: mlir.Builder(8192) = .{};
    const layout_value = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{1}));
    const meta_a = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(0) },
        layout_value,
        typing.Float32,
        .generic,
    );
    const meta_b = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(1) },
        layout_value,
        typing.Float32,
        .generic,
    );
    const meta_c = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(2) },
        layout_value,
        typing.Float32,
        .generic,
    );
    const meta_d = try tensor_ssa.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(3) },
        layout_value,
        typing.Float32,
        .generic,
    );
    var ty_buf: mlir.TextBuffer(512) = .{};
    try meta_a.cutlassTensorTypeText(&ty_buf);
    try builder.beginModule();
    try builder.beginFunc(
        "routed_mma_atom",
        &.{
            mlir.Type.raw(ty_buf.slice()),
            mlir.Type.raw(ty_buf.slice()),
            mlir.Type.raw(ty_buf.slice()),
            mlir.Type.raw(ty_buf.slice()),
        },
        null,
    );
    const a = tensor_ssa.TensorValue.init(meta_a, mlir.Value.arg(0), ty_buf.slice());
    const b = tensor_ssa.TensorValue.init(meta_b, mlir.Value.arg(1), ty_buf.slice());
    const c = tensor_ssa.TensorValue.init(meta_c, mlir.Value.arg(2), ty_buf.slice());
    const d = tensor_ssa.TensorValue.init(meta_d, mlir.Value.arg(3), ty_buf.slice());
    _ = try copy_mma.lowerMmaAtom(
        &builder,
        try nvgpu.universalMma(typing.Float32),
        d,
        a,
        b,
        c,
    );
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    const text = try builder.finish();
    try out.append(text);
}

pub fn emitByName(name: []const u8, out: anytype) Error!void {
    if (std.mem.eql(u8, name, "cutlass_routed_tensor_vector"))
        return emitTensorVectorModule(out);
    if (std.mem.eql(u8, name, "cutlass_routed_copy_atom"))
        return emitCopyAtomModule(out);
    if (std.mem.eql(u8, name, "cutlass_routed_mma_atom")) return emitMmaAtomModule(out);
    return Error.InvalidRoutedFixture;
}

pub fn writeAllGenerated(out: anytype) Error!void {
    inline for (.{
        "cutlass_routed_tensor_vector",
        "cutlass_routed_copy_atom",
        "cutlass_routed_mma_atom",
    }) |name| {
        try out.append("// ----- ");
        try out.append(name);
        try out.append(" -----\n");
        try emitByName(name, out);
    }
}

pub fn writeStatus(out: anytype) Error!void {
    try out.append("Routed CUTLASS emission connects tensor vector load/store and copy/MMA atom lowering to parser-aligned emitters. ");
    try out.append("Generated modules use !cute.memref, cute.memref.load_vec/store_vec, cute.copy_atom_call, and cute.mma_atom_call forms accepted by the installed CUTLASS DSL parser.\n");
}

test "cutlass_routed static routed fixtures are placeholder-free" {
    for (routed_fixtures) |fixture| {
        try fixture.validate();
    }
}

test "cutlass_routed generated tensor module uses parser-aligned memref vector ops" {
    var out: mlir.TextBuffer(4096) = .{};
    try emitTensorVectorModule(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.memref.load_vec(%arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.memref.store_vec(%0, %arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") == null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.memref_load_vec") == null);
}

test "cutlass_routed generated copy module uses parser-aligned atom call" {
    var out: mlir.TextBuffer(4096) = .{};
    try emitCopyAtomModule(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.make_atom()") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.copy_atom_call(%0, %arg0, %arg1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") == null);
}

test "cutlass_routed generated mma module uses parser-aligned atom call" {
    var out: mlir.TextBuffer(8192) = .{};
    try emitMmaAtomModule(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute_nvgpu.atom.universal_fma<1x1x1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.mma_atom_call(%0, %arg3, %arg0, %arg1, %arg2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") == null);
}

test "cutlass_routed status names routing boundary" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "connects tensor") != null);
}
