const std = @import("std");
const mlir = @import("mlir_text.zig");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
pub const Error = mlir.Error || layout.Error || error{
    Invalidcutlass_emitFixture,
    InvalidAtomType,
    InvalidTensorType,
    InvalidCuteTypePayload,
    InvalidCuteMemorySpace,
};

/// This module contains generated-MLIR spelling helpers. It keeps the
/// old placeholder goldens out of default verifier paths and exposes a
/// parser-aligned emitter for the tensor/copy/MMA forms that the installed
/// CUTLASS DSL package accepts today.
pub const FixtureKind = enum {
    tensor_scalar,
    tensor_vector,
    copy_atom,
    tiled_copy,
    mma_atom,
};

pub const Fixture = struct {
    name: []const u8,
    kind: FixtureKind,
    mlir_text: []const u8,

    pub fn validate(self: Fixture) Error!void {
        if (self.name.len == 0 or self.mlir_text.len == 0)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "module") == null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_load") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_store") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_copy_") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_mma_") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.mma_make_fragment") != null)
            return Error.Invalidcutlass_emitFixture;
    }
};

pub const f32_gmem_1d = "!cute.memref<f32, gmem, align<16>, \"(4):(1)\">";
pub const f32_gmem_scalar = "!cute.memref<f32, gmem, align<16>, \"(1):(1)\">";
pub const f32_gmem_2d = "!cute.memref<f32, gmem, align<16>, \"(1,1):(1,1)\">";
pub const f32_gmem_partitioned = "!cute.memref<f32, gmem, align<16>, \"((1,1),1,1):((0,0),0,0)\">";
pub const f32_rmem_scalar = "!cute.memref<f32, rmem, \"(1):(1)\">";
pub const coord_1d = "!cute.coord<\"(2)\">";
pub const coord_scalar_zero = "!cute.coord<\"0\">";
pub const universal_copy_f32_32b = "!cute_nvgpu.atom.universal_copy<f32, 32 b>";
pub const universal_fma_f32_1x1x1 = "!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >";
pub const tiled_copy_f32_1x1 = "!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>";

pub const tensor_scalar_fixture =
    \\module {
    \\  func.func @tensor_scalar_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: !cute.coord<"(2)">, %arg2: f32) -> f32 {
    \\    %0 = cute.memref.load(%arg0, %arg1) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">, !cute.coord<"(2)">) -> f32
    \\    cute.memref.store(%arg0, %arg1, %arg2) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">, !cute.coord<"(2)">, f32) -> ()
    \\    return %0 : f32
    \\  }
    \\}
    \\
;

pub const tensor_vector_fixture =
    \\module {
    \\  func.func @tensor_vector_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: vector<4xf32>) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">) -> vector<4xf32>
    \\    %1 = arith.addf %0, %arg1 : vector<4xf32>
    \\    cute.memref.store_vec(%1, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, align<16>, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const copy_atom_fixture =
    \\module {
    \\  func.func @copy_atom_case(%arg0: !cute.memref<f32, gmem, align<16>, "(1):(1)">, %arg1: !cute.memref<f32, gmem, align<16>, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%atom, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, align<16>, "(1):(1)">, !cute.memref<f32, gmem, align<16>, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const tiled_copy_fixture =
    \\!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
    \\module {
    \\  func.func @tiled_copy_case(%arg0: !copy_simt, %arg1: !cute.memref<f32, gmem, align<16>, "(1,1):(1,1)">, %arg2: !cute.coord<"0">) {
    \\    %0 = cute.tiled.copy.partition_S(%arg0, %arg1, %arg2) : (!copy_simt, !cute.memref<f32, gmem, align<16>, "(1,1):(1,1)">, !cute.coord<"0">) -> !cute.memref<f32, gmem, align<16>, "((1,1),1,1):((0,0),0,0)">
    \\    return
    \\  }
    \\}
    \\
;

pub const mma_atom_fixture =
    \\module {
    \\  func.func @mma_atom_case(%arg0: !cute.memref<f32, rmem, "(1):(1)">, %arg1: !cute.memref<f32, rmem, "(1):(1)">, %arg2: !cute.memref<f32, rmem, "(1):(1)">, %arg3: !cute.memref<f32, rmem, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%atom, %arg0, %arg1, %arg2, %arg3) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const fixtures = [_]Fixture{
    .{
        .name = "tensor_scalar_case",
        .kind = .tensor_scalar,
        .mlir_text = tensor_scalar_fixture,
    },
    .{
        .name = "tensor_vector_case",
        .kind = .tensor_vector,
        .mlir_text = tensor_vector_fixture,
    },
    .{ .name = "copy_atom_case", .kind = .copy_atom, .mlir_text = copy_atom_fixture },
    .{
        .name = "tiled_copy_case",
        .kind = .tiled_copy,
        .mlir_text = tiled_copy_fixture,
    },
    .{ .name = "mma_atom_case", .kind = .mma_atom, .mlir_text = mma_atom_fixture },
};

pub fn fixtureByName(name: []const u8) ?Fixture {
    for (fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn writeTreePayload(out: anytype, tree: *const layout.Tree) Error!void {
    try writeTreePayloadSub(out, tree, tree.root);
}

fn writeTreePayloadSub(out: anytype, tree: *const layout.Tree, id: u16) Error!void {
    switch (tree.nodes.at(id)) {
        .leaf => |v| try out.appendSigned(v),
        .tuple => |span| {
            try out.append("(");
            for (0..span.len) |i| {
                if (i != 0) try out.append(",");
                try writeTreePayloadSub(out, tree, tree.children.at(span.start + i));
            }
            try out.append(")");
        },
    }
}

pub fn writeLayoutPayload(out: anytype, value: *const layout.Layout) Error!void {
    try writeTreePayload(out, &value.shape);
    try out.append(":");
    try writeTreePayload(out, &value.stride);
}

pub fn writeMemRefTypeForLayout(
    out: anytype,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace,
    alignment: usize,
    value: *const layout.Layout,
) Error!void {
    var payload: mlir.TextBuffer(512) = .{};
    try writeLayoutPayload(&payload, value);
    try writeMemRefType(
        out,
        if (dtype.width == 1) "i8" else dtype.mlir_type,
        memspace.mlirName(),
        alignment,
        payload.slice(),
    );
}

pub fn memRefTypeForLayout(
    dtype: typing.Numeric,
    memspace: typing.AddressSpace,
    alignment: usize,
    value: *const layout.Layout,
) Error!mlir.TextBuffer(512) {
    var out: mlir.TextBuffer(512) = .{};
    try writeMemRefTypeForLayout(&out, dtype, memspace, alignment, value);
    return out;
}

pub fn writeCoordPayloadFromScalar(out: anytype, offset: layout.Scalar) Error!void {
    try out.appendSigned(offset);
}

pub fn writeCoordTypeFromScalar(out: anytype, offset: layout.Scalar) Error!void {
    var payload: mlir.TextBuffer(96) = .{};
    try writeCoordPayloadFromScalar(&payload, offset);
    try writeCoordType(out, payload.slice());
}

pub fn makeCoordFromScalar(
    builder: anytype,
    offset: layout.Scalar,
) Error!struct { value: mlir.Value, ty: mlir.Type } {
    var ty_buf: mlir.TextBuffer(128) = .{};
    try writeCoordTypeFromScalar(&ty_buf, offset);
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(ty_buf.slice())}, result.id);
    try builder.append("cute.make_coord() : () -> ");
    try builder.append(ty_buf.slice());
    try builder.newline();
    return .{ .value = result, .ty = mlir.Type.raw(ty_buf.slice()) };
}

pub fn emitMemrefLoad(
    builder: anytype,
    memref: mlir.Value,
    coord: mlir.Value,
    memref_ty: mlir.Type,
    coord_ty: mlir.Type,
    elem_ty: mlir.Type,
) Error!mlir.Value {
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{elem_ty}, result.id);
    try builder.append("cute.memref.load(");
    try memref.writeTo(builder);
    try builder.append(", ");
    try coord.writeTo(builder);
    try builder.append(") : (");
    try builder.append(memref_ty.text);
    try builder.append(", ");
    try builder.append(coord_ty.text);
    try builder.append(") -> ");
    try builder.append(elem_ty.text);
    try builder.newline();
    return result;
}

pub fn emitMemrefStore(
    builder: anytype,
    memref: mlir.Value,
    coord: mlir.Value,
    data: mlir.Value,
    memref_ty: mlir.Type,
    coord_ty: mlir.Type,
    elem_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.memref.store(");
    try memref.writeTo(builder);
    try builder.append(", ");
    try coord.writeTo(builder);
    try builder.append(", ");
    try data.writeTo(builder);
    try builder.append(") : (");
    try builder.append(memref_ty.text);
    try builder.append(", ");
    try builder.append(coord_ty.text);
    try builder.append(", ");
    try builder.append(elem_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn emitMemrefLoadVec(
    builder: anytype,
    memref: mlir.Value,
    memref_ty: mlir.Type,
    result_ty: mlir.Type,
) Error!mlir.Value {
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{result_ty}, result.id);
    try builder.append("cute.memref.load_vec(");
    try memref.writeTo(builder);
    try builder.append(") : (");
    try builder.append(memref_ty.text);
    try builder.append(") -> ");
    try builder.append(result_ty.text);
    try builder.newline();
    return result;
}

pub fn emitMemrefStoreVec(
    builder: anytype,
    data: mlir.Value,
    memref: mlir.Value,
    data_ty: mlir.Type,
    memref_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.memref.store_vec(");
    try data.writeTo(builder);
    try builder.append(", ");
    try memref.writeTo(builder);
    try builder.append(") : (");
    try builder.append(data_ty.text);
    try builder.append(", ");
    try builder.append(memref_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn emitMakeUniversalCopyAtom(
    builder: anytype,
    dtype: typing.Numeric,
    bits: usize,
) Error!struct { value: mlir.Value, ty: mlir.Type } {
    var ty_buf: mlir.TextBuffer(256) = .{};
    try writeUniversalCopyAtomType(&ty_buf, dtype.mlir_type, bits);
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(ty_buf.slice())}, result.id);
    try builder.append("cute.make_atom() : () -> ");
    try builder.append(ty_buf.slice());
    try builder.newline();
    return .{ .value = result, .ty = mlir.Type.raw(ty_buf.slice()) };
}

pub fn emitMakeUniversalFmaAtom(
    builder: anytype,
    dtype: typing.Numeric,
    m: usize,
    n: usize,
    k: usize,
) Error!struct { value: mlir.Value, ty: mlir.Type } {
    var ty_buf: mlir.TextBuffer(256) = .{};
    try writeUniversalFmaAtomType(&ty_buf, dtype.mlir_type, m, n, k);
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(ty_buf.slice())}, result.id);
    try builder.append("cute.make_atom() : () -> ");
    try builder.append(ty_buf.slice());
    try builder.newline();
    return .{ .value = result, .ty = mlir.Type.raw(ty_buf.slice()) };
}

pub fn emitCopyAtomCall(
    builder: anytype,
    atom_value: mlir.Value,
    atom_ty: mlir.Type,
    src: mlir.Value,
    dst: mlir.Value,
    src_ty: mlir.Type,
    dst_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.copy_atom_call(");
    try atom_value.writeTo(builder);
    try builder.append(", ");
    try src.writeTo(builder);
    try builder.append(", ");
    try dst.writeTo(builder);
    try builder.append(") : (");
    try builder.append(atom_ty.text);
    try builder.append(", ");
    try builder.append(src_ty.text);
    try builder.append(", ");
    try builder.append(dst_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn emitMmaAtomCall(
    builder: anytype,
    atom_value: mlir.Value,
    atom_ty: mlir.Type,
    d: mlir.Value,
    a: mlir.Value,
    b: mlir.Value,
    c: mlir.Value,
    d_ty: mlir.Type,
    a_ty: mlir.Type,
    b_ty: mlir.Type,
    c_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.mma_atom_call(");
    try atom_value.writeTo(builder);
    try builder.append(", ");
    try d.writeTo(builder);
    try builder.append(", ");
    try a.writeTo(builder);
    try builder.append(", ");
    try b.writeTo(builder);
    try builder.append(", ");
    try c.writeTo(builder);
    try builder.append(") : (");
    try builder.append(atom_ty.text);
    try builder.append(", ");
    try builder.append(d_ty.text);
    try builder.append(", ");
    try builder.append(a_ty.text);
    try builder.append(", ");
    try builder.append(b_ty.text);
    try builder.append(", ");
    try builder.append(c_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn writeCoordType(out: anytype, coord: []const u8) Error!void {
    try validateCutePayload(coord);
    try out.append("!cute.coord<");
    try out.appendQuotedString(coord);
    try out.append(">");
}

pub fn writeMemRefType(
    out: anytype,
    elem: []const u8,
    memory_space: []const u8,
    alignment: usize,
    layout_text: []const u8,
) Error!void {
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

pub fn writeUniversalCopyAtomType(
    out: anytype,
    elem: []const u8,
    bits: usize,
) Error!void {
    try validateElementType(elem);
    if (bits == 0) return Error.InvalidAtomType;
    try out.append("!cute_nvgpu.atom.universal_copy<");
    try out.append(elem);
    try out.append(", ");
    try out.appendUnsigned(bits);
    try out.append(" b>");
}

pub fn writeUniversalFmaAtomType(
    out: anytype,
    elem: []const u8,
    m: usize,
    n: usize,
    k: usize,
) Error!void {
    try validateElementType(elem);
    if (m == 0 or n == 0 or k == 0) return Error.InvalidAtomType;
    try out.append("!cute_nvgpu.atom.universal_fma<");
    try out.appendUnsigned(m);
    try out.append("x");
    try out.appendUnsigned(n);
    try out.append("x");
    try out.appendUnsigned(k);
    try out.append(", (");
    try out.append(elem);
    try out.append(", ");
    try out.append(elem);
    try out.append(") -> ");
    try out.append(elem);
    try out.append(" >");
}

pub fn writeTiledCopyType(
    out: anytype,
    copy_atom_type: []const u8,
    layout_copy_tv: []const u8,
    tiler_mn: []const u8,
) Error!void {
    if (std.mem.indexOf(u8, copy_atom_type, "!cute_nvgpu.atom.universal_copy") == null)
        return Error.InvalidAtomType;
    try validateCutePayload(layout_copy_tv);
    try validateTilePayload(tiler_mn);
    try out.append("!cute.tiled_copy<");
    try out.append(copy_atom_type);
    try out.append(", layout_copy_tv = <");
    try out.appendQuotedString(layout_copy_tv);
    try out.append(">, tiler_mn = <");
    try out.appendQuotedString(tiler_mn);
    try out.append(">>");
}

pub fn validateCutePayload(payload: []const u8) Error!void {
    if (payload.len == 0) return Error.InvalidCuteTypePayload;
    for (payload) |c| switch (c) {
        '0'...'9', 'a'...'z', 'A'...'Z', '(', ')', ',', ':', '@', '_', '-', '+', '*', '/', ' ', '.', '[', ']', ';' => {},
        else => return Error.InvalidCuteTypePayload,
    };
    if (std.mem.indexOfScalar(u8, payload, '"') != null)
        return Error.InvalidCuteTypePayload;
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

fn validateTilePayload(payload: []const u8) Error!void {
    if (payload.len == 0) return Error.InvalidCuteTypePayload;
    for (payload) |c| switch (c) {
        '0'...'9', 'a'...'z', 'A'...'Z', '(', ')', '[', ']', ',', ':', ';', '@', '_', '-', '+', '*', '/', ' ', '.' => {},
        else => return Error.InvalidCuteTypePayload,
    };
    if (std.mem.indexOfScalar(u8, payload, '"') != null)
        return Error.InvalidCuteTypePayload;
}

pub fn emitTensorScalarModule(out: anytype) Error!void {
    try out.append(tensor_scalar_fixture);
}

pub fn emitTensorVectorModule(out: anytype) Error!void {
    try out.append(tensor_vector_fixture);
}

pub fn emitCopyAtomModule(out: anytype) Error!void {
    try out.append(copy_atom_fixture);
}

pub fn emitTiledCopyModule(out: anytype) Error!void {
    try out.append(tiled_copy_fixture);
}

pub fn emitMmaAtomModule(out: anytype) Error!void {
    try out.append(mma_atom_fixture);
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

pub fn writeStatus(out: anytype) Error!void {
    try out.append("CUTLASS emission helpers generate tensor/copy/MMA MLIR spelling using parser-aligned Cute syntax. ");
    try out.append("Fixtures use !cute.memref, !cute_nvgpu.atom.universal_copy, !cute.tiled_copy, and !cute_nvgpu.atom.universal_fma; ");
    try out.append("integration audit later replaced the remaining default example/golden placeholders with parser-aligned forms.\n");
}

test "cutlass_emit fixtures are structural and placeholder-free" {
    for (fixtures) |fixture| {
        try fixture.validate();
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "module") != null);
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "!cute.tensor") == null);
    }
}

test "cutlass_emit writes parser-aligned atom and tiled-copy types" {
    var out: mlir.TextBuffer(512) = .{};
    try writeUniversalCopyAtomType(&out, "f32", 32);
    try std.testing.expectEqualStrings(universal_copy_f32_32b, out.slice());
    out.clear();
    try writeUniversalFmaAtomType(&out, "f32", 1, 1, 1);
    try std.testing.expectEqualStrings(universal_fma_f32_1x1x1, out.slice());
    out.clear();
    try writeTiledCopyType(&out, universal_copy_f32_32b, "(1,1):(1,1)", "[1:0;1:0]");
    try std.testing.expectEqualStrings(tiled_copy_f32_1x1, out.slice());
}

test "cutlass_emit tensor spelling uses dot ops and parenthesized operands" {
    try std.testing.expect(std.mem.indexOf(u8, tensor_vector_fixture, "cute.memref.load_vec(%arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, tensor_vector_fixture, "cute.memref.store_vec(%1, %arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, tensor_vector_fixture, "cute.memref_load_vec") == null);
}

test "cutlass_emit copy and MMA fixtures use real CUTLASS atom types" {
    try std.testing.expect(std.mem.indexOf(u8, copy_atom_fixture, universal_copy_f32_32b) != null);
    try std.testing.expect(std.mem.indexOf(u8, tiled_copy_fixture, tiled_copy_f32_1x1) != null);
    try std.testing.expect(std.mem.indexOf(u8, mma_atom_fixture, universal_fma_f32_1x1x1) != null);
    try std.testing.expect(std.mem.indexOf(u8, mma_atom_fixture, "cute.mma_atom_call") != null);
}

test "cutlass_emit status points to parser-aligned follow-up" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "integration audit") != null);
}
