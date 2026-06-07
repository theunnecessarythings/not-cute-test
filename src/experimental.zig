const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");
const algorithm = @import("algorithm.zig");

pub const Error = layout.Error || mlir.Error || runtime.Error || algorithm.Error || error{ InvalidMemoryAllocation, InvalidTmaOperation };

pub const AllocationKind = enum { global, shared, tensor_memory, register };

pub const Allocation = struct {
    kind: AllocationKind,
    dtype: typing.Numeric,
    shape: layout.Tree,
    alignment: usize,

    pub fn init(
        kind: AllocationKind,
        dtype: typing.Numeric,
        shape: layout.Tree,
        alignment: usize,
    ) Error!Allocation {
        if (alignment == 0) return Error.InvalidMemoryAllocation;
        try shape.assertPositive();
        return .{
            .kind = kind,
            .dtype = dtype,
            .shape = shape,
            .alignment = alignment,
        };
    }

    pub fn bytes(self: Allocation) Error!usize {
        const n = try self.shape.product();
        const b = n * self.dtype.bytes();
        if (b > std.math.maxInt(usize)) return Error.Overflow;
        return @intCast(b);
    }

    pub fn emit(self: Allocation, builder: anytype) Error!mlir.Value {
        return builder.genericOp(
            "cute.experimental.allocate",
            &.{},
            &.{.{ .key = "kind", .value = @tagName(self.kind) }},
            &.{},
            &.{mlir.Type.raw("!cute.ptr")},
        );
    }
};

pub const TmaLoadKind = enum { normal, im2col, multicast, bulk };

pub const TmaDescriptor = struct {
    rank: usize,
    element: typing.Numeric,
    load_kind: TmaLoadKind = .normal,

    pub fn init(
        rank: usize,
        element: typing.Numeric,
        load_kind: TmaLoadKind,
    ) Error!TmaDescriptor {
        if (rank == 0 or rank > 5) return Error.InvalidTmaOperation;
        return .{ .rank = rank, .element = element, .load_kind = load_kind };
    }
};

pub fn tmaLoad(
    builder: anytype,
    desc: TmaDescriptor,
    src: mlir.Operand,
    dst: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.experimental.tma_load",
        .operands = &.{ src, dst },
        .attrs = &.{.{ .key = "kind", .value = @tagName(desc.load_kind) }},
        .operand_types = &.{ ty, ty },
        .result_types = &.{},
    });
}

pub fn tmaLoadMulticast(
    builder: anytype,
    desc: TmaDescriptor,
    src: mlir.Operand,
    dst: mlir.Operand,
    mask: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.experimental.tma_load_multicast",
        .operands = &.{ src, dst, mask },
        .attrs = &.{
            .{ .key = "rank", .value = "dynamic" },
            .{ .key = "kind", .value = @tagName(desc.load_kind) },
        },
        .operand_types = &.{ ty, ty, mlir.Type.i(16) },
        .result_types = &.{},
    });
}

pub fn tmaStore(
    builder: anytype,
    desc: TmaDescriptor,
    src: mlir.Operand,
    dst: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.experimental.tma_store",
        .operands = &.{ src, dst },
        .attrs = &.{.{ .key = "kind", .value = @tagName(desc.load_kind) }},
        .operand_types = &.{ ty, ty },
        .result_types = &.{},
    });
}

pub fn memoryCopy(
    builder: anytype,
    src: mlir.Operand,
    dst: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    try algorithm.basicCopy(builder, src, dst, ty);
}

test "experimental: allocation and tma ops emit" {
    var b: mlir.Builder(2048) = .{};
    const alloc = try Allocation.init(
        .shared,
        typing.Float32,
        layout.Tree.fromComptime(.{ 8, 8 }),
        16,
    );
    _ = try alloc.emit(&b);
    const desc = try TmaDescriptor.init(2, typing.Float32, .normal);
    try tmaLoad(
        &b,
        desc,
        .arg(0),
        .arg(1),
        mlir.Type.raw("!cute.memref<f32, gmem, \"(1,1):(1,1)\">"),
    );
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "experimental.allocate") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "tma_load") != null);
}
