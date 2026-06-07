const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir.zig");
const atom = @import("atom.zig");
const tensor = @import("tensor.zig");

pub const Error = layout.Error || mlir.Error || atom.Error || tensor.Error || error{
    InvalidCopyConfiguration,
    InvalidGemmConfiguration,
};

pub const Predicate = enum { always, if_in_bounds, explicit_mask };
pub const CopyKind = enum { scalar, vectorized, cp_async, tma, tiled };
pub const PrefetchKind = enum { generic, tma_descriptor, cp_async_cache };

pub const CopyPlan = struct {
    kind: CopyKind = .scalar,
    predicate: Predicate = .always,
    bytes_per_transaction: usize = 0,
    atom_desc: ?atom.CopyAtom = null,

    pub fn init(
        kind: CopyKind,
        predicate: Predicate,
        bytes_per_transaction: usize,
    ) Error!CopyPlan {
        if (bytes_per_transaction == 0 and kind != .tma) return Error.InvalidCopyConfiguration;
        return .{
            .kind = kind,
            .predicate = predicate,
            .bytes_per_transaction = bytes_per_transaction,
        };
    }

    pub fn withAtom(self: CopyPlan, copy_atom: atom.CopyAtom) CopyPlan {
        var out = self;
        out.atom_desc = copy_atom;
        return out;
    }
};

pub const GemmPlan = struct {
    mma: atom.TiledMma,
    element_a: typing.Numeric,
    element_b: typing.Numeric,
    element_c: typing.Numeric,
    accumulator: typing.Numeric,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,

    pub fn init(
        mma: atom.TiledMma,
        element_a: typing.Numeric,
        element_b: typing.Numeric,
        element_c: typing.Numeric,
        accumulator: typing.Numeric,
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
    ) Error!GemmPlan {
        if (tile_m == 0 or tile_n == 0 or tile_k == 0) return Error.InvalidGemmConfiguration;
        return .{
            .mma = mma,
            .element_a = element_a,
            .element_b = element_b,
            .element_c = element_c,
            .accumulator = accumulator,
            .tile_m = tile_m,
            .tile_n = tile_n,
            .tile_k = tile_k,
        };
    }
};

pub fn basicCopy(
    builder: anytype,
    src: mlir.Operand,
    dst: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.copy",
        .operands = &.{ src, dst },
        .operand_types = &.{ ty, ty },
        .result_types = &.{},
    });
}

pub fn basicCopyIf(
    builder: anytype,
    pred: mlir.Operand,
    src: mlir.Operand,
    dst: mlir.Operand,
    pred_type: mlir.Type,
    ty: mlir.Type,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.copy_if",
        .operands = &.{ pred, src, dst },
        .operand_types = &.{ pred_type, ty, ty },
        .result_types = &.{},
    });
}

pub fn autovecCopy(
    builder: anytype,
    src: mlir.Operand,
    dst: mlir.Operand,
    ty: mlir.Type,
    vector_bytes: usize,
) Error!void {
    if (vector_bytes == 0) return Error.InvalidCopyConfiguration;
    try builder.operationNoResult(.{
        .name = "cute.autovec_copy",
        .operands = &.{ src, dst },
        .attrs = &.{.{ .key = "vector_bytes", .value = comptimeIntAttr(vector_bytes) }},
        .operand_types = &.{ ty, ty },
        .result_types = &.{},
    });
}

pub fn copy(
    builder: anytype,
    plan: CopyPlan,
    src: mlir.Operand,
    dst: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    const name = switch (plan.kind) {
        .scalar => "cute.copy",
        .vectorized => "cute.autovec_copy",
        .cp_async => "cute.cp_async_copy",
        .tma => "cute.tma_copy",
        .tiled => "cute.tiled_copy",
    };
    try builder.operationNoResult(.{
        .name = name,
        .operands = &.{ src, dst },
        .attrs = &.{
            .{ .key = "predicate", .value = predicateName(plan.predicate) },
            .{
                .key = "bytes_per_transaction",
                .value = comptimeIntAttr(plan.bytes_per_transaction),
            },
        },
        .operand_types = &.{ ty, ty },
        .result_types = &.{},
    });
}

pub fn prefetch(
    builder: anytype,
    kind: PrefetchKind,
    ptr: mlir.Operand,
    ty: mlir.Type,
) Error!void {
    const op = switch (kind) {
        .generic => "cute.prefetch",
        .tma_descriptor => "cute.tma_prefetch_descriptor",
        .cp_async_cache => "cute.cp_async_prefetch",
    };
    try builder.operationNoResult(.{
        .name = op,
        .operands = &.{ptr},
        .operand_types = &.{ty},
        .result_types = &.{},
    });
}

pub fn gemm(
    builder: anytype,
    plan: GemmPlan,
    a: mlir.Operand,
    b: mlir.Operand,
    c: mlir.Operand,
    tensor_type: mlir.Type,
) Error!mlir.Value {
    _ = plan;
    return builder.genericOp(
        "cute.gemm",
        &.{ a, b, c },
        &.{},
        &.{ tensor_type, tensor_type, tensor_type },
        &.{tensor_type},
    );
}

fn predicateName(pred: Predicate) []const u8 {
    return switch (pred) {
        .always => "always",
        .if_in_bounds => "if_in_bounds",
        .explicit_mask => "explicit_mask",
    };
}

fn comptimeIntAttr(value: usize) []const u8 {
    return switch (value) {
        0 => "0",
        1 => "1",
        2 => "2",
        4 => "4",
        8 => "8",
        16 => "16",
        32 => "32",
        64 => "64",
        128 => "128",
        else => "dynamic",
    };
}

test "algorithm: copy and prefetch emit source-shaped operations" {
    var b: mlir.Builder(2048) = .{};
    try copy(
        &b,
        try CopyPlan.init(.vectorized, .always, 16),
        .arg(0),
        .arg(1),
        mlir.Type.raw("!cute.memref<f32, gmem, align<16>, \"(1):(1)\">"),
    );
    try prefetch(&b, .cp_async_cache, .arg(0), mlir.Type.raw("!cute.ptr"));
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.autovec_copy") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.cp_async_prefetch") != null);
}
