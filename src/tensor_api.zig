const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const tensor = @import("tensor.zig");
const tensor_ssa = @import("tensor_ssa.zig");
const runtime = @import("runtime.zig");

pub const Error = tensor_ssa.Error || tensor.Error || runtime.Error || error{InvalidTensorConstruction};
pub const Tensor = tensor.Tensor;
pub const TensorSSA = tensor_ssa.SsaTensor;
pub const TensorSsa = tensor_ssa.SsaTensor;
pub const TensorMeta = tensor_ssa.TensorMeta;
pub const TensorValue = tensor_ssa.TensorValue;
pub const SsaValue = tensor_ssa.SsaValue;
pub const Engine = tensor.Engine;

pub fn make_tensor(meta: TensorMeta, value: mlir.Value) Error!TensorValue {
    return TensorValue.initFromMeta(meta, value);
}
pub fn make_identity_tensor(builder: anytype, shape: layout.Tree) Error!TensorSSA {
    return TensorSSA.empty(builder, shape, typing.Int32);
}
pub fn make_rmem_tensor(builder: anytype, shape: layout.Tree, dtype: typing.Numeric) Error!TensorSSA {
    return TensorSSA.empty(builder, shape, dtype);
}
pub fn make_fragment(builder: anytype, shape: layout.Tree, dtype: typing.Numeric) Error!TensorSSA {
    return TensorSSA.empty(builder, shape, dtype);
}
pub fn make_rmem_tensor_like(builder: anytype, source: TensorSSA, dtype: ?typing.Numeric) Error!TensorSSA {
    return TensorSSA.emptyLike(builder, source, dtype);
}
pub fn make_fragment_like(builder: anytype, source: TensorSSA, dtype: ?typing.Numeric) Error!TensorSSA {
    return TensorSSA.emptyLike(builder, source, dtype);
}
pub fn recast_tensor(source: Tensor, dtype: typing.Numeric) Error!Tensor {
    return tensor.recastTensor(source, dtype);
}
pub fn domain_offset(source: Tensor, coord: layout.Tree) Error!Tensor {
    return tensor.domainOffset(source, coord);
}
pub fn full_like(builder: anytype, source: TensorSSA, scalar: mlir.Operand, dtype: ?typing.Numeric) Error!TensorSSA {
    return TensorSSA.fullLike(builder, source, scalar, dtype);
}
pub fn empty_like(builder: anytype, source: TensorSSA, dtype: ?typing.Numeric) Error!TensorSSA {
    return TensorSSA.emptyLike(builder, source, dtype);
}
pub fn ones_like(builder: anytype, source: TensorSSA) Error!TensorSSA {
    return TensorSSA.ones(builder, source.shape_value, source.dtype);
}
pub fn zeros_like(builder: anytype, source: TensorSSA) Error!TensorSSA {
    return TensorSSA.zeros(builder, source.shape_value, source.dtype);
}
pub fn any_(source: TensorSSA, builder: anytype) Error!SsaValue {
    return source.reduceAll(builder, .any);
}
pub fn all_(source: TensorSSA, builder: anytype) Error!SsaValue {
    return source.reduceAll(builder, .all);
}
pub fn print_tensor(builder: anytype, source: TensorValue) Error!void {
    try builder.operationNoResult(.{ .name = "cute.print_tensor", .operands = &.{.{ .value = source.value }}, .operand_types = &.{source.type_()}, .result_types = &.{} });
}

test "tensor_api: source-name constructors wrap tensor SSA" {
    var b: mlir.Builder(4096) = .{};
    const t = try make_rmem_tensor(&b, layout.Tree.fromComptime(.{ 2, 2 }), typing.Float32);
    const z = try zeros_like(&b, t);
    try std.testing.expectEqual(@as(usize, 2), z.shape_value.rank());
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.undef") != null);
}
