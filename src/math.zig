const std = @import("std");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const tensor = @import("tensor.zig");

pub const Error = mlir.Error || tensor.Error || error{InvalidMathOp};

pub const MathOp = enum {
    acos,
    asin,
    atan,
    atan2,
    absf,
    copysign,
    cos,
    erf,
    exp,
    exp2,
    floor,
    log,
    log2,
    log10,
    rsqrt,
    sin,
    sqrt,
    tan,
    tanh,

    pub fn mlirName(self: MathOp) []const u8 {
        return switch (self) {
            .acos => "math.acos",
            .asin => "math.asin",
            .atan => "math.atan",
            .atan2 => "math.atan2",
            .absf => "math.absf",
            .copysign => "math.copysign",
            .cos => "math.cos",
            .erf => "math.erf",
            .exp => "math.exp",
            .exp2 => "math.exp2",
            .floor => "math.floor",
            .log => "math.log",
            .log2 => "math.log2",
            .log10 => "math.log10",
            .rsqrt => "math.rsqrt",
            .sin => "math.sin",
            .sqrt => "math.sqrt",
            .tan => "math.tan",
            .tanh => "math.tanh",
        };
    }
};

pub fn unary(
    builder: anytype,
    op: MathOp,
    value: mlir.Operand,
    ty: mlir.Type,
) Error!mlir.Value {
    if (op == .atan2 or op == .copysign) return Error.InvalidMathOp;
    return builder.genericOp(op.mlirName(), &.{value}, &.{}, &.{ty}, &.{ty});
}

pub fn binary(
    builder: anytype,
    op: MathOp,
    lhs: mlir.Operand,
    rhs: mlir.Operand,
    ty: mlir.Type,
) Error!mlir.Value {
    if (op != .atan2 and op != .copysign) return Error.InvalidMathOp;
    return builder.genericOp(op.mlirName(), &.{ lhs, rhs }, &.{}, &.{ ty, ty }, &.{ty});
}

pub fn tensorUnary(
    builder: anytype,
    op: MathOp,
    input: tensor.TensorSsa,
) Error!tensor.TensorSsa {
    var vt: mlir.TextBuffer(128) = .{};
    try input.vectorType(&vt);
    const ty = mlir.Type.raw(vt.slice());
    const v = try unary(builder, op, .{ .value = input.value }, ty);
    return tensor.TensorSsa.init(v, input.shape_value, input.dtype);
}

pub fn tensorBinary(
    builder: anytype,
    op: MathOp,
    lhs: tensor.TensorSsa,
    rhs: tensor.TensorSsa,
) Error!tensor.TensorSsa {
    var vt: mlir.TextBuffer(128) = .{};
    try lhs.vectorType(&vt);
    const ty = mlir.Type.raw(vt.slice());
    const v = try binary(
        builder,
        op,
        .{ .value = lhs.value },
        .{ .value = rhs.value },
        ty,
    );
    return tensor.TensorSsa.init(v, lhs.shape_value, lhs.dtype);
}

pub fn acos(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .acos, value, ty);
}
pub fn asin(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .asin, value, ty);
}
pub fn atan(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .atan, value, ty);
}
pub fn absf(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .absf, value, ty);
}
pub fn cos(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .cos, value, ty);
}
pub fn erf(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .erf, value, ty);
}
pub fn exp(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .exp, value, ty);
}
pub fn exp2(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .exp2, value, ty);
}
pub fn floor(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .floor, value, ty);
}
pub fn log(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .log, value, ty);
}
pub fn log2(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .log2, value, ty);
}
pub fn log10(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .log10, value, ty);
}
pub fn rsqrt(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .rsqrt, value, ty);
}
pub fn sin(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .sin, value, ty);
}
pub fn sqrt(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .sqrt, value, ty);
}
pub fn tan(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .tan, value, ty);
}
pub fn tanh(builder: anytype, value: mlir.Operand, ty: mlir.Type) Error!mlir.Value {
    return unary(builder, .tanh, value, ty);
}
pub fn atan2(
    builder: anytype,
    lhs: mlir.Operand,
    rhs: mlir.Operand,
    ty: mlir.Type,
) Error!mlir.Value {
    return binary(builder, .atan2, lhs, rhs, ty);
}
pub fn copysign(
    builder: anytype,
    lhs: mlir.Operand,
    rhs: mlir.Operand,
    ty: mlir.Type,
) Error!mlir.Value {
    return binary(builder, .copysign, lhs, rhs, ty);
}

test "math: emits scalar and tensor math ops" {
    var b: mlir.Builder(2048) = .{};
    const x = try b.constantF(1.0, mlir.Type.f(32));
    _ = try sqrt(&b, .{ .value = x }, mlir.Type.f(32));
    _ = try atan2(&b, .{ .value = x }, .{ .value = x }, mlir.Type.f(32));
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "math.sqrt") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "math.atan2") != null);
}
