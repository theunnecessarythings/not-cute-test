const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    var builder: cute.mlir.Builder(16384) = .{};
    const atom = try cute.nvgpu.universalMma(cute.typing.Float32);
    const a_layout = cute.layout.makeCompactLayout(.{ 1, 1 });
    const b_layout = cute.layout.makeCompactLayout(.{ 1, 1 });
    const c_layout = cute.layout.makeCompactLayout(.{ 1, 1 });

    const a_meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.mlir.Value.arg(0) },
        a_layout,
        cute.typing.Float32,
        .generic,
    );
    const b_meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.mlir.Value.arg(1) },
        b_layout,
        cute.typing.Float32,
        .generic,
    );
    const c_meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.mlir.Value.arg(2) },
        c_layout,
        cute.typing.Float32,
        .generic,
    );
    const d_meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.mlir.Value.arg(3) },
        c_layout,
        cute.typing.Float32,
        .generic,
    );

    var a_type: cute.mlir.TextBuffer(512) = .{};
    var b_type: cute.mlir.TextBuffer(512) = .{};
    var c_type: cute.mlir.TextBuffer(512) = .{};
    try a_meta.tensorTypeText(&a_type);
    try b_meta.tensorTypeText(&b_type);
    try c_meta.tensorTypeText(&c_type);

    try builder.beginModule();
    try builder.beginFunc(
        "scalar_mma",
        &.{
            cute.mlir.Type.raw(a_type.slice()),
            cute.mlir.Type.raw(b_type.slice()),
            cute.mlir.Type.raw(c_type.slice()),
            cute.mlir.Type.raw(c_type.slice()),
        },
        null,
    );
    const a = cute.tensor.TensorValue.init(
        a_meta,
        cute.mlir.Value.arg(0),
        a_type.slice(),
    );
    const b = cute.tensor.TensorValue.init(
        b_meta,
        cute.mlir.Value.arg(1),
        b_type.slice(),
    );
    const c = cute.tensor.TensorValue.init(
        c_meta,
        cute.mlir.Value.arg(2),
        c_type.slice(),
    );
    const d = cute.tensor.TensorValue.init(
        d_meta,
        cute.mlir.Value.arg(3),
        c_type.slice(),
    );
    const plan = try cute.copy_mma.lowerMmaAtom(&builder, atom, d, a, b, c);
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();

    std.debug.print("// mma={}x{}x{}\n{s}", .{
        plan.m,
        plan.n,
        plan.k,
        try builder.finish(),
    });
}
