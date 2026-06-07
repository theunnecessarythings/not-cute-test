const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    var builder: cute.ir.IR.Module(8192) = .{};
    const shape = cute.layout.Tree.fromComptime(.{4});
    const tensor_layout = try cute.layout.Layout.makeCompact(shape);
    const meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.ir.Value.arg(0) },
        tensor_layout,
        cute.typing.Float32,
        .gmem,
    );
    var tensor_type: cute.ir.IR.Storage(512) = .{};
    try meta.tensorTypeText(&tensor_type);

    try builder.beginModule();
    try builder.beginFunc(
        "add_one",
        &.{cute.ir.Type.custom(tensor_type.contents())},
        null,
    );

    const tensor = cute.tensor.TensorValue.init(
        meta,
        cute.ir.Value.arg(0),
        tensor_type.contents(),
    );
    const values = try tensor.load(&builder, null, null);
    const one = try builder.constantF(1.0, cute.ir.Type.f(32));
    const ones = try cute.tensor.SsaTensor.full(
        &builder,
        shape,
        cute.typing.Float32,
        .{ .value = one },
    );
    const incremented = try values.add(&builder, ones);
    try tensor.store(&builder, incremented, null);

    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    std.debug.print("{s}", .{try builder.finish()});
}
