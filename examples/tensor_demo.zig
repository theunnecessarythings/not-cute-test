const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    var builder: cute.mlir_text.Builder(8192) = .{};
    const shape = cute.layout.Tree.fromComptime(.{4});
    const tensor_layout = try cute.layout.Layout.makeCompact(shape);
    const meta = try cute.tensor_ssa.TensorMeta.init(
        .{ .mlir_value = cute.mlir_text.Value.arg(0) },
        tensor_layout,
        cute.typing.Float32,
        .gmem,
    );
    var tensor_type: cute.mlir_text.TextBuffer(512) = .{};
    try meta.tensorTypeText(&tensor_type);

    try builder.beginModule();
    try builder.beginFunc(
        "add_one",
        &.{cute.mlir_text.Type.raw(tensor_type.slice())},
        null,
    );

    const tensor = cute.tensor_ssa.TensorValue.init(
        meta,
        cute.mlir_text.Value.arg(0),
        tensor_type.slice(),
    );
    const values = try tensor.load(&builder, null, null);
    const one = try builder.constantF(1.0, cute.mlir_text.Type.f(32));
    const ones = try cute.tensor_ssa.SsaTensor.full(
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
