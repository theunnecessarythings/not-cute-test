const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    var builder: cute.mlir.Builder(8192) = .{};
    const tensor_layout = try cute.layout.Layout.makeCompact(
        cute.layout.Tree.fromComptime(.{4}),
    );
    const src_meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.mlir.Value.arg(0) },
        tensor_layout,
        cute.typing.Float32,
        .gmem,
    );
    const dst_meta = try cute.tensor.TensorMeta.init(
        .{ .mlir_value = cute.mlir.Value.arg(1) },
        tensor_layout,
        cute.typing.Float32,
        .generic,
    );
    var src_type: cute.mlir.TextBuffer(512) = .{};
    var dst_type: cute.mlir.TextBuffer(512) = .{};
    try src_meta.tensorTypeText(&src_type);
    try dst_meta.tensorTypeText(&dst_type);

    const copy_atom = try cute.nvgpu.copyG2R(cute.typing.Float32, .{
        .num_bits_per_copy = 128,
    });
    const plan = try cute.copy_mma.validateCopy(copy_atom, src_meta, dst_meta, null);

    try builder.beginModule();
    try builder.beginFunc(
        "copy_vector",
        &.{
            cute.mlir.Type.raw(src_type.slice()),
            cute.mlir.Type.raw(dst_type.slice()),
        },
        null,
    );
    const src = cute.tensor.TensorValue.init(
        src_meta,
        cute.mlir.Value.arg(0),
        src_type.slice(),
    );
    const dst = cute.tensor.TensorValue.init(
        dst_meta,
        cute.mlir.Value.arg(1),
        dst_type.slice(),
    );
    _ = try cute.copy_mma.lowerCopyAtom(&builder, copy_atom, src, dst, null);
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();

    std.debug.print("// elements={} vector_bits={}\n{s}", .{
        plan.element_count,
        plan.vector_bits,
        try builder.finish(),
    });
}
