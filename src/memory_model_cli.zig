const std = @import("std");
const memory_model = @import("memory_model.zig");
const mlir = @import("mlir_text.zig");

pub fn main() !void {
    const alignment = try memory_model.AlignmentPolicy.init(16);
    const src = try memory_model.ExternalPointer.init(0x1000, 256, .host, alignment, null);
    const dst = try memory_model.ExternalPointer.init(0x2000, 256, .device, alignment, 0);
    const plan = try memory_model.transferPlan(try dst.descriptor(), try src.descriptor(), 256);
    var out: mlir.TextBuffer(512) = .{};
    try memory_model.writeOwnershipJson(plan, &out);
    std.debug.print("{s}", .{out.slice()});
}
