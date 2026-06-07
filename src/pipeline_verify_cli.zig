const std = @import("std");
const mlir = @import("mlir_text.zig");
const pipeline_verify = @import("pipeline_verify.zig");

fn parseShard(text: []const u8) !pipeline_verify.VerifyShard {
    if (std.mem.eql(u8, text, "layout")) return .layout;
    if (std.mem.eql(u8, text, "tensor")) return .tensor;
    if (std.mem.eql(u8, text, "copy")) return .copy;
    if (std.mem.eql(u8, text, "mma")) return .mma;
    if (std.mem.eql(u8, text, "tiled")) return .tiled;
    if (std.mem.eql(u8, text, "negative")) return .negative;
    if (std.mem.eql(u8, text, "all")) return .all;
    return error.InvalidArgs;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse "manifest";
    var out: mlir.TextBuffer(16384) = .{};
    if (std.mem.eql(u8, cmd, "manifest")) {
        try pipeline_verify.writeVerifierManifest(&out);
    } else if (std.mem.eql(u8, cmd, "script")) {
        const shard = try parseShard(args.next() orelse "all");
        try pipeline_verify.writeShardScript(.{}, shard, &out);
    } else {
        return error.InvalidArgs;
    }
    std.debug.print("{s}\n", .{out.slice()});
}
