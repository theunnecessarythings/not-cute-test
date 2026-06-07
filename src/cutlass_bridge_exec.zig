const std = @import("std");
const mlir_harness = @import("mlir_harness.zig");
const cutlass_bridge = @import("cutlass_bridge.zig");

pub const Error = cutlass_bridge.Error || error{
    BridgeFailed,
};

pub const BridgeStatus = enum {
    skipped,
    passed,
    failed,
};

pub const OwnedBridgeResult = struct {
    term: std.process.Child.Term,
    stdout: []u8,
    stderr: []u8,

    pub fn deinit(self: OwnedBridgeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }

    pub fn success(self: OwnedBridgeResult) bool {
        return switch (self.term) {
            .exited => |code| code == 0,
            else => false,
        };
    }
};

pub const BridgeRunResult = struct {
    status: BridgeStatus,
    term: ?std.process.Child.Term = null,
};

pub fn runInvocation(
    allocator: std.mem.Allocator,
    inv: mlir_harness.Invocation,
    max_output_bytes: usize,
) !OwnedBridgeResult {
    _ = max_output_bytes;
    const io = std.Io.Threaded.global_single_threaded.io();
    var child = try std.process.spawn(io, .{
        .argv = inv.args(),
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    const term = try child.wait(io);
    return .{
        .term = term,
        .stdout = try allocator.dupe(u8, ""),
        .stderr = try allocator.dupe(u8, ""),
    };
}

pub fn discoverMaybe(
    allocator: std.mem.Allocator,
    config: cutlass_bridge.PythonBridgeConfig,
    enabled: bool,
) !BridgeRunResult {
    if (!enabled) return .{ .status = .skipped };
    const inv = try cutlass_bridge.discoveryInvocation(config);
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    return .{
        .status = if (result.success()) .passed else .failed,
        .term = result.term,
    };
}

pub fn verifyMaybe(
    allocator: std.mem.Allocator,
    config: cutlass_bridge.PythonBridgeConfig,
    input_mlir: []const u8,
    pipeline: []const u8,
    enabled: bool,
) !BridgeRunResult {
    if (!enabled) return .{ .status = .skipped };
    const inv = try cutlass_bridge.verifyInvocation(config, input_mlir, pipeline);
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    return .{
        .status = if (result.success()) .passed else .failed,
        .term = result.term,
    };
}

test "cutlass_bridge exec can skip external bridge" {
    const res = try discoverMaybe(std.testing.allocator, .{}, false);
    try std.testing.expect(res.status == .skipped);
}
