const std = @import("std");
const mlir_harness = @import("mlir_harness.zig");

pub const Error = mlir_harness.Error;

pub const OwnedToolResult = struct {
    term: std.process.Child.Term,
    stdout: []u8,
    stderr: []u8,

    pub fn deinit(self: OwnedToolResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }

    pub fn success(self: OwnedToolResult) bool {
        return termSuccess(self.term);
    }
};

pub const VerificationStatus = enum {
    skipped,
    passed,
    failed,
};

pub const VerificationResult = struct {
    status: VerificationStatus,
    term: ?std.process.Child.Term = null,
};

/// Execute an external verifier/lowering command.  Diagnostics are inherited by
/// default so CI logs show the native tool output; callers still receive the
/// exit status and can layer file-based stdout/stderr capture when desired.
pub fn runInvocation(allocator: std.mem.Allocator, inv: mlir_harness.Invocation, max_output_bytes: usize) !OwnedToolResult {
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

pub fn verifyMlirMaybe(allocator: std.mem.Allocator, config: mlir_harness.ToolConfig, input_path: []const u8) !VerificationResult {
    if (!config.shouldRunExternal()) return .{ .status = .skipped };
    const inv = try mlir_harness.cuteOptVerifyInvocation(config, input_path);
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    return .{ .status = if (result.success()) .passed else .failed, .term = result.term };
}

pub fn runPipelineMaybe(allocator: std.mem.Allocator, config: mlir_harness.ToolConfig, input_path: []const u8, output_path: []const u8, pass_pipeline: []const u8) !VerificationResult {
    if (!config.shouldRunExternal()) return .{ .status = .skipped };
    const inv = try mlir_harness.mlirOptPipelineInvocation(config, input_path, output_path, pass_pipeline);
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    return .{ .status = if (result.success()) .passed else .failed, .term = result.term };
}

pub fn expectToolFailureContains(allocator: std.mem.Allocator, config: mlir_harness.ToolConfig, inv: mlir_harness.Invocation, diagnostic: []const u8) !VerificationResult {
    if (!config.shouldRunExternal()) return .{ .status = .skipped };
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    if (result.success()) return Error.NegativeTestUnexpectedSuccess;
    if (std.mem.indexOf(u8, result.stderr, diagnostic) == null and std.mem.indexOf(u8, result.stdout, diagnostic) == null) return Error.MissingExpectedDiagnostic;
    return .{ .status = .passed, .term = result.term };
}

pub fn writeCaseFile(allocator: std.mem.Allocator, out_dir: []const u8, case_name: []const u8, text: []const u8) !void {
    const filename = try std.fmt.allocPrint(allocator, "{s}.mlir", .{case_name});
    defer allocator.free(filename);
    const path = try std.fs.path.join(allocator, &.{ out_dir, filename });
    defer allocator.free(path);
    const io = std.Io.Threaded.global_single_threaded.io();
    var file = try std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, text);
}

pub fn writeAllGeneratedCases(allocator: std.mem.Allocator, out_dir: []const u8) !void {
    var b: @import("mlir_text.zig").Builder(32768) = .{};

    b.reset();
    try mlir_harness.emitLayoutCase(&b);
    _ = try b.finish();
    try writeCaseFile(allocator, out_dir, "layout_case", b.slice());

    b.reset();
    try mlir_harness.emitTensorCase(&b);
    _ = try b.finish();
    try writeCaseFile(allocator, out_dir, "tensor_case", b.slice());

    b.reset();
    try mlir_harness.emitCopyCase(&b);
    _ = try b.finish();
    try writeCaseFile(allocator, out_dir, "copy_case", b.slice());

    b.reset();
    try mlir_harness.emitMmaCase(&b);
    _ = try b.finish();
    try writeCaseFile(allocator, out_dir, "mma_case", b.slice());

    b.reset();
    try mlir_harness.emitNegativeCase(&b);
    try writeCaseFile(allocator, out_dir, "negative_case", b.slice());
}

fn termSuccess(term: std.process.Child.Term) bool {
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}
