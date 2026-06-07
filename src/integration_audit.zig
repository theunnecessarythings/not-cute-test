const std = @import("std");
const mlir = @import("mlir_text.zig");
const mlir_harness = @import("mlir_harness.zig");
const examples_api = @import("examples_api.zig");
const cutlass_routed = @import("cutlass_routed.zig");
const tiled_emit = @import("tiled_emit.zig");

pub const Error = error{BannedPlaceholderFound} || mlir.Error || examples_api.Error || cutlass_routed.Error || tiled_emit.Error;

pub const banned_default_patterns = [_][]const u8{
    "!cute.tensor",
    "cute.memref_load",
    "cute.memref_store",
    "cute.copy_tensor",
    "cute.tiled_copy_",
    "cute.tiled_mma_",
    "cute.mma_make_fragment",
    "cute.mma_ssa",
};

pub fn assertNoDefaultPlaceholders(text: []const u8) Error!void {
    for (banned_default_patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return Error.BannedPlaceholderFound;
    }
}

pub fn writeStatus(out: anytype) Error!void {
    try out.append("integration audit is a corrective integration pass: default examples and MLIR harness generated golden cases now use CUTLASS parser-aligned Cute MLIR, not the old !cute.tensor placeholder dialect. ");
    try out.append("The negative fixture still intentionally contains !cute.tensor to prove the bridge rejects the fake type. This is not a full production port; it removes the known placeholder-default lie and leaves remaining source-parity work explicit.\n");
}
test "integration_audit mlir_harness default goldens are placeholder-free" {
    try assertNoDefaultPlaceholders(mlir_harness.integration_audit_layout_case_mlir);
    try assertNoDefaultPlaceholders(mlir_harness.integration_audit_tensor_case_mlir);
    try assertNoDefaultPlaceholders(mlir_harness.integration_audit_copy_case_mlir);
    try assertNoDefaultPlaceholders(mlir_harness.integration_audit_mma_case_mlir);
}

test "integration_audit examples_api public examples are placeholder-free" {
    for (examples_api.all_examples) |kind| {
        try assertNoDefaultPlaceholders(examples_api.exampleText(kind));
    }
}

test "integration_audit routed and full-tiled fixtures remain placeholder-free" {
    for (cutlass_routed.routed_fixtures) |fixture| try assertNoDefaultPlaceholders(fixture.mlir_text);
    for (tiled_emit.full_tiled_fixtures) |fixture| try assertNoDefaultPlaceholders(fixture.mlir_text);
}

test "integration_audit status is explicit about non-completion" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "not a full production port") != null);
}
