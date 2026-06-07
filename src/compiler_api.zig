const std = @import("std");
const runtime_plan = @import("runtime_plan.zig");

pub const CompilationError = error{ CompilationFailed, InvalidCompileOption };
pub const OptLevel = enum { O0, O1, O2, O3 };
pub const GPUArch = enum { sm70, sm75, sm80, sm89, sm90, sm100, sm103, sm120 };
pub const CompileOption = struct { key: []const u8, value: []const u8 = "" };
pub const BooleanCompileOption = struct { key: []const u8, enabled: bool = true };
pub const StringCompileOption = struct { key: []const u8, value: []const u8 };
pub const BooleanBasedFileDumpOption = BooleanCompileOption;
pub const EmptyCompileOption = struct { key: []const u8 };
pub const PtxasOptions = struct { options: []const []const u8 = &.{} };
pub const EnableAssertions = BooleanCompileOption;
pub const GenerateLineInfo = BooleanCompileOption;
pub const KeepCUBIN = BooleanCompileOption;
pub const KeepPTX = BooleanCompileOption;
pub const LinkLibraries = struct { paths: []const []const u8 = &.{} };
pub const EnableTVMFFI = BooleanCompileOption;
pub const DumpDir = StringCompileOption;
pub const CompileCallable = *const fn () void;
pub const PostCompileHookContext = struct { artifact_path: []const u8 = "", cubin_hash: u64 = 0 };
pub const Compiler = struct { opt: OptLevel = .O2, arch: GPUArch = .sm90 };

pub fn makeCompilePlan(
    function_name: []const u8,
    pipeline: []const u8,
) runtime_plan.CompilePlan {
    return .{ .function_name = function_name, .pipeline = pipeline };
}
pub fn option(key: []const u8, value: []const u8) CompileOption {
    return .{ .key = key, .value = value };
}
pub fn boolOption(key: []const u8, enabled: bool) BooleanCompileOption {
    return .{ .key = key, .enabled = enabled };
}
pub fn gpuArchName(arch: GPUArch) []const u8 {
    return switch (arch) {
        .sm70 => "sm_70",
        .sm75 => "sm_75",
        .sm80 => "sm_80",
        .sm89 => "sm_89",
        .sm90 => "sm_90",
        .sm100 => "sm_100",
        .sm103 => "sm_103",
        .sm120 => "sm_120",
    };
}

test "compiler_api: source named options and arch spellings" {
    const c: Compiler = .{ .arch = .sm100 };
    try std.testing.expectEqualStrings("sm_100", gpuArchName(c.arch));
    const o = boolOption("keep-ptx", true);
    try std.testing.expect(o.enabled);
}
