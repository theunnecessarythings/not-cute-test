const std = @import("std");
const runtime = @import("runtime.zig");

pub const Error = runtime.Error || error{ CudaUnavailable, InvalidCudaResult };
pub const DevicePrimaryContext = struct { device: i32 = 0, handle: usize = 0 };
pub const CudaModule = struct { handle: usize = 0, bytes_hash: u64 = 0 };
pub const CudaFunction = struct { module: CudaModule = .{}, symbol: []const u8 = "" };
pub const CudaLibrary = struct { handle: usize = 0, path: []const u8 = "" };
pub const KernelArgument = struct { name: []const u8 = "", bytes: []const u8 = "" };
pub const KernelLaunch = struct { function_name: []const u8, launch: runtime.LaunchConfig, argument_count: usize };

pub fn get_compute_capability_major_minor() struct { major: u16, minor: u16 } {
    return .{ .major = 0, .minor = 0 };
}
pub fn get_device_info(device: i32) runtime.DeviceInfo {
    return .{ .device_id = device };
}
pub fn checkCudaErrors(code: i32) Error!void {
    if (code != 0) return Error.InvalidCudaResult;
}
pub fn get_current_device() i32 {
    return 0;
}
pub fn get_device(device: i32) i32 {
    return device;
}
pub fn initialize_cuda_context(device: i32) DevicePrimaryContext {
    return .{ .device = device, .handle = @intCast(device + 1) };
}
pub fn device_primary_context_retain(device: i32) DevicePrimaryContext {
    return initialize_cuda_context(device);
}
pub fn device_primary_context_release(_: DevicePrimaryContext) void {}
pub fn load_cubin_module(bytes: []const u8) CudaModule {
    return .{ .bytes_hash = std.hash.Wyhash.hash(0, bytes) };
}
pub fn unload_cubin_module(_: CudaModule) void {}
pub fn load_cubin_module_data(bytes: []const u8) CudaModule {
    return load_cubin_module(bytes);
}
pub fn get_kernel_function(module: CudaModule, symbol: []const u8) CudaFunction {
    return .{ .module = module, .symbol = symbol };
}
pub fn load_library(path: []const u8) CudaLibrary {
    return .{ .path = path, .handle = std.hash.Wyhash.hash(0, path) };
}
pub fn unload_library(_: CudaLibrary) void {}
pub fn load_library_data(bytes: []const u8) CudaLibrary {
    return .{ .handle = std.hash.Wyhash.hash(1, bytes), .path = "<memory>" };
}
pub fn get_library_kernel(lib: CudaLibrary, symbol: []const u8) CudaFunction {
    return .{ .module = .{ .handle = lib.handle }, .symbol = symbol };
}
pub fn get_function_from_kernel(kernel: CudaFunction) CudaFunction {
    return kernel;
}
pub fn load_library_from_file(path: []const u8) CudaLibrary {
    return load_library(path);
}
pub fn launch_kernel(function: CudaFunction, launch: runtime.LaunchConfig, args: []const KernelArgument) KernelLaunch {
    return .{ .function_name = function.symbol, .launch = launch, .argument_count = args.len };
}
pub fn stream_sync(_: runtime.Stream) void {}
pub fn stream_create() runtime.Stream {
    return runtime.Stream{ .handle = 1 };
}
pub fn stream_destroy(_: runtime.Stream) void {}
pub fn context_destroy(_: DevicePrimaryContext) void {}
pub fn allocate(bytes: usize, dtype: @import("typing.zig").Numeric, memspace: @import("typing.zig").AddressSpace) Error!runtime.Pointer {
    return runtime.Pointer.init(bytes, dtype, memspace, 1);
}
pub fn free(_: runtime.Pointer) void {}
pub fn memcpy_htod(_: runtime.Pointer, _: []const u8) void {}
pub fn memcpy_dtoh(_: []u8, _: runtime.Pointer) void {}
pub fn memcpy_dtod(_: runtime.Pointer, _: runtime.Pointer, _: usize) void {}
pub fn event_create() usize {
    return 1;
}
pub fn event_destroy(_: usize) void {}
pub fn event_record(_: usize, _: runtime.Stream) void {}
pub fn event_elapsed_time_ms(_: usize, _: usize) f32 {
    return 0.0;
}

test "runtime_cuda: descriptor functions build launch metadata without CUDA driver" {
    const m = load_cubin_module("abc");
    const f = get_kernel_function(m, "kernel");
    const launch = runtime.LaunchConfig.init(.{ .x = 1 }, .{ .x = 1 }, 0, runtime.Stream.default()) catch unreachable;
    const k = launch_kernel(f, launch, &.{});
    try std.testing.expectEqualStrings("kernel", k.function_name);
}
