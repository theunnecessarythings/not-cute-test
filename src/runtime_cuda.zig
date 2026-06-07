const std = @import("std");
const runtime = @import("runtime.zig");

pub const Error = runtime.Error || error{ CudaUnavailable, InvalidCudaResult };
pub const DevicePrimaryContext = struct { device: i32 = 0, handle: usize = 0 };
pub const CudaModule = struct { handle: usize = 0, bytes_hash: u64 = 0 };
pub const CudaFunction = struct { module: CudaModule = .{}, symbol: []const u8 = "" };
pub const CudaLibrary = struct { handle: usize = 0, path: []const u8 = "" };
pub const KernelArgument = struct { name: []const u8 = "", bytes: []const u8 = "" };
pub const KernelLaunch = struct { function_name: []const u8, launch: runtime.LaunchConfig, argument_count: usize };

pub fn getComputeCapabilityMajorMinor() struct { major: u16, minor: u16 } {
    return .{ .major = 0, .minor = 0 };
}
pub fn getDeviceInfo(device: i32) runtime.DeviceInfo {
    return .{ .device_id = device };
}
pub fn checkCudaErrors(code: i32) Error!void {
    if (code != 0) return Error.InvalidCudaResult;
}
pub fn getCurrentDevice() i32 {
    return 0;
}
pub fn getDevice(device: i32) i32 {
    return device;
}
pub fn initializeCudaContext(device: i32) DevicePrimaryContext {
    return .{ .device = device, .handle = @intCast(device + 1) };
}
pub fn devicePrimaryContextRetain(device: i32) DevicePrimaryContext {
    return initializeCudaContext(device);
}
pub fn devicePrimaryContextRelease(_: DevicePrimaryContext) void {}
pub fn loadCubinModule(bytes: []const u8) CudaModule {
    return .{ .bytes_hash = std.hash.Wyhash.hash(0, bytes) };
}
pub fn unloadCubinModule(_: CudaModule) void {}
pub fn loadCubinModuleData(bytes: []const u8) CudaModule {
    return loadCubinModule(bytes);
}
pub fn getKernelFunction(module: CudaModule, symbol: []const u8) CudaFunction {
    return .{ .module = module, .symbol = symbol };
}
pub fn loadLibrary(path: []const u8) CudaLibrary {
    return .{ .path = path, .handle = std.hash.Wyhash.hash(0, path) };
}
pub fn unloadLibrary(_: CudaLibrary) void {}
pub fn loadLibraryData(bytes: []const u8) CudaLibrary {
    return .{ .handle = std.hash.Wyhash.hash(1, bytes), .path = "<memory>" };
}
pub fn getLibraryKernel(lib: CudaLibrary, symbol: []const u8) CudaFunction {
    return .{ .module = .{ .handle = lib.handle }, .symbol = symbol };
}
pub fn getFunctionFromKernel(kernel: CudaFunction) CudaFunction {
    return kernel;
}
pub fn loadLibraryFromFile(path: []const u8) CudaLibrary {
    return loadLibrary(path);
}
pub fn launchKernel(
    function: CudaFunction,
    launch: runtime.LaunchConfig,
    args: []const KernelArgument,
) KernelLaunch {
    return .{
        .function_name = function.symbol,
        .launch = launch,
        .argument_count = args.len,
    };
}
pub fn streamSync(_: runtime.Stream) void {}
pub fn streamCreate() runtime.Stream {
    return runtime.Stream{ .handle = 1 };
}
pub fn streamDestroy(_: runtime.Stream) void {}
pub fn contextDestroy(_: DevicePrimaryContext) void {}
pub fn allocate(
    bytes: usize,
    dtype: @import("typing.zig").Numeric,
    memspace: @import("typing.zig").AddressSpace,
) Error!runtime.Pointer {
    return runtime.Pointer.init(bytes, dtype, memspace, 1);
}
pub fn free(_: runtime.Pointer) void {}
pub fn memcpyHtod(_: runtime.Pointer, _: []const u8) void {}
pub fn memcpyDtoh(_: []u8, _: runtime.Pointer) void {}
pub fn memcpyDtod(_: runtime.Pointer, _: runtime.Pointer, _: usize) void {}
pub fn eventCreate() usize {
    return 1;
}
pub fn eventDestroy(_: usize) void {}
pub fn eventRecord(_: usize, _: runtime.Stream) void {}
pub fn eventElapsedTimeMs(_: usize, _: usize) f32 {
    return 0.0;
}

test "runtime_cuda: descriptor functions build launch metadata without CUDA driver" {
    const m = loadCubinModule("abc");
    const f = getKernelFunction(m, "kernel");
    const launch = runtime.LaunchConfig.init(
        .{ .x = 1 },
        .{ .x = 1 },
        0,
        runtime.Stream.default(),
    ) catch unreachable;
    const k = launchKernel(f, launch, &.{});
    try std.testing.expectEqualStrings("kernel", k.function_name);
}
