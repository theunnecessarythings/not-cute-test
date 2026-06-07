const std = @import("std");
const runtime = @import("runtime.zig");
const runtime_plan = @import("runtime_plan.zig");

pub const Error = runtime.Error || runtime_plan.Error || error{
    OutOfMemory,
    CudaDriverUnavailable,
    CudaSymbolUnavailable,
    CudaCallFailed,
    InvalidCudaModule,
    InvalidCudaFunction,
    InvalidCudaStream,
    InvalidCudaArgument,
    InvalidCudaMemory,
    InvalidCString,
    FileReadFailed,
};

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUcontext = usize;
pub const CUmodule = usize;
pub const CUfunction = usize;
pub const CUstream = usize;
pub const CUevent = usize;
pub const CUdeviceptr = u64;

pub const success: CUresult = 0;

pub const DriverFunctionName = enum {
    cuInit,
    cuDriverGetVersion,
    cuDeviceGetCount,
    cuDeviceGet,
    cuDeviceGetName,
    cuDeviceComputeCapability,
    cuDevicePrimaryCtxRetain,
    cuDevicePrimaryCtxRelease,
    cuCtxCreate_v2,
    cuCtxDestroy_v2,
    cuCtxSetCurrent,
    cuCtxGetCurrent,
    cuModuleLoad,
    cuModuleLoadData,
    cuModuleUnload,
    cuModuleGetFunction,
    cuLaunchKernel,
    cuStreamCreate,
    cuStreamDestroy_v2,
    cuStreamSynchronize,
    cuMemAlloc_v2,
    cuMemFree_v2,
    cuMemcpyHtoD_v2,
    cuMemcpyDtoH_v2,
    cuMemcpyDtoD_v2,
    cuGetErrorString,
};

pub const Fn = struct {
    pub const cuInit = *const fn (c_uint) callconv(.c) CUresult;
    pub const cuDriverGetVersion = *const fn (*c_int) callconv(.c) CUresult;
    pub const cuDeviceGetCount = *const fn (*c_int) callconv(.c) CUresult;
    pub const cuDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
    pub const cuDeviceGetName = *const fn ([*c]u8, c_int, CUdevice) callconv(.c) CUresult;
    pub const cuDeviceComputeCapability = *const fn (*c_int, *c_int, CUdevice) callconv(.c) CUresult;
    pub const cuDevicePrimaryCtxRetain = *const fn (*CUcontext, CUdevice) callconv(.c) CUresult;
    pub const cuDevicePrimaryCtxRelease = *const fn (CUdevice) callconv(.c) CUresult;
    pub const cuCtxCreate_v2 = *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult;
    pub const cuCtxDestroy_v2 = *const fn (CUcontext) callconv(.c) CUresult;
    pub const cuCtxSetCurrent = *const fn (CUcontext) callconv(.c) CUresult;
    pub const cuCtxGetCurrent = *const fn (*CUcontext) callconv(.c) CUresult;
    pub const cuModuleLoad = *const fn (*CUmodule, [*:0]const u8) callconv(.c) CUresult;
    pub const cuModuleLoadData = *const fn (*CUmodule, *const anyopaque) callconv(.c) CUresult;
    pub const cuModuleUnload = *const fn (CUmodule) callconv(.c) CUresult;
    pub const cuModuleGetFunction = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult;
    pub const cuLaunchKernel = *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, [*c]?*anyopaque, ?*anyopaque) callconv(.c) CUresult;
    pub const cuStreamCreate = *const fn (*CUstream, c_uint) callconv(.c) CUresult;
    pub const cuStreamDestroy_v2 = *const fn (CUstream) callconv(.c) CUresult;
    pub const cuStreamSynchronize = *const fn (CUstream) callconv(.c) CUresult;
    pub const cuMemAlloc_v2 = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
    pub const cuMemFree_v2 = *const fn (CUdeviceptr) callconv(.c) CUresult;
    pub const cuMemcpyHtoD_v2 = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
    pub const cuMemcpyDtoH_v2 = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
    pub const cuMemcpyDtoD_v2 = *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult;
    pub const cuGetErrorString = *const fn (CUresult, *[*:0]const u8) callconv(.c) CUresult;
};

pub const DriverSymbols = struct {
    cuInit: Fn.cuInit,
    cuDriverGetVersion: Fn.cuDriverGetVersion,
    cuDeviceGetCount: Fn.cuDeviceGetCount,
    cuDeviceGet: Fn.cuDeviceGet,
    cuDeviceGetName: Fn.cuDeviceGetName,
    cuDeviceComputeCapability: Fn.cuDeviceComputeCapability,
    cuDevicePrimaryCtxRetain: Fn.cuDevicePrimaryCtxRetain,
    cuDevicePrimaryCtxRelease: Fn.cuDevicePrimaryCtxRelease,
    cuCtxCreate_v2: Fn.cuCtxCreate_v2,
    cuCtxDestroy_v2: Fn.cuCtxDestroy_v2,
    cuCtxSetCurrent: Fn.cuCtxSetCurrent,
    cuCtxGetCurrent: Fn.cuCtxGetCurrent,
    cuModuleLoad: Fn.cuModuleLoad,
    cuModuleLoadData: Fn.cuModuleLoadData,
    cuModuleUnload: Fn.cuModuleUnload,
    cuModuleGetFunction: Fn.cuModuleGetFunction,
    cuLaunchKernel: Fn.cuLaunchKernel,
    cuStreamCreate: Fn.cuStreamCreate,
    cuStreamDestroy_v2: Fn.cuStreamDestroy_v2,
    cuStreamSynchronize: Fn.cuStreamSynchronize,
    cuMemAlloc_v2: Fn.cuMemAlloc_v2,
    cuMemFree_v2: Fn.cuMemFree_v2,
    cuMemcpyHtoD_v2: Fn.cuMemcpyHtoD_v2,
    cuMemcpyDtoH_v2: Fn.cuMemcpyDtoH_v2,
    cuMemcpyDtoD_v2: Fn.cuMemcpyDtoD_v2,
    cuGetErrorString: Fn.cuGetErrorString,

    pub fn load(lib: *std.DynLib) Error!DriverSymbols {
        return .{
            .cuInit = try lookup(lib, Fn.cuInit, "cuInit"),
            .cuDriverGetVersion = try lookup(
                lib,
                Fn.cuDriverGetVersion,
                "cuDriverGetVersion",
            ),
            .cuDeviceGetCount = try lookup(
                lib,
                Fn.cuDeviceGetCount,
                "cuDeviceGetCount",
            ),
            .cuDeviceGet = try lookup(lib, Fn.cuDeviceGet, "cuDeviceGet"),
            .cuDeviceGetName = try lookup(lib, Fn.cuDeviceGetName, "cuDeviceGetName"),
            .cuDeviceComputeCapability = try lookup(
                lib,
                Fn.cuDeviceComputeCapability,
                "cuDeviceComputeCapability",
            ),
            .cuDevicePrimaryCtxRetain = try lookup(
                lib,
                Fn.cuDevicePrimaryCtxRetain,
                "cuDevicePrimaryCtxRetain",
            ),
            .cuDevicePrimaryCtxRelease = try lookup(
                lib,
                Fn.cuDevicePrimaryCtxRelease,
                "cuDevicePrimaryCtxRelease",
            ),
            .cuCtxCreate_v2 = try lookup(lib, Fn.cuCtxCreate_v2, "cuCtxCreate_v2"),
            .cuCtxDestroy_v2 = try lookup(lib, Fn.cuCtxDestroy_v2, "cuCtxDestroy_v2"),
            .cuCtxSetCurrent = try lookup(lib, Fn.cuCtxSetCurrent, "cuCtxSetCurrent"),
            .cuCtxGetCurrent = try lookup(lib, Fn.cuCtxGetCurrent, "cuCtxGetCurrent"),
            .cuModuleLoad = try lookup(lib, Fn.cuModuleLoad, "cuModuleLoad"),
            .cuModuleLoadData = try lookup(
                lib,
                Fn.cuModuleLoadData,
                "cuModuleLoadData",
            ),
            .cuModuleUnload = try lookup(lib, Fn.cuModuleUnload, "cuModuleUnload"),
            .cuModuleGetFunction = try lookup(
                lib,
                Fn.cuModuleGetFunction,
                "cuModuleGetFunction",
            ),
            .cuLaunchKernel = try lookup(lib, Fn.cuLaunchKernel, "cuLaunchKernel"),
            .cuStreamCreate = try lookup(lib, Fn.cuStreamCreate, "cuStreamCreate"),
            .cuStreamDestroy_v2 = try lookup(
                lib,
                Fn.cuStreamDestroy_v2,
                "cuStreamDestroy_v2",
            ),
            .cuStreamSynchronize = try lookup(
                lib,
                Fn.cuStreamSynchronize,
                "cuStreamSynchronize",
            ),
            .cuMemAlloc_v2 = try lookup(lib, Fn.cuMemAlloc_v2, "cuMemAlloc_v2"),
            .cuMemFree_v2 = try lookup(lib, Fn.cuMemFree_v2, "cuMemFree_v2"),
            .cuMemcpyHtoD_v2 = try lookup(lib, Fn.cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2"),
            .cuMemcpyDtoH_v2 = try lookup(lib, Fn.cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2"),
            .cuMemcpyDtoD_v2 = try lookup(lib, Fn.cuMemcpyDtoD_v2, "cuMemcpyDtoD_v2"),
            .cuGetErrorString = try lookup(
                lib,
                Fn.cuGetErrorString,
                "cuGetErrorString",
            ),
        };
    }

    fn lookup(lib: *std.DynLib, comptime T: type, name: [:0]const u8) Error!T {
        return lib.lookup(T, name) orelse Error.CudaSymbolUnavailable;
    }

    pub fn check(self: DriverSymbols, code: CUresult) Error!void {
        if (code == success) return;
        _ = self;
        return Error.CudaCallFailed;
    }

    pub fn errorString(self: DriverSymbols, code: CUresult) []const u8 {
        var msg: [*:0]const u8 = "unknown CUDA driver error";
        if (self.cuGetErrorString(code, &msg) == success) return std.mem.span(msg);
        return "unknown CUDA driver error";
    }
};

pub const LoadedDriver = struct {
    lib: std.DynLib,
    symbols: DriverSymbols,

    pub fn open(path: []const u8) Error!LoadedDriver {
        var lib = std.DynLib.open(path) catch return Error.CudaDriverUnavailable;
        errdefer lib.close();
        const symbols = try DriverSymbols.load(&lib);
        return .{ .lib = lib, .symbols = symbols };
    }

    pub fn close(self: *LoadedDriver) void {
        self.lib.close();
    }
};

pub const DeviceInfo = struct {
    device: CUdevice,
    name: [256]u8 = [_]u8{0} ** 256,
    name_len: usize = 0,
    compute_major: c_int = 0,
    compute_minor: c_int = 0,

    pub fn nameSlice(self: *const DeviceInfo) []const u8 {
        return self.name[0..self.name_len];
    }
};

pub const Context = struct {
    handle: CUcontext = 0,
    device: CUdevice = 0,
    retained_primary: bool = true,
};

pub const Module = struct {
    handle: CUmodule = 0,
    path: []const u8 = "",
    from_memory: bool = false,
};

pub const Function = struct {
    handle: CUfunction = 0,
    symbol: []const u8 = "",
};

pub const Stream = struct {
    handle: CUstream = 0,
    owns: bool = false,
};

pub const DeviceMemory = struct {
    ptr: CUdeviceptr = 0,
    bytes: usize = 0,
};

pub fn init(driver: DriverSymbols) Error!void {
    try driver.check(driver.cuInit(0));
}

pub fn driverVersion(driver: DriverSymbols) Error!c_int {
    var v: c_int = 0;
    try driver.check(driver.cuDriverGetVersion(&v));
    return v;
}

pub fn deviceCount(driver: DriverSymbols) Error!c_int {
    var count: c_int = 0;
    try driver.check(driver.cuDeviceGetCount(&count));
    return count;
}

pub fn getDevice(driver: DriverSymbols, ordinal: c_int) Error!CUdevice {
    var dev: CUdevice = 0;
    try driver.check(driver.cuDeviceGet(&dev, ordinal));
    return dev;
}

pub fn getDeviceInfo(driver: DriverSymbols, ordinal: c_int) Error!DeviceInfo {
    const dev = try getDevice(driver, ordinal);
    var info: DeviceInfo = .{ .device = dev };
    try driver.check(driver.cuDeviceGetName(&info.name, @intCast(info.name.len), dev));
    info.name_len = std.mem.indexOfScalar(u8, &info.name, 0) orelse info.name.len;
    try driver.check(driver.cuDeviceComputeCapability(
        &info.compute_major,
        &info.compute_minor,
        dev,
    ));
    return info;
}

pub fn retainPrimaryContext(driver: DriverSymbols, device: CUdevice) Error!Context {
    var ctx: CUcontext = 0;
    try driver.check(driver.cuDevicePrimaryCtxRetain(&ctx, device));
    try driver.check(driver.cuCtxSetCurrent(ctx));
    return .{ .handle = ctx, .device = device, .retained_primary = true };
}

pub fn createContext(
    driver: DriverSymbols,
    device: CUdevice,
    flags: c_uint,
) Error!Context {
    var ctx: CUcontext = 0;
    try driver.check(driver.cuCtxCreate_v2(&ctx, flags, device));
    return .{ .handle = ctx, .device = device, .retained_primary = false };
}

pub fn destroyContext(driver: DriverSymbols, ctx: Context) Error!void {
    if (ctx.handle == 0) return;
    if (ctx.retained_primary)
        try driver.check(driver.cuDevicePrimaryCtxRelease(ctx.device))
    else
        try driver.check(driver.cuCtxDestroy_v2(ctx.handle));
}

pub fn loadModuleFromPath(
    allocator: std.mem.Allocator,
    driver: DriverSymbols,
    path: []const u8,
) Error!Module {
    if (path.len == 0) return Error.InvalidCudaModule;
    const zpath = allocator.dupeZ(u8, path) catch return Error.OutOfMemory;
    defer allocator.free(zpath);
    var module: CUmodule = 0;
    try driver.check(driver.cuModuleLoad(&module, zpath));
    return .{ .handle = module, .path = path, .from_memory = false };
}

pub fn loadModuleFromBytes(driver: DriverSymbols, bytes: []const u8) Error!Module {
    if (bytes.len == 0) return Error.InvalidCudaModule;
    var module: CUmodule = 0;
    try driver.check(driver.cuModuleLoadData(&module, bytes.ptr));
    return .{ .handle = module, .path = "<memory>", .from_memory = true };
}

pub fn unloadModule(driver: DriverSymbols, module: Module) Error!void {
    if (module.handle != 0) try driver.check(driver.cuModuleUnload(module.handle));
}

pub fn getFunction(
    allocator: std.mem.Allocator,
    driver: DriverSymbols,
    module: Module,
    symbol: []const u8,
) Error!Function {
    if (module.handle == 0) return Error.InvalidCudaModule;
    if (symbol.len == 0) return Error.InvalidCudaFunction;
    const zsym = allocator.dupeZ(u8, symbol) catch return Error.OutOfMemory;
    defer allocator.free(zsym);
    var function: CUfunction = 0;
    try driver.check(driver.cuModuleGetFunction(&function, module.handle, zsym));
    return .{ .handle = function, .symbol = symbol };
}

pub fn createStream(driver: DriverSymbols, flags: c_uint) Error!Stream {
    var s: CUstream = 0;
    try driver.check(driver.cuStreamCreate(&s, flags));
    return .{ .handle = s, .owns = true };
}

pub fn destroyStream(driver: DriverSymbols, stream: Stream) Error!void {
    if (stream.owns and stream.handle != 0)
        try driver.check(driver.cuStreamDestroy_v2(stream.handle));
}

pub fn synchronizeStream(driver: DriverSymbols, stream: Stream) Error!void {
    try driver.check(driver.cuStreamSynchronize(stream.handle));
}

pub fn allocateDevice(driver: DriverSymbols, bytes: usize) Error!DeviceMemory {
    if (bytes == 0) return Error.InvalidCudaMemory;
    var ptr: CUdeviceptr = 0;
    try driver.check(driver.cuMemAlloc_v2(&ptr, bytes));
    return .{ .ptr = ptr, .bytes = bytes };
}

pub fn freeDevice(driver: DriverSymbols, mem: DeviceMemory) Error!void {
    if (mem.ptr != 0) try driver.check(driver.cuMemFree_v2(mem.ptr));
}

pub fn memcpyHtoD(
    driver: DriverSymbols,
    dst: DeviceMemory,
    src: []const u8,
) Error!void {
    if (src.len > dst.bytes) return Error.InvalidCudaMemory;
    if (src.len == 0) return;
    try driver.check(driver.cuMemcpyHtoD_v2(dst.ptr, src.ptr, src.len));
}

pub fn memcpyDtoH(driver: DriverSymbols, dst: []u8, src: DeviceMemory) Error!void {
    if (dst.len > src.bytes) return Error.InvalidCudaMemory;
    if (dst.len == 0) return;
    try driver.check(driver.cuMemcpyDtoH_v2(dst.ptr, src.ptr, dst.len));
}

pub fn memcpyDtoD(
    driver: DriverSymbols,
    dst: DeviceMemory,
    src: DeviceMemory,
    bytes: usize,
) Error!void {
    if (bytes > dst.bytes or bytes > src.bytes) return Error.InvalidCudaMemory;
    if (bytes == 0) return;
    try driver.check(driver.cuMemcpyDtoD_v2(dst.ptr, src.ptr, bytes));
}

pub const KernelArgument = struct {
    name: []const u8,
    ptr: ?*anyopaque,

    pub fn init(name: []const u8, ptr: ?*anyopaque) Error!KernelArgument {
        if (name.len == 0 or ptr == null) return Error.InvalidCudaArgument;
        return .{ .name = name, .ptr = ptr };
    }
};

pub const LaunchArguments = struct {
    slots: [64]?*anyopaque = undefined,
    len: usize = 0,

    pub fn append(self: *LaunchArguments, arg: KernelArgument) Error!void {
        if (self.len >= self.slots.len) return Error.TooManyPackedArguments;
        self.slots[self.len] = arg.ptr;
        self.len += 1;
    }

    pub fn kernelParams(self: *LaunchArguments) [*c]?*anyopaque {
        if (self.len == 0) return null;
        return self.slots[0..self.len].ptr;
    }
};

pub fn launchKernel(
    driver: DriverSymbols,
    function: Function,
    config: runtime.LaunchConfig,
    args: *LaunchArguments,
    stream: Stream,
) Error!void {
    if (function.handle == 0) return Error.InvalidCudaFunction;
    try driver.check(driver.cuLaunchKernel(
        function.handle,
        config.grid.x,
        config.grid.y,
        config.grid.z,
        config.block.x,
        config.block.y,
        config.block.z,
        @intCast(config.dynamic_smem_bytes),
        stream.handle,
        args.kernelParams(),
        null,
    ));
}

pub const ExecutionRequest = struct {
    driver_library: []const u8 = "libcuda.so.1",
    device_ordinal: c_int = 0,
    module_path: []const u8,
    kernel_symbol: []const u8,
    launch: runtime.LaunchConfig,
    retain_primary_context: bool = true,
    create_stream: bool = true,

    pub fn validate(self: ExecutionRequest) Error!void {
        if (self.driver_library.len == 0) return Error.CudaDriverUnavailable;
        if (self.module_path.len == 0) return Error.InvalidCudaModule;
        if (self.kernel_symbol.len == 0) return Error.InvalidCudaFunction;
        if (self.launch.grid.volume() == 0 or self.launch.block.volume() == 0)
            return Error.InvalidLaunchShape;
    }
};

pub const DryRunReport = struct {
    request: ExecutionRequest,
    argument_slots: usize,

    pub fn writeJson(self: DryRunReport, out: anytype) Error!void {
        try self.request.validate();
        try out.append("{\n");
        try out.append("  \"driver_library\": ");
        try out.appendQuotedString(self.request.driver_library);
        try out.append(",\n  \"module_path\": ");
        try out.appendQuotedString(self.request.module_path);
        try out.append(",\n  \"kernel_symbol\": ");
        try out.appendQuotedString(self.request.kernel_symbol);
        try out.append(",\n  \"device_ordinal\": ");
        try out.appendSigned(self.request.device_ordinal);
        try out.append(",\n  \"grid\": [");
        try out.appendUnsigned(self.request.launch.grid.x);
        try out.append(", ");
        try out.appendUnsigned(self.request.launch.grid.y);
        try out.append(", ");
        try out.appendUnsigned(self.request.launch.grid.z);
        try out.append("],\n  \"block\": [");
        try out.appendUnsigned(self.request.launch.block.x);
        try out.append(", ");
        try out.appendUnsigned(self.request.launch.block.y);
        try out.append(", ");
        try out.appendUnsigned(self.request.launch.block.z);
        try out.append("],\n  \"dynamic_smem_bytes\": ");
        try out.appendUnsigned(self.request.launch.dynamic_smem_bytes);
        try out.append(",\n  \"argument_slots\": ");
        try out.appendUnsigned(self.argument_slots);
        try out.append("\n}\n");
    }
};

pub fn dryRun(request: ExecutionRequest, argument_slots: usize) Error!DryRunReport {
    try request.validate();
    return .{ .request = request, .argument_slots = argument_slots };
}

pub const ManagedSession = struct {
    driver: LoadedDriver,
    context: Context,
    stream: Stream,

    pub fn open(path: []const u8, device_ordinal: c_int) Error!ManagedSession {
        var loaded = try LoadedDriver.open(path);
        errdefer loaded.close();
        try init(loaded.symbols);
        const dev = try getDevice(loaded.symbols, device_ordinal);
        const ctx = try retainPrimaryContext(loaded.symbols, dev);
        errdefer destroyContext(loaded.symbols, ctx) catch {};
        const stream = try createStream(loaded.symbols, 0);
        errdefer destroyStream(loaded.symbols, stream) catch {};
        return .{ .driver = loaded, .context = ctx, .stream = stream };
    }

    pub fn close(self: *ManagedSession) void {
        destroyStream(self.driver.symbols, self.stream) catch {};
        destroyContext(self.driver.symbols, self.context) catch {};
        self.driver.close();
    }

    pub fn loadModule(
        self: *ManagedSession,
        allocator: std.mem.Allocator,
        path: []const u8,
    ) Error!Module {
        return loadModuleFromPath(allocator, self.driver.symbols, path);
    }

    pub fn loadFunction(
        self: *ManagedSession,
        allocator: std.mem.Allocator,
        module: Module,
        symbol: []const u8,
    ) Error!Function {
        return getFunction(allocator, self.driver.symbols, module, symbol);
    }

    pub fn launch(
        self: *ManagedSession,
        function: Function,
        config: runtime.LaunchConfig,
        args: *LaunchArguments,
    ) Error!void {
        try launchKernel(self.driver.symbols, function, config, args, self.stream);
    }
};

pub fn writeCDriverDeclarations(out: anytype) Error!void {
    try out.append("typedef int CUresult;\n");
    try out.append("typedef int CUdevice;\n");
    try out.append("typedef void* CUcontext;\n");
    try out.append("typedef void* CUmodule;\n");
    try out.append("typedef void* CUfunction;\n");
    try out.append("typedef void* CUstream;\n");
    try out.append("CUresult cuInit(unsigned int Flags);\n");
    try out.append("CUresult cuModuleLoad(CUmodule *module, const char *fname);\n");
    try out.append("CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);\n");
    try out.append("CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);\n");
}

test "cuda_driver: dry-run execution request emits launch JSON" {
    const cfg = try runtime.LaunchConfig.init(
        try runtime.Dim3.init(2, 1, 1),
        try runtime.Dim3.init(128, 1, 1),
        4096,
        runtime.Stream.default(),
    );
    const report = try dryRun(
        .{ .module_path = "kernel.cubin", .kernel_symbol = "kernel", .launch = cfg },
        3,
    );
    var out: @import("mlir_text.zig").TextBuffer(1024) = .{};
    try report.writeJson(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "kernel.cubin") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "\"argument_slots\": 3") != null);
}

test "cuda_driver: argument pack preserves kernel parameter pointers" {
    var x: u64 = 7;
    var args: LaunchArguments = .{};
    try std.testing.expect(args.kernelParams() == null);
    try args.append(try KernelArgument.init("x", @ptrCast(&x)));
    try std.testing.expectEqual(@as(usize, 1), args.len);
    try std.testing.expect(args.kernelParams() != null);
    try std.testing.expect(args.slots[0] != null);
}

test "cuda_driver: C declarations include launch ABI" {
    var out: @import("mlir_text.zig").TextBuffer(1024) = .{};
    try writeCDriverDeclarations(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cuLaunchKernel") != null);
}
