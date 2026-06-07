pub const DevicePrimaryContext = enum(i32) { _ };
const export_ = @import("export.zig");
const jit = @import("jit.zig");
const layout = @import("layout.zig");
const mlir = @import("mlir.zig");
const std = @import("std");
const typing = @import("typing.zig");

pub const Error = layout.Error || error{
    InvalidExportName,
    TooManyArguments,
    BufferTooSmall,

    InvalidArtifactPath,
    InvalidCompileOption,
    InvalidDevice,
    InvalidDynamicMask,
    InvalidLaunchShape,
    InvalidLibraryPath,
    InvalidPackedArgument,
    InvalidRuntimeSymbol,
    InvalidSymbol,
    InvalidTensorDescriptor,
    InvalidToolPath,
    MisalignedPointer,
    TooManyPackedArguments,
};

pub const HostPointer = usize;
pub const DevicePointer = usize;

pub const Pointer = struct {
    address: usize,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace = .generic,
    assumed_align: usize = 1,

    pub fn init(
        address: usize,
        dtype: typing.Numeric,
        memspace: typing.AddressSpace,
        assumed_align: ?usize,
    ) Error!Pointer {
        const alignment = assumed_align orelse dtype.bytes();
        if (alignment == 0 or address % alignment != 0) return Error.MisalignedPointer;
        return .{
            .address = address,
            .dtype = dtype,
            .memspace = memspace,
            .assumed_align = alignment,
        };
    }

    pub fn nullptr(dtype: typing.Numeric, memspace: typing.AddressSpace) Pointer {
        return .{
            .address = 0,
            .dtype = dtype,
            .memspace = memspace,
            .assumed_align = dtype.bytes(),
        };
    }

    pub fn sizeInBytes(_: Pointer) usize {
        return @sizeOf(usize);
    }

    pub fn add(self: Pointer, offset_elements: isize) Pointer {
        const elem_bytes = self.dtype.bytes();
        const delta = offset_elements * @as(isize, @intCast(elem_bytes));
        const new_addr: usize = if (delta >= 0)
            self.address + @as(usize, @intCast(delta))
        else
            self.address - @as(usize, @intCast(-delta));
        const offset_bytes: usize = if (delta >= 0) @intCast(delta) else @intCast(-delta);
        const alignment = gcd(
            self.assumed_align,
            if (offset_bytes == 0) self.assumed_align else offset_bytes,
        );
        return .{
            .address = new_addr,
            .dtype = self.dtype,
            .memspace = self.memspace,
            .assumed_align = alignment,
        };
    }

    pub fn sub(self: Pointer, offset_elements: isize) Pointer {
        return self.add(-offset_elements);
    }

    pub fn mlirType(self: Pointer) mlir.Type {
        _ = self;
        return mlir.Type.raw("!cute.ptr");
    }

    pub fn writeMlirType(self: Pointer, out: anytype) Error!void {
        try out.append("!cute.ptr<");
        try out.append(self.dtype.mlir_type);
        try out.append(", ");
        try out.append(self.memspace.mlirName());
        try out.append(", align=");
        try out.appendUnsigned(self.assumed_align);
        try out.append(">");
    }

    pub fn writeRepr(self: Pointer, out: anytype) Error!void {
        try out.append("Ptr<0x");
        var tmp: [32]u8 = undefined;
        const printed = std.fmt.bufPrint(
            &tmp,
            "{x:0>16}",
            .{self.address},
        ) catch unreachable;
        try out.append(printed);
        try out.append("@");
        try out.append(self.memspace.mlirName());
        try out.append(">");
    }

    pub fn cacheKeyHash(self: Pointer) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&self.dtype.width));
        h.update(std.mem.asBytes(&self.assumed_align));
        const space: u32 = @intFromEnum(self.memspace);
        h.update(std.mem.asBytes(&space));
        return h.final();
    }
};

pub const DynamicMask = struct {
    bits: u128 = 0,

    pub fn mark(self: *DynamicMask, index: usize) Error!void {
        if (index >= 128) return Error.InvalidDynamicMask;
        self.bits |= (@as(u128, 1) << @intCast(index));
    }

    pub fn isMarked(self: DynamicMask, index: usize) bool {
        if (index >= 128) return false;
        return (self.bits & (@as(u128, 1) << @intCast(index))) != 0;
    }

    pub fn empty(self: DynamicMask) bool {
        return self.bits == 0;
    }
};

pub const TensorDescriptor = struct {
    pointer: Pointer,
    shape: layout.Tree,
    stride: layout.Tree,
    dynamic_shapes: DynamicMask = .{},
    dynamic_strides: DynamicMask = .{},
    use_32bit_stride: bool = false,

    pub fn init(
        pointer: Pointer,
        shape: layout.Tree,
        stride: layout.Tree,
    ) Error!TensorDescriptor {
        if (!shape.sameProfile(&stride)) return Error.InvalidTensorDescriptor;
        try shape.assertPositive();
        return .{ .pointer = pointer, .shape = shape, .stride = stride };
    }

    pub fn elementType(self: TensorDescriptor) typing.Numeric {
        return self.pointer.dtype;
    }

    pub fn memspace(self: TensorDescriptor) typing.AddressSpace {
        return self.pointer.memspace;
    }

    pub fn rank(self: TensorDescriptor) Error!usize {
        return self.shape.rank();
    }

    pub fn sizeInElements(self: TensorDescriptor) Error!layout.Unsigned {
        return self.shape.product();
    }

    pub fn sizeInBytes(self: TensorDescriptor) Error!usize {
        const elems = try self.sizeInElements();
        const bytes = elems * self.pointer.dtype.bytes();
        if (bytes > std.math.maxInt(usize)) return Error.Overflow;
        return @intCast(bytes);
    }

    pub fn markLayoutDynamic(self: *TensorDescriptor, leadingDim: ?usize) Error!void {
        const leaves = try self.stride.flattenLeaves();
        if (leadingDim) |dim| {
            if (dim >= leaves.len or leaves.at(dim) != 1)
                return Error.InvalidTensorDescriptor;
        }
        for (0..leaves.len) |i| {
            if (leadingDim == null or i != leadingDim.?) try self.dynamic_strides.mark(i);
        }
    }

    pub fn markCompactShapeDynamic(
        self: *TensorDescriptor,
        mode: usize,
        divisibility: u64,
    ) Error!void {
        if (divisibility == 0) return Error.InvalidDynamicMask;
        const leaves = try self.shape.flattenLeaves();
        if (mode >= leaves.len) return Error.InvalidDynamicMask;
        try self.dynamic_shapes.mark(mode);
        for (mode + 1..leaves.len) |i| try self.dynamic_strides.mark(i);
    }

    pub fn typedTensor(self: TensorDescriptor) Error!typing.TypedTensor {
        return typing.TypedTensor.init(
            self.pointer.dtype,
            self.shape,
            self.stride,
            self.pointer.memspace,
            self.pointer.assumed_align,
        );
    }

    pub fn writeMlirType(self: TensorDescriptor, out: anytype) Error!void {
        const tt = try self.typedTensor();
        try tt.writeMlirType(out);
    }

    pub fn isCompatible(self: TensorDescriptor, other: TensorDescriptor) bool {
        return self.pointer.dtype.name.ptr == other.pointer.dtype.name.ptr and
            self.pointer.memspace == other.pointer.memspace and
            self.shape.equals(&other.shape) and
            self.stride.equals(&other.stride);
    }
};

pub const FakeTensor = struct {
    dtype: typing.Numeric,
    shape: layout.Tree,
    stride: layout.Tree,
    memspace: typing.AddressSpace = .gmem,
    assumed_align: usize = 1,
    dynamic_shapes: DynamicMask = .{},
    dynamic_strides: DynamicMask = .{},

    pub fn compact(
        dtype: typing.Numeric,
        shape: layout.Tree,
        memspace: typing.AddressSpace,
        assumed_align: ?usize,
    ) Error!FakeTensor {
        const compact_layout = try layout.Layout.makeCompact(shape);
        return init(
            dtype,
            compact_layout.shape,
            compact_layout.stride,
            memspace,
            assumed_align,
        );
    }

    pub fn init(
        dtype: typing.Numeric,
        shape: layout.Tree,
        stride: layout.Tree,
        memspace: typing.AddressSpace,
        assumed_align: ?usize,
    ) Error!FakeTensor {
        if (!shape.sameProfile(&stride)) return Error.ProfileMismatch;
        try shape.assertPositive();
        return .{
            .dtype = dtype,
            .shape = shape,
            .stride = stride,
            .memspace = memspace,
            .assumed_align = assumed_align orelse dtype.bytes(),
        };
    }

    pub fn typedTensor(self: FakeTensor) Error!typing.TypedTensor {
        return typing.TypedTensor.init(
            self.dtype,
            self.shape,
            self.stride,
            self.memspace,
            self.assumed_align,
        );
    }

    pub fn sizeInBytes(self: FakeTensor) Error!usize {
        const elems = try self.shape.product();
        const bytes = elems * self.dtype.bytes();
        if (bytes > std.math.maxInt(usize)) return Error.Overflow;
        return @intCast(bytes);
    }
};

pub const Stream = struct {
    handle: usize = 0,

    pub fn default() Stream {
        return .{};
    }

    pub fn writeMlirType(_: Stream, out: anytype) Error!void {
        try out.append("!cuda.stream");
    }
};

pub const TensorAdapter = struct {
    tensor: TensorDescriptor,

    pub fn init(tensor: TensorDescriptor) TensorAdapter {
        return .{ .tensor = tensor };
    }

    pub fn cPointerCount(_: TensorAdapter) usize {
        return 1;
    }
};

pub const DeviceInfo = struct {
    device_id: i32 = 0,
    name: []const u8 = "unknown",
    compute_capability_major: u16 = 0,
    compute_capability_minor: u16 = 0,
    multiprocessor_count: u32 = 0,
    max_threads_per_block: u32 = 1024,
    shared_memory_per_block: usize = 0,
    total_global_memory: usize = 0,

    pub fn archString(self: DeviceInfo, out: anytype) Error!void {
        try out.append("sm_");
        try out.appendUnsigned(self.compute_capability_major);
        try out.appendUnsigned(self.compute_capability_minor);
    }
};

pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    pub fn init(x: u32, y: u32, z: u32) Error!Dim3 {
        if (x == 0 or y == 0 or z == 0) return Error.InvalidLaunchShape;
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn volume(self: Dim3) u64 {
        return @as(u64, self.x) * @as(u64, self.y) * @as(u64, self.z);
    }
};

pub const LaunchConfig = struct {
    grid: Dim3,
    block: Dim3,
    dynamic_smem_bytes: usize = 0,
    stream: Stream = .{},

    pub fn init(
        grid: Dim3,
        block: Dim3,
        dynamic_smem_bytes: usize,
        stream: Stream,
    ) Error!LaunchConfig {
        if (grid.volume() == 0 or block.volume() == 0) return Error.InvalidLaunchShape;
        return .{
            .grid = grid,
            .block = block,
            .dynamic_smem_bytes = dynamic_smem_bytes,
            .stream = stream,
        };
    }
};

pub const BinaryModule = struct {
    image_path: []const u8,
    format: enum { cubin, fatbin, ptx, object } = .cubin,

    pub fn init(
        image_path: []const u8,
        format: @FieldType(BinaryModule, "format"),
    ) Error!BinaryModule {
        if (image_path.len == 0) return Error.InvalidLibraryPath;
        return .{ .image_path = image_path, .format = format };
    }
};

pub const KernelFunction = struct {
    module: BinaryModule,
    name: []const u8,

    pub fn init(module: BinaryModule, name: []const u8) Error!KernelFunction {
        try mlir.validateSymbol(name);
        return .{ .module = module, .name = name };
    }
};

pub const LaunchRecord = struct {
    kernel: KernelFunction,
    config: LaunchConfig,
    argument_count: usize,
};

pub fn makePtr(
    address: usize,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace,
    assumed_align: ?usize,
) Error!Pointer {
    return Pointer.init(address, dtype, memspace, assumed_align);
}

pub fn makeFakeCompactTensor(
    dtype: typing.Numeric,
    shape: layout.Tree,
    memspace: typing.AddressSpace,
    assumed_align: ?usize,
) Error!FakeTensor {
    return FakeTensor.compact(dtype, shape, memspace, assumed_align);
}

pub fn makeFakeTensor(
    dtype: typing.Numeric,
    shape: layout.Tree,
    stride: layout.Tree,
    memspace: typing.AddressSpace,
    assumed_align: ?usize,
) Error!FakeTensor {
    return FakeTensor.init(dtype, shape, stride, memspace, assumed_align);
}

pub fn recordLaunch(
    kernel: KernelFunction,
    config: LaunchConfig,
    argument_count: usize,
) LaunchRecord {
    return .{ .kernel = kernel, .config = config, .argument_count = argument_count };
}

fn gcd(a_in: usize, b_in: usize) usize {
    var a = a_in;
    var b = b_in;
    while (b != 0) {
        const r = a % b;
        a = b;
        b = r;
    }
    return a;
}

test "runtime: pointer alignment and pointer arithmetic" {
    const p = try Pointer.init(0x1000, typing.Float32, .gmem, null);
    try std.testing.expectEqual(@as(usize, 4), p.assumed_align);
    const q = p.add(3);
    try std.testing.expectEqual(@as(usize, 0x100c), q.address);
    try std.testing.expectEqual(@as(usize, 4), q.assumed_align);
    try std.testing.expectError(
        Error.MisalignedPointer,
        Pointer.init(0x1002, typing.Float32, .gmem, 4),
    );
}

test "runtime: fake tensor and descriptor MLIR type" {
    const shape = layout.Tree.fromComptime(.{ 8, 4 });
    const fake = try makeFakeCompactTensor(typing.Float16, shape, .gmem, null);
    try std.testing.expectEqual(@as(usize, 64), try fake.sizeInBytes());
    var out: mlir.TextBuffer(256) = .{};
    const tt = try fake.typedTensor();
    try tt.writeMlirType(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.memref") != null);
}

test "runtime: launch record is validated" {
    const module = try BinaryModule.init("kernel.cubin", .cubin);
    const k = try KernelFunction.init(module, "kernel_main");
    const cfg = try LaunchConfig.init(
        try Dim3.init(2, 1, 1),
        try Dim3.init(128, 1, 1),
        4096,
        .{},
    );
    const rec = recordLaunch(k, cfg, 3);
    try std.testing.expectEqual(@as(usize, 3), rec.argument_count);
    try std.testing.expectEqual(@as(u64, 128), rec.config.block.volume());
}

pub const CudaModule = struct { handle: usize = 0, bytes_hash: u64 = 0 };
pub const CudaFunction = struct { module: CudaModule = .{}, symbol: []const u8 = "" };
pub const CudaLibrary = struct { handle: usize = 0, path: []const u8 = "" };
pub const KernelArgument = struct { name: []const u8 = "", bytes: []const u8 = "" };
pub const KernelLaunch = struct { function_name: []const u8, launch: LaunchConfig, argument_count: usize };

pub fn getComputeCapabilityMajorMinor() struct { major: u16, minor: u16 } {
    return .{ .major = 0, .minor = 0 };
}
pub fn getDeviceInfo(device: i32) DeviceInfo {
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
    launch: LaunchConfig,
    args: []const KernelArgument,
) KernelLaunch {
    return .{
        .function_name = function.symbol,
        .launch = launch,
        .argument_count = args.len,
    };
}
pub fn streamSync(_: Stream) void {}
pub fn streamCreate() Stream {
    return Stream{ .handle = 1 };
}
pub fn streamDestroy(_: Stream) void {}
pub fn contextDestroy(_: DevicePrimaryContext) void {}
pub fn allocate(
    bytes: usize,
    dtype: @import("typing.zig").Numeric,
    memspace: @import("typing.zig").AddressSpace,
) Error!Pointer {
    return Pointer.init(bytes, dtype, memspace, 1);
}
pub fn free(_: Pointer) void {}
pub fn memcpyHtod(_: Pointer, _: []const u8) void {}
pub fn memcpyDtoh(_: []u8, _: Pointer) void {}
pub fn memcpyDtod(_: Pointer, _: Pointer, _: usize) void {}
pub fn eventCreate() usize {
    return 1;
}
pub fn eventDestroy(_: usize) void {}
pub fn eventRecord(_: usize, _: Stream) void {}
pub fn eventElapsedTimeMs(_: usize, _: usize) f32 {
    return 0.0;
}

test "runtime_cuda: descriptor functions build launch metadata without CUDA driver" {
    const m = loadCubinModule("abc");
    const f = getKernelFunction(m, "kernel");
    const launch = LaunchConfig.init(
        .{ .x = 1 },
        .{ .x = 1 },
        0,
        Stream.default(),
    ) catch unreachable;
    const k = launchKernel(f, launch, &.{});
    try std.testing.expectEqualStrings("kernel", k.function_name);
}

pub const ToolPaths = struct {
    /// In the original Python CuteDSL source this is not a shipped executable;
    /// the analogous work is done through cutlass._mlir.passmanager.  In this
    /// Zig port it names the external opt-like verifier/lowering binary.
    cute_opt: []const u8 = "cute-opt",
    mlir_opt: []const u8 = "mlir-opt",
    filecheck: []const u8 = "FileCheck",
    ptxas: []const u8 = "ptxas",
    nvcc: []const u8 = "nvcc",
    cuda_driver_library: []const u8 = "libcuda.so.1",
    cutlass_runtime_library: []const u8 = "libcutlass_cute_so",

    pub fn validate(self: ToolPaths) Error!void {
        const paths = [_][]const u8{ self.cute_opt, self.mlir_opt, self.filecheck, self.ptxas, self.nvcc, self.cuda_driver_library };
        for (paths) |p| if (p.len == 0) return Error.InvalidToolPath;
    }
};

pub const CompileFlavor = enum {
    /// Mirrors cutlass/cutlass_dsl/cutlass.py CutlassBaseDSL._get_pipeline.
    cutlass_dsl,
    /// Mirrors CuteExperimentalDSL._get_pipeline, with LIR-to-Cute finalization.
    cute_experimental_lir,
};

pub const CompileOptions = struct {
    opt_level: u8 = 3,
    arch: []const u8 = "sm_90",
    ptxas_options: []const u8 = "",
    link_libraries: []const u8 = "",
    dump_dir: []const u8 = ".",
    function_name: []const u8 = "kernel",
    enable_assertions: bool = false,
    preserve_line_info: bool = false,
    keep_ptx: bool = false,
    keep_cubin: bool = true,
    enable_cuda_dialect: bool = true,
    cuda_dialect_external_module: bool = true,

    pub fn validate(self: CompileOptions) Error!void {
        if (self.opt_level > 3) return Error.InvalidCompileOption;
        if (self.arch.len == 0 or self.function_name.len == 0)
            return Error.InvalidCompileOption;
        try mlir.validateSymbol(self.function_name);
    }

    pub fn writeCuteDslOptionString(self: CompileOptions, out: anytype) Error!void {
        try self.validate();
        try out.append("opt-level=");
        try out.appendUnsigned(self.opt_level);
        try out.append(" ");
        if (self.ptxas_options.len != 0) {
            try out.append("ptx-options='");
            try out.append(self.ptxas_options);
            try out.append("' ");
        }
        if (self.enable_assertions) try out.append("enable-assertions=true ");
        if (self.preserve_line_info) try out.append("preserve-line-info=true ");
        if (self.link_libraries.len != 0) {
            try out.append("link-libraries='");
            try out.append(self.link_libraries);
            try out.append("' ");
        }
        try out.append("cubin-chip='");
        try out.append(self.arch);
        try out.append("' ");
        if (self.keep_ptx) {
            try out.append("dump-ptx-path='");
            try out.append(self.dump_dir);
            try out.append("/");
            try out.append(self.function_name);
            try out.append("' ");
        }
        if (self.keep_cubin) {
            try out.append("dump-cubin-path='");
            try out.append(self.dump_dir);
            try out.append("/");
            try out.append(self.function_name);
            try out.append("' ");
        }
    }

    pub fn writePipeline(
        self: CompileOptions,
        flavor: CompileFlavor,
        out: anytype,
    ) Error!void {
        var opts: mlir.TextBuffer(1024) = .{};
        try self.writeCuteDslOptionString(&opts);
        switch (flavor) {
            .cutlass_dsl => {
                try out.append("builtin.module(cute-to-nvvm{cubin-format=bin ");
                if (self.enable_cuda_dialect) try out.append("enable-cuda-dialect=true ");
                if (self.cuda_dialect_external_module) try out.append("cuda-dialect-external-module=true ");
                try out.append(opts.slice());
                try out.append("})");
            },
            .cute_experimental_lir => {
                try out.append("builtin.module(gpu.module(lir-to-cute{enable-cuda-dialect enable-lir-func-finalization=false}), ");
                try out.append("lir-func-finalization{enable-cuda-dialect=true}, ");
                try out.append("cute-to-nvvm{cubin-format=bin enable-cuda-dialect ");
                try out.append(opts.slice());
                try out.append("})");
            },
        }
    }
};

pub const ArtifactPaths = struct {
    work_dir: []const u8 = "build/not-cute",
    mlir_path: []const u8,
    cubin_path: []const u8,
    ptx_path: ?[]const u8 = null,
    object_path: ?[]const u8 = null,
    header_path: ?[]const u8 = null,
    source_path: ?[]const u8 = null,

    pub fn init(work_dir: []const u8, stem: []const u8) Error!ArtifactPaths {
        if (work_dir.len == 0 or stem.len == 0) return Error.InvalidArtifactPath;
        try mlir.validateSymbol(stem);
        return .{
            .work_dir = work_dir,
            .mlir_path = stem ++ ".mlir",
            .cubin_path = stem ++ ".cubin",
            .ptx_path = stem ++ ".ptx",
            .object_path = stem ++ ".o",
            .header_path = stem ++ ".h",
            .source_path = stem ++ ".c",
        };
    }
};

pub const CompilePlan = struct {
    tools: ToolPaths = .{},
    options: CompileOptions = .{},
    flavor: CompileFlavor = .cutlass_dsl,
    input_mlir: []const u8,
    output_cubin: []const u8,

    pub fn writePipeline(self: CompilePlan, out: anytype) Error!void {
        try self.options.writePipeline(self.flavor, out);
    }

    pub fn writeCuteOptCommand(self: CompilePlan, out: anytype) Error!void {
        try self.tools.validate();
        try out.append(self.tools.cute_opt);
        try out.append(" --pass-pipeline=");
        var pipeline: mlir.TextBuffer(2048) = .{};
        try self.writePipeline(&pipeline);
        try out.appendQuotedString(pipeline.slice());
        try out.append(" ");
        try out.append(self.input_mlir);
        try out.append(" -o ");
        try out.append(self.output_cubin);
    }

    pub fn writeVerifierCommand(self: CompilePlan, out: anytype) Error!void {
        try self.tools.validate();
        try out.append(self.tools.cute_opt);
        try out.append(" --verify-diagnostics ");
        try out.append(self.input_mlir);
    }

    pub fn writeCacheKey(
        self: CompilePlan,
        signature: *const jit.JitSignature,
        out: anytype,
    ) Error!void {
        try out.append("tool=");
        try out.append(self.tools.cute_opt);
        try out.append(";flavor=");
        try out.append(@tagName(self.flavor));
        try out.append(";input=");
        try out.append(self.input_mlir);
        try out.append(";output=");
        try out.append(self.output_cubin);
        try out.append(";pipeline=");
        try self.writePipeline(out);
        try out.append(";signature=");
        try signature.writeCacheKey(out);
    }

    pub fn hashCacheKey(
        self: CompilePlan,
        signature: *const jit.JitSignature,
    ) Error!u64 {
        var buf: mlir.TextBuffer(4096) = .{};
        try self.writeCacheKey(signature, &buf);
        return std.hash.Wyhash.hash(0, buf.slice());
    }
};

pub const RuntimeSymbols = struct {
    prefix: []const u8,
    function_name: []const u8,

    pub fn init(prefix: []const u8, function_name: []const u8) Error!RuntimeSymbols {
        if (prefix.len == 0) return Error.InvalidRuntimeSymbol;
        try mlir.validateSymbol(function_name);
        return .{ .prefix = prefix, .function_name = function_name };
    }

    pub fn writeCudaInit(self: RuntimeSymbols, out: anytype) Error!void {
        try out.append("_mlir_");
        try out.append(self.prefix);
        try out.append("_cuda_init");
    }

    pub fn writeCudaLoad(self: RuntimeSymbols, out: anytype) Error!void {
        try out.append("_mlir_");
        try out.append(self.prefix);
        try out.append("_cuda_load");
    }

    pub fn writeCudaLoadToDevice(self: RuntimeSymbols, out: anytype) Error!void {
        try out.append("_mlir_");
        try out.append(self.prefix);
        try out.append("_cuda_load_to_device");
    }

    pub fn writeCInterface(self: RuntimeSymbols, out: anytype) Error!void {
        try out.append("_mlir_");
        try out.append(self.prefix);
        try out.append("__mlir_ciface_");
        try out.append(self.function_name);
    }

    pub fn writeKernelEntry(self: RuntimeSymbols, out: anytype) Error!void {
        try out.append(self.function_name);
    }
};

pub const PackedArgKind = enum { pointer, tensor, stream, scalar_i64, scalar_u64, scalar_f64, opaque_ref };

pub const PackedArg = struct {
    name: []const u8,
    kind: PackedArgKind,
    bytes: []const u8 = "",
    pointer: ?Pointer = null,
    tensor: ?TensorDescriptor = null,
    stream: ?Stream = null,

    pub fn ptr(name: []const u8, p: Pointer) Error!PackedArg {
        try mlir.validateSymbol(name);
        return .{ .name = name, .kind = .pointer, .pointer = p };
    }

    pub fn tensorArg(name: []const u8, t: TensorDescriptor) Error!PackedArg {
        try mlir.validateSymbol(name);
        return .{ .name = name, .kind = .tensor, .tensor = t };
    }

    pub fn streamArg(name: []const u8, s: Stream) Error!PackedArg {
        try mlir.validateSymbol(name);
        return .{ .name = name, .kind = .stream, .stream = s };
    }

    pub fn scalarBytes(
        name: []const u8,
        kind: PackedArgKind,
        bytes: []const u8,
    ) Error!PackedArg {
        try mlir.validateSymbol(name);
        if (bytes.len == 0) return Error.InvalidPackedArgument;
        return .{ .name = name, .kind = kind, .bytes = bytes };
    }

    pub fn runtimeSlotCount(self: PackedArg) usize {
        return switch (self.kind) {
            .tensor => 1,
            else => 1,
        };
    }

    pub fn writeDebug(self: PackedArg, out: anytype) Error!void {
        try out.append(self.name);
        try out.append(":");
        try out.append(@tagName(self.kind));
        if (self.pointer) |p| {
            try out.append("@");
            try out.append(p.memspace().mlirName());
        }
    }
};

pub const ArgPack = struct {
    args: [64]PackedArg = undefined,
    len: usize = 0,

    pub fn append(self: *ArgPack, arg: PackedArg) Error!void {
        if (self.len >= self.args.len) return Error.TooManyPackedArguments;
        self.args[self.len] = arg;
        self.len += 1;
    }

    pub fn runtimeSlotCount(self: *const ArgPack) usize {
        var n: usize = 0;
        for (self.args[0..self.len]) |arg| n += arg.runtimeSlotCount();
        return n;
    }

    pub fn writeDebug(self: *const ArgPack, out: anytype) Error!void {
        for (self.args[0..self.len], 0..) |arg, i| {
            if (i != 0) try out.append(",");
            try arg.writeDebug(out);
        }
    }
};

pub const LaunchPlan = struct {
    symbols: RuntimeSymbols,
    module: BinaryModule,
    config: LaunchConfig,
    args: ArgPack,

    pub fn prepareRecord(self: LaunchPlan) Error!LaunchRecord {
        const k = try KernelFunction.init(
            self.module,
            self.symbols.function_name,
        );
        return recordLaunch(k, self.config, self.args.runtimeSlotCount());
    }

    pub fn writeManifest(self: LaunchPlan, out: anytype) Error!void {
        try out.append("{\n");
        try out.append("  \"module\": \"");
        try out.append(self.module.image_path);
        try out.append("\",\n  \"kernel\": \"");
        try out.append(self.symbols.function_name);
        try out.append("\",\n  \"grid\": [");
        try out.appendUnsigned(self.config.grid.x);
        try out.append(", ");
        try out.appendUnsigned(self.config.grid.y);
        try out.append(", ");
        try out.appendUnsigned(self.config.grid.z);
        try out.append("],\n  \"block\": [");
        try out.appendUnsigned(self.config.block.x);
        try out.append(", ");
        try out.appendUnsigned(self.config.block.y);
        try out.append(", ");
        try out.appendUnsigned(self.config.block.z);
        try out.append("],\n  \"dynamic_smem_bytes\": ");
        try out.appendUnsigned(self.config.dynamic_smem_bytes);
        try out.append(",\n  \"argument_slots\": ");
        try out.appendUnsigned(self.args.runtimeSlotCount());
        try out.append("\n}\n");
    }
};

pub const DriverLoadPlan = struct {
    tools: ToolPaths = .{},
    symbols: RuntimeSymbols,
    binary_path: []const u8,
    device_id: i32 = 0,

    pub fn writePseudoSequence(self: DriverLoadPlan, out: anytype) Error!void {
        try self.tools.validate();
        try out.append("dlopen ");
        try out.append(self.tools.cuda_driver_library);
        try out.append("\nresolve cuInit/cuDeviceGet/cuCtxCreate/cuModuleLoad/cuModuleGetFunction/cuLaunchKernel\n");
        try out.append("load binary ");
        try out.append(self.binary_path);
        try out.append(" for device ");
        try out.appendSigned(self.device_id);
        try out.append("\nresolve MLIR helper symbols: ");
        try self.symbols.writeCudaInit(out);
        try out.append(", ");
        try self.symbols.writeCudaLoad(out);
        try out.append(", ");
        try self.symbols.writeCInterface(out);
        try out.append("\n");
    }
};

pub const CudaDriverAbi = struct {
    pub const cuInit = "cuInit";
    pub const cuDeviceGet = "cuDeviceGet";
    pub const cuCtxCreate = "cuCtxCreate_v2";
    pub const cuModuleLoad = "cuModuleLoad";
    pub const cuModuleLoadData = "cuModuleLoadData";
    pub const cuModuleGetFunction = "cuModuleGetFunction";
    pub const cuLaunchKernel = "cuLaunchKernel";
    pub const cuStreamSynchronize = "cuStreamSynchronize";
    pub const cuGetErrorString = "cuGetErrorString";
};

pub fn writeAotHeader(
    out: anytype,
    symbols: RuntimeSymbols,
    args: []const export_.CArgument,
) Error!void {
    var cfg = try export_.WrapperConfig.init(
        symbols.function_name,
        symbols.function_name,
    );
    cfg.extern_c = true;
    try export_.writeCompleteHeader(out, cfg, args);
}

pub fn writeCInterfaceDeclaration(out: anytype, symbols: RuntimeSymbols) Error!void {
    try out.append("extern void ");
    try symbols.writeCInterface(out);
    try out.append("(void **args);\n");
}

pub fn writeCInterfaceWrapperSource(
    out: anytype,
    symbols: RuntimeSymbols,
    args: []const export_.CArgument,
) Error!void {
    try out.append("#include <stdint.h>\n#include <stdbool.h>\n\n");
    try writeCInterfaceDeclaration(out, symbols);
    try out.append("int ");
    try out.append(symbols.function_name);
    try out.append("(");
    for (args, 0..) |arg, i| {
        if (i != 0) try out.append(", ");
        try out.append(arg.c_type);
        try out.append(" ");
        try out.append(arg.name);
    }
    try out.append(") {\n");
    try out.append("  void *packed[");
    try out.appendUnsigned(args.len);
    try out.append("];\n");
    for (args, 0..) |arg, i| {
        try out.append("  packed[");
        try out.appendUnsigned(i);
        try out.append("] = (void *)&");
        try out.append(arg.name);
        try out.append(";\n");
    }
    try out.append("  ");
    try symbols.writeCInterface(out);
    try out.append("(packed);\n  return 0;\n}\n");
}

pub fn sourceFindingCuteOptSummary(out: anytype) Error!void {
    try out.append("Uploaded CuteDSL source does not contain a standalone cute-opt executable. ");
    try out.append("cutlass/base_dsl/compiler.py calls cutlass._mlir.passmanager.PassManager.parse(...).run(...), ");
    try out.append("and cutlass._mlir.execution_engine.ExecutionEngine for JIT. ");
    try out.append("cutlass/cutlass_dsl/cutlass.py builds cute-to-nvvm or lir-to-cute-to-nvvm pipelines.");
}

test "runtime_plan: CuteDSL-style compile option and pipeline strings" {
    const opts: CompileOptions = .{
        .function_name = "gemm",
        .arch = "sm_100",
        .keep_ptx = true,
        .preserve_line_info = true,
    };
    var p: mlir.TextBuffer(2048) = .{};
    try opts.writePipeline(.cutlass_dsl, &p);
    try std.testing.expect(std.mem.indexOf(u8, p.slice(), "cute-to-nvvm") != null);
    try std.testing.expect(std.mem.indexOf(u8, p.slice(), "cubin-format=bin") != null);
    try std.testing.expect(std.mem.indexOf(u8, p.slice(), "cubin-chip='sm_100'") != null);
    try std.testing.expect(std.mem.indexOf(u8, p.slice(), "dump-ptx-path='./gemm'") != null);
}

test "runtime_plan: compile command and cache key are deterministic" {
    var sig: jit.JitSignature = .{};
    try sig.append(try jit.JitArgument.init("A", .pointer, mlir.Type.raw("!cute.ptr")));
    const plan: CompilePlan = .{
        .options = .{ .function_name = "kernel" },
        .input_mlir = "kernel.mlir",
        .output_cubin = "kernel.cubin",
    };
    var cmd: mlir.TextBuffer(4096) = .{};
    try plan.writeCuteOptCommand(&cmd);
    try std.testing.expect(std.mem.startsWith(u8, cmd.slice(), "cute-opt --pass-pipeline="));
    try std.testing.expect(std.mem.indexOf(u8, cmd.slice(), "kernel.mlir -o kernel.cubin") != null);
    const h1 = try plan.hashCacheKey(&sig);
    const h2 = try plan.hashCacheKey(&sig);
    try std.testing.expectEqual(h1, h2);
}

test "runtime_plan: MLIR runtime symbol names mirror upstream convention" {
    const syms = try RuntimeSymbols.init("gemm", "cutlass_host_func");
    var out: mlir.TextBuffer(512) = .{};
    try syms.writeCudaInit(&out);
    try std.testing.expectEqualStrings("_mlir_gemm_cuda_init", out.slice());
    out.clear();
    try syms.writeCInterface(&out);
    try std.testing.expectEqualStrings(
        "_mlir_gemm__mlir_ciface_cutlass_host_func",
        out.slice(),
    );
}

test "runtime_plan: packed arguments prepare a launch record" {
    const ptr = try Pointer.init(
        0x1000,
        @import("typing.zig").Float32,
        .gmem,
        null,
    );
    var args: ArgPack = .{};
    try args.append(try PackedArg.ptr("A", ptr));
    try args.append(try PackedArg.streamArg("stream", .{}));
    const lp: LaunchPlan = .{
        .symbols = try RuntimeSymbols.init("gemm", "kernel_main"),
        .module = try BinaryModule.init("gemm.cubin", .cubin),
        .config = try LaunchConfig.init(
            try Dim3.init(2, 1, 1),
            try Dim3.init(128, 1, 1),
            0,
            .{},
        ),
        .args = args,
    };
    const rec = try lp.prepareRecord();
    try std.testing.expectEqual(@as(usize, 2), rec.argument_count);
    var manifest: mlir.TextBuffer(1024) = .{};
    try lp.writeManifest(&manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest.slice(), "\"kernel\": \"kernel_main\"") != null);
}

test "runtime_plan: AOT wrapper source packs C arguments for MLIR C interface" {
    var args: export_.CHeaderArguments = .{};
    try args.append(try export_.CArgument.init("A", .pointer, "void *"));
    try args.append(try export_.CArgument.init("M", .scalar, "int32_t"));
    const syms = try RuntimeSymbols.init("gemm", "launch_gemm");
    var out: mlir.TextBuffer(2048) = .{};
    try writeCInterfaceWrapperSource(&out, syms, args.slice());
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "_mlir_gemm__mlir_ciface_launch_gemm") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "packed[1] = (void *)&M") != null);
}

test "runtime_plan: source finding summary records no cute-opt binary" {
    var out: mlir.TextBuffer(512) = .{};
    try sourceFindingCuteOptSummary(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "does not contain a standalone cute-opt") != null);
}


