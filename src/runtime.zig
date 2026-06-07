const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");

pub const Error = layout.Error || mlir.Error || typing.Error || error{
    MisalignedPointer,
    InvalidDevice,
    InvalidLaunchShape,
    InvalidTensorDescriptor,
    InvalidDynamicMask,
    InvalidLibraryPath,
    InvalidSymbol,
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
