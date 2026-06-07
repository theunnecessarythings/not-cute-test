const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const runtime = @import("runtime.zig");
const cuda = @import("cuda_driver.zig");
const mlir = @import("mlir_text.zig");

pub const Error = runtime.Error || typing.Error || layout.Error || cuda.Error || mlir.Error || error{
    InvalidBuffer,
    InvalidAlignment,
    InvalidOwnership,
    InvalidTransfer,
    InvalidInteropDescriptor,
    OutOfBounds,
    OutOfMemory,
};

pub const Ownership = enum {
    borrowed,
    owned,
    external,
    managed,
};

pub const MemoryKind = enum {
    host,
    device,
    managed,
    external,
};

pub const AlignmentPolicy = struct {
    bytes: usize = 16,

    pub fn init(bytes: usize) Error!AlignmentPolicy {
        if (bytes == 0 or !std.math.isPowerOfTwo(bytes)) return Error.InvalidAlignment;
        return .{ .bytes = bytes };
    }

    pub fn validateAddress(self: AlignmentPolicy, address: usize) Error!void {
        if (address != 0 and address % self.bytes != 0) return Error.InvalidAlignment;
    }
};

pub const BufferDescriptor = struct {
    kind: MemoryKind,
    ownership: Ownership,
    address: usize,
    bytes: usize,
    alignment: AlignmentPolicy,
    device_ordinal: ?i32 = null,

    pub fn validate(self: BufferDescriptor) Error!void {
        if (self.bytes == 0) return Error.InvalidBuffer;
        try self.alignment.validateAddress(self.address);
        if (self.kind == .device and self.device_ordinal == null)
            return Error.InvalidBuffer;
        if (self.ownership == .borrowed and self.address == 0)
            return Error.InvalidOwnership;
    }

    pub fn runtimePointer(
        self: BufferDescriptor,
        dtype: typing.Numeric,
    ) Error!runtime.Pointer {
        try self.validate();
        const memspace: typing.AddressSpace = switch (self.kind) {
            .host => .generic,
            .device => .gmem,
            .managed => .generic,
            .external => .generic,
        };
        return runtime.Pointer.init(
            self.address,
            dtype,
            memspace,
            self.alignment.bytes,
        );
    }
};

pub const HostBuffer = struct {
    allocator: ?std.mem.Allocator = null,
    bytes: []u8 = &.{},
    ownership: Ownership = .borrowed,
    alignment: AlignmentPolicy = .{},

    pub fn allocate(
        allocator: std.mem.Allocator,
        byte_count: usize,
        alignment: AlignmentPolicy,
    ) Error!HostBuffer {
        if (byte_count == 0) return Error.InvalidBuffer;
        const ptr = allocator.rawAlloc(
            byte_count,
            std.mem.Alignment.fromByteUnits(alignment.bytes),
            @returnAddress(),
        ) orelse return Error.OutOfMemory;
        const mem = ptr[0..byte_count];
        return .{
            .allocator = allocator,
            .bytes = mem,
            .ownership = .owned,
            .alignment = alignment,
        };
    }

    pub fn borrow(bytes: []u8, alignment: AlignmentPolicy) Error!HostBuffer {
        if (bytes.len == 0) return Error.InvalidBuffer;
        try alignment.validateAddress(@intFromPtr(bytes.ptr));
        return .{ .bytes = bytes, .ownership = .borrowed, .alignment = alignment };
    }

    pub fn deinit(self: *HostBuffer) void {
        if (self.ownership == .owned) {
            if (self.allocator) |a| a.rawFree(self.bytes, std.mem.Alignment.fromByteUnits(self.alignment.bytes), @returnAddress());
        }
        self.bytes = &.{};
        self.allocator = null;
        self.ownership = .borrowed;
    }

    pub fn descriptor(self: HostBuffer) Error!BufferDescriptor {
        if (self.bytes.len == 0) return Error.InvalidBuffer;
        return .{
            .kind = .host,
            .ownership = self.ownership,
            .address = @intFromPtr(self.bytes.ptr),
            .bytes = self.bytes.len,
            .alignment = self.alignment,
        };
    }
};

pub const DeviceBuffer = struct {
    handle: cuda.DeviceMemory,
    ownership: Ownership = .borrowed,
    device_ordinal: i32 = 0,
    alignment: AlignmentPolicy = .{},

    pub fn external(
        ptr: cuda.CUdeviceptr,
        byte_count: usize,
        device_ordinal: i32,
        alignment: AlignmentPolicy,
    ) Error!DeviceBuffer {
        if (ptr == 0 or byte_count == 0) return Error.InvalidBuffer;
        return .{
            .handle = .{ .ptr = ptr, .bytes = byte_count },
            .ownership = .external,
            .device_ordinal = device_ordinal,
            .alignment = alignment,
        };
    }

    pub fn allocate(
        driver: cuda.DriverSymbols,
        byte_count: usize,
        device_ordinal: i32,
        alignment: AlignmentPolicy,
    ) Error!DeviceBuffer {
        if (byte_count == 0) return Error.InvalidBuffer;
        const mem = try cuda.allocateDevice(driver, byte_count);
        return .{
            .handle = mem,
            .ownership = .owned,
            .device_ordinal = device_ordinal,
            .alignment = alignment,
        };
    }

    pub fn deinit(self: *DeviceBuffer, driver: cuda.DriverSymbols) Error!void {
        if (self.ownership == .owned and self.handle.ptr != 0) try cuda.freeDevice(
            driver,
            self.handle,
        );
        self.handle = .{};
        self.ownership = .borrowed;
    }

    pub fn descriptor(self: DeviceBuffer) Error!BufferDescriptor {
        if (self.handle.ptr == 0 or self.handle.bytes == 0) return Error.InvalidBuffer;
        return .{
            .kind = .device,
            .ownership = self.ownership,
            .address = @intCast(self.handle.ptr),
            .bytes = self.handle.bytes,
            .alignment = self.alignment,
            .device_ordinal = self.device_ordinal,
        };
    }
};

pub const ManagedPointer = struct {
    address: usize,
    bytes: usize,
    ownership: Ownership = .managed,
    alignment: AlignmentPolicy = .{},

    pub fn init(
        address: usize,
        bytes: usize,
        alignment: AlignmentPolicy,
    ) Error!ManagedPointer {
        if (address == 0 or bytes == 0) return Error.InvalidBuffer;
        try alignment.validateAddress(address);
        return .{ .address = address, .bytes = bytes, .alignment = alignment };
    }

    pub fn descriptor(self: ManagedPointer) Error!BufferDescriptor {
        return .{
            .kind = .managed,
            .ownership = self.ownership,
            .address = self.address,
            .bytes = self.bytes,
            .alignment = self.alignment,
        };
    }
};

pub const ExternalPointer = struct {
    address: usize,
    bytes: usize,
    kind: MemoryKind,
    device_ordinal: ?i32 = null,
    alignment: AlignmentPolicy = .{},

    pub fn init(
        address: usize,
        bytes: usize,
        kind: MemoryKind,
        alignment: AlignmentPolicy,
        device_ordinal: ?i32,
    ) Error!ExternalPointer {
        if (address == 0 or bytes == 0) return Error.InvalidBuffer;
        try alignment.validateAddress(address);
        if (kind == .device and device_ordinal == null) return Error.InvalidBuffer;
        return .{
            .address = address,
            .bytes = bytes,
            .kind = kind,
            .alignment = alignment,
            .device_ordinal = device_ordinal,
        };
    }

    pub fn descriptor(self: ExternalPointer) Error!BufferDescriptor {
        return .{
            .kind = self.kind,
            .ownership = .external,
            .address = self.address,
            .bytes = self.bytes,
            .alignment = self.alignment,
            .device_ordinal = self.device_ordinal,
        };
    }
};

pub const DLDeviceType = enum(i32) {
    cpu = 1,
    cuda = 2,
    cuda_host = 3,
    cuda_managed = 13,
};

pub const DLDataType = struct {
    code: u8,
    bits: u8,
    lanes: u16 = 1,

    pub fn fromNumeric(dtype: typing.Numeric) DLDataType {
        const code: u8 = switch (dtype.kind) {
            .unsigned_int => 1,
            .float, .bfloat, .tfloat, .fp8_e5m2, .fp8_e4m3fn, .fp8_e4m3b11fnuz, .fp8_e4m3, .fp8_e8m0fnu, .fp4_e2m1fn, .fp6_e2m3fn, .fp6_e3m2fn => 2,
            else => 0,
        };
        return .{ .code = code, .bits = @intCast(dtype.width), .lanes = 1 };
    }
};

pub const DLPackDescriptor = struct {
    data: usize,
    device_type: DLDeviceType,
    device_id: i32,
    dtype: DLDataType,
    ndim: usize,
    shape: [8]i64 = [_]i64{0} ** 8,
    strides: [8]i64 = [_]i64{0} ** 8,
    byte_offset: usize = 0,

    pub fn validate(self: DLPackDescriptor) Error!void {
        if (self.data == 0 or self.ndim == 0 or self.ndim > self.shape.len)
            return Error.InvalidInteropDescriptor;
        for (0..self.ndim) |i| if (self.shape[i] <= 0) return Error.InvalidInteropDescriptor;
    }
};

pub const TensorView = struct {
    buffer: BufferDescriptor,
    dtype: typing.Numeric,
    shape: layout.Tree,
    stride: layout.Tree,
    byte_offset: usize = 0,

    pub fn init(
        buffer: BufferDescriptor,
        dtype: typing.Numeric,
        shape: layout.Tree,
        stride: layout.Tree,
        byte_offset: usize,
    ) Error!TensorView {
        try buffer.validate();
        if (!shape.sameProfile(&stride)) return Error.InvalidBuffer;
        const elems = try shape.product();
        const need = @as(usize, @intCast(elems)) * dtype.bytes() + byte_offset;
        if (need > buffer.bytes) return Error.OutOfBounds;
        return .{
            .buffer = buffer,
            .dtype = dtype,
            .shape = shape,
            .stride = stride,
            .byte_offset = byte_offset,
        };
    }

    pub fn compact(
        buffer: BufferDescriptor,
        dtype: typing.Numeric,
        shape: layout.Tree,
    ) Error!TensorView {
        const compact_layout = try layout.Layout.makeCompact(shape);
        return init(buffer, dtype, compact_layout.shape, compact_layout.stride, 0);
    }

    pub fn runtimeDescriptor(self: TensorView) Error!runtime.TensorDescriptor {
        const ptr = try runtime.Pointer.init(self.buffer.address + self.byte_offset, self.dtype, switch (self.buffer.kind) {
            .host => .generic,
            .device => .gmem,
            .managed => .generic,
            .external => .generic,
        }, self.buffer.alignment.bytes);
        return runtime.TensorDescriptor.init(ptr, self.shape, self.stride);
    }

    pub fn toDLPack(self: TensorView) Error!DLPackDescriptor {
        try self.buffer.validate();
        const shape_flat = try self.shape.flattenLeaves();
        const stride_flat = try self.stride.flattenLeaves();
        if (shape_flat.len > 8 or stride_flat.len > 8)
            return Error.InvalidInteropDescriptor;
        var out: DLPackDescriptor = .{
            .data = self.buffer.address,
            .device_type = switch (self.buffer.kind) {
                .host => .cpu,
                .device => .cuda,
                .managed => .cuda_managed,
                .external => if (self.buffer.device_ordinal == null) .cpu else .cuda,
            },
            .device_id = self.buffer.device_ordinal orelse 0,
            .dtype = DLDataType.fromNumeric(self.dtype),
            .ndim = shape_flat.len,
            .byte_offset = self.byte_offset,
        };
        for (0..shape_flat.len) |i| out.shape[i] = @intCast(shape_flat.at(i));
        for (0..stride_flat.len) |i| out.strides[i] = @intCast(stride_flat.at(i));
        try out.validate();
        return out;
    }
};

pub const TransferKind = enum {
    host_to_device,
    device_to_host,
    device_to_device,
    host_to_host,
};

pub const TransferPlan = struct {
    kind: TransferKind,
    bytes: usize,
    src_address: usize,
    dst_address: usize,

    pub fn validate(self: TransferPlan) Error!void {
        if (self.bytes == 0 or self.src_address == 0 or self.dst_address == 0)
            return Error.InvalidTransfer;
    }
};

pub fn transferPlan(
    dst: BufferDescriptor,
    src: BufferDescriptor,
    bytes: usize,
) Error!TransferPlan {
    try dst.validate();
    try src.validate();
    if (bytes == 0 or bytes > dst.bytes or bytes > src.bytes)
        return Error.InvalidTransfer;
    const kind: TransferKind = switch (dst.kind) {
        .host => switch (src.kind) {
            .device => .device_to_host,
            else => .host_to_host,
        },
        .device => switch (src.kind) {
            .host, .managed, .external => .host_to_device,
            .device => .device_to_device,
        },
        .managed, .external => .host_to_host,
    };
    return .{
        .kind = kind,
        .bytes = bytes,
        .src_address = src.address,
        .dst_address = dst.address,
    };
}

pub fn copyHostToDevice(
    driver: cuda.DriverSymbols,
    dst: DeviceBuffer,
    src: HostBuffer,
) Error!void {
    try cuda.memcpyHtoD(driver, dst.handle, src.bytes);
}

pub fn copyDeviceToHost(
    driver: cuda.DriverSymbols,
    dst: HostBuffer,
    src: DeviceBuffer,
) Error!void {
    try cuda.memcpyDtoH(driver, dst.bytes, src.handle);
}

pub fn copyDeviceToDevice(
    driver: cuda.DriverSymbols,
    dst: DeviceBuffer,
    src: DeviceBuffer,
    bytes: usize,
) Error!void {
    try cuda.memcpyDtoD(driver, dst.handle, src.handle, bytes);
}

pub fn writeOwnershipJson(plan: TransferPlan, out: anytype) Error!void {
    try plan.validate();
    try out.append("{\n  \"kind\": ");
    try out.appendQuotedString(@tagName(plan.kind));
    try out.append(",\n  \"bytes\": ");
    try out.appendUnsigned(plan.bytes);
    try out.append(",\n  \"src\": ");
    try out.appendUnsigned(plan.src_address);
    try out.append(",\n  \"dst\": ");
    try out.appendUnsigned(plan.dst_address);
    try out.append("\n}\n");
}

test "memory_model: host buffer allocation has descriptor and frees" {
    const alignment = try AlignmentPolicy.init(16);
    var host = try HostBuffer.allocate(std.testing.allocator, 64, alignment);
    defer host.deinit();
    const desc = try host.descriptor();
    try std.testing.expectEqual(MemoryKind.host, desc.kind);
    try std.testing.expectEqual(@as(usize, 64), desc.bytes);
}

test "memory_model: external device buffer and DLPack tensor view" {
    const alignment = try AlignmentPolicy.init(16);
    const dev = try DeviceBuffer.external(0x1000, 1024, 0, alignment);
    const shape = layout.Tree.fromComptime(.{ 4, 4 });
    const view = try TensorView.compact(try dev.descriptor(), typing.Float32, shape);
    const dl = try view.toDLPack();
    try std.testing.expectEqual(DLDeviceType.cuda, dl.device_type);
    try std.testing.expectEqual(@as(usize, 2), dl.ndim);
    try std.testing.expectEqual(@as(i64, 4), dl.shape[0]);
}

test "memory_model: transfer plan classifies host-device directions" {
    const alignment = try AlignmentPolicy.init(16);
    var storage: [64]u8 align(16) = undefined;
    const host = try HostBuffer.borrow(storage[0..], alignment);
    const dev = try DeviceBuffer.external(0x2000, 64, 0, alignment);
    const h2d = try transferPlan(try dev.descriptor(), try host.descriptor(), 64);
    try std.testing.expectEqual(TransferKind.host_to_device, h2d.kind);
    var out: mlir.TextBuffer(512) = .{};
    try writeOwnershipJson(h2d, &out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "host_to_device") != null);
}

test "memory_model: tensor view rejects out of bounds" {
    const alignment = try AlignmentPolicy.init(16);
    var storage: [8]u8 align(16) = undefined;
    const host = try HostBuffer.borrow(storage[0..], alignment);
    const shape = layout.Tree.fromComptime(.{4});
    try std.testing.expectError(
        Error.OutOfBounds,
        TensorView.compact(try host.descriptor(), typing.Float32, shape),
    );
}
