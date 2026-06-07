const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const atom = @import("atom.zig");
const nvgpu = @import("nvgpu.zig");
const arch_catalog = @import("arch_catalog.zig");
const arch_exact = @import("arch_exact.zig");

pub const Error = nvgpu.Error || atom.Error || typing.Error || error{
    UnsupportedArch,
    InvalidShape,
    InvalidFieldCombination,
};

pub const SmArch = arch_catalog.SmArch;
pub const CopyOptions = nvgpu.CopyOptions;
pub const MmaOptions = nvgpu.MmaOptions;
pub const OperandMajorMode = nvgpu.OperandMajorMode;
pub const OperandSource = nvgpu.OperandSource;
pub const CtaGroup = nvgpu.CtaGroup;
pub const MemoryOrder = nvgpu.MemoryOrder;
pub const MemoryScope = nvgpu.MemoryScope;
pub const LoadCacheMode = nvgpu.LoadCacheMode;
pub const StoreCacheMode = nvgpu.StoreCacheMode;
pub const CacheEvictionPriority = nvgpu.CacheEvictionPriority;
pub const L2PrefetchSize = nvgpu.L2PrefetchSize;

pub const ExactOp = struct {
    source_name: []const u8,
    source_module: []const u8,
    min_arch: SmArch,
    kind: atom.OpKind,
    family: []const u8,
    mlir_type_factory: []const u8 = "",
    source_line: u32 = 0,
};

pub const MmaConfig = struct {
    a_dtype: typing.Numeric = typing.Float16,
    b_dtype: typing.Numeric = typing.Float16,
    acc_dtype: typing.Numeric = typing.Float32,
    sf_dtype: typing.Numeric = typing.Float8E8M0FNU,
    shape_mnk: layout.Tree = layout.Tree.fromComptime(.{ 16, 8, 16 }),
    a_src: OperandSource = .smem,
    a_major_mode: OperandMajorMode = .K,
    b_major_mode: OperandMajorMode = .K,
    cta_group: CtaGroup = .one,
};

pub const CopyConfig = struct {
    dtype: typing.Numeric = typing.Float32,
    bits_per_copy: u16 = 32,
    transpose: bool = false,
    num_matrices: u2 = 1,
    unpack_bits: ?u16 = null,
    cache_mode: LoadCacheMode = .always,
    memory_order: MemoryOrder = .weak,
    memory_scope: MemoryScope = .cta,
    cta_group: CtaGroup = .one,
};

pub fn requireArch(required: SmArch, actual: SmArch) Error!void {
    if (@intFromEnum(actual) < @intFromEnum(required)) return Error.UnsupportedArch;
}

pub fn validateMmaConfig(config: MmaConfig) Error!void {
    const shape = try config.shape_mnk.flattenLeaves();
    if (shape.len != 3) return Error.InvalidShape;
    for (shape.slice()) |v| if (v <= 0) return Error.InvalidShape;
    if (config.a_dtype.width == 0 or config.b_dtype.width == 0 or config.acc_dtype.width == 0) return Error.InvalidNumericType;
}

pub fn validateCopyConfig(config: CopyConfig) Error!void {
    if (config.bits_per_copy == 0) return Error.InvalidCopyBits;
    if (config.num_matrices == 0) return Error.InvalidFieldCombination;
    if (config.dtype.width == 0) return Error.InvalidNumericType;
}

fn toNvgpuMmaOptions(config: MmaConfig) nvgpu.MmaOptions {
    return .{
        .instruction_shape_mnk = config.shape_mnk,
        .a_src = config.a_src,
        .a_major_mode = config.a_major_mode,
        .b_major_mode = config.b_major_mode,
        .cta_group = config.cta_group,
    };
}

fn toNvgpuCopyOptions(config: CopyConfig) nvgpu.CopyOptions {
    return .{
        .num_bits_per_copy = config.bits_per_copy,
        .memory_order = config.memory_order,
        .memory_scope = config.memory_scope,
        .load_cache_mode = config.cache_mode,
    };
}

pub fn makeMmaBySourceName(name: []const u8, arch: SmArch, config: MmaConfig) Error!atom.MmaAtom {
    try validateMmaConfig(config);
    const opts = toNvgpuMmaOptions(config);
    if (std.mem.eql(u8, name, "MmaUniversalOp")) return nvgpu.universalMma(config.acc_dtype);
    if (std.mem.eql(u8, name, "MmaF16BF16Op")) {
        if (@intFromEnum(arch) >= @intFromEnum(SmArch.sm100)) return nvgpu.tcgen05MmaF16BF16(config.a_dtype, config.acc_dtype, opts);
        if (@intFromEnum(arch) >= @intFromEnum(SmArch.sm90)) return nvgpu.warpgroupMmaF16BF16(config.a_dtype, config.acc_dtype, opts);
        return nvgpu.warpMmaF16BF16(config.a_dtype, config.acc_dtype, opts);
    }
    if (std.mem.eql(u8, name, "MmaFP8Op") or std.mem.eql(u8, name, "MmaF8Op")) return nvgpu.warpMmaFP8(config.a_dtype, config.acc_dtype, opts);
    if (std.mem.eql(u8, name, "MmaTF32Op")) return nvgpu.tcgen05BlockScaledMma("MmaTF32Op", typing.TFloat32, typing.TFloat32, typing.Float8E8M0FNU, opts);
    if (std.mem.eql(u8, name, "MmaI8Op")) return nvgpu.tcgen05BlockScaledMma("MmaI8Op", typing.Int8, typing.Int8, typing.Float8E8M0FNU, opts);
    if (std.mem.eql(u8, name, "MmaF8F6F4Op")) return nvgpu.tcgen05BlockScaledMma("MmaF8F6F4Op", config.a_dtype, config.b_dtype, config.sf_dtype, opts);
    if (std.mem.eql(u8, name, "MmaMXF8Op")) return nvgpu.tcgen05BlockScaledMma("MmaMXF8Op", config.a_dtype, config.b_dtype, config.sf_dtype, opts);
    if (std.mem.eql(u8, name, "MmaMXF8F6F4Op")) return nvgpu.tcgen05BlockScaledMma("MmaMXF8F6F4Op", config.a_dtype, config.b_dtype, config.sf_dtype, opts);
    if (std.mem.eql(u8, name, "MmaMXF4Op")) return nvgpu.tcgen05BlockScaledMma("MmaMXF4Op", typing.Float4E2M1FN, typing.Float4E2M1FN, config.sf_dtype, opts);
    if (std.mem.eql(u8, name, "MmaMXF4NVF4Op")) return nvgpu.tcgen05BlockScaledMma("MmaMXF4NVF4Op", typing.Float4E2M1FN, typing.Float4E2M1FN, config.sf_dtype, opts);
    return Error.UnknownOperation;
}

pub fn makeCopyBySourceName(name: []const u8, arch: SmArch, config: CopyConfig) Error!atom.CopyAtom {
    try validateCopyConfig(config);
    const opts = toNvgpuCopyOptions(config);
    _ = arch;
    if (std.mem.eql(u8, name, "CopyUniversalOp")) return nvgpu.copyUniversal(config.dtype, opts);
    if (std.mem.eql(u8, name, "CopyG2ROp")) return nvgpu.copyG2R(config.dtype, opts);
    if (std.mem.eql(u8, name, "CopyR2GOp")) return nvgpu.copyR2G(config.dtype, opts);
    if (std.mem.eql(u8, name, "CopyS2ROp")) return nvgpu.copyS2R(config.dtype, opts);
    if (std.mem.eql(u8, name, "CopyR2SOp")) return nvgpu.copyR2S(config.dtype, opts);
    if (std.mem.eql(u8, name, "CopyG2SOp")) return nvgpu.cpAsyncG2S(config.dtype, opts);
    if (startsWith(name, "CopyBulkTensor") or startsWith(name, "CopyReduceBulkTensor")) return nvgpu.tmaCopy(name, config.dtype, .gmem, .smem, opts);
    if (startsWith(name, "CopyBulkG2S")) return nvgpu.tmaCopy(name, config.dtype, .gmem, .smem, opts);
    if (startsWith(name, "CopyBulkS2G")) return nvgpu.tmaCopy(name, config.dtype, .smem, .gmem, opts);
    if (startsWith(name, "CopyBulkS2S")) return nvgpu.tmaCopy(name, config.dtype, .smem, .smem, opts);
    if (startsWith(name, "CopyDsmemStore")) return nvgpu.tmaCopy(name, config.dtype, .smem, .smem, opts);
    if (startsWith(name, "LdMatrix")) return nvgpu.warpLdMatrix(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "StMatrix")) return nvgpu.warpStMatrix(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "Ld") or startsWith(name, "LdRed")) return nvgpu.tcgen05Load(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "St")) return nvgpu.tcgen05Store(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "Cp")) return nvgpu.tcgen05S2T(name, config.dtype, config.bits_per_copy);
    return Error.UnknownOperation;
}

fn startsWith(haystack: []const u8, needle: []const u8) bool {
    return std.mem.startsWith(u8, haystack, needle);
}

pub fn MmaUniversalOp(dtype: typing.Numeric) Error!atom.MmaAtom {
    return nvgpu.universalMma(dtype);
}

pub fn MmaAtomSM80(config: MmaConfig) Error!atom.MmaAtom {
    try requireArch(.sm80, .sm80);
    return nvgpu.warpMmaF16BF16(config.a_dtype, config.acc_dtype, toNvgpuMmaOptions(config));
}

pub fn MmaAtomSM90(config: MmaConfig) Error!atom.MmaAtom {
    return nvgpu.warpgroupMmaF16BF16(config.a_dtype, config.acc_dtype, toNvgpuMmaOptions(config));
}

pub fn MmaAtomSM100(config: MmaConfig) Error!atom.MmaAtom {
    return nvgpu.tcgen05MmaF16BF16(config.a_dtype, config.acc_dtype, toNvgpuMmaOptions(config));
}

pub fn MmaF16BF16Op(config: MmaConfig, arch: SmArch) Error!atom.MmaAtom {
    return makeMmaBySourceName("MmaF16BF16Op", arch, config);
}

pub fn MmaFP8Op(config: MmaConfig, arch: SmArch) Error!atom.MmaAtom {
    return makeMmaBySourceName("MmaFP8Op", arch, config);
}

pub fn MmaI8Op(config: MmaConfig, arch: SmArch) Error!atom.MmaAtom {
    return makeMmaBySourceName("MmaI8Op", arch, config);
}

pub fn MmaMXF4Op(config: MmaConfig, arch: SmArch) Error!atom.MmaAtom {
    return makeMmaBySourceName("MmaMXF4Op", arch, config);
}

pub fn CopyUniversalOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.copyUniversal(config.dtype, toNvgpuCopyOptions(config));
}

pub fn CopyG2ROp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.copyG2R(config.dtype, toNvgpuCopyOptions(config));
}

pub fn CopyR2GOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.copyR2G(config.dtype, toNvgpuCopyOptions(config));
}

pub fn CopyS2ROp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.copyS2R(config.dtype, toNvgpuCopyOptions(config));
}

pub fn CopyR2SOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.copyR2S(config.dtype, toNvgpuCopyOptions(config));
}

pub fn CopyG2SOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.cpAsyncG2S(config.dtype, toNvgpuCopyOptions(config));
}

pub fn CopyAtomCpAsync(config: CopyConfig) Error!atom.CopyAtom {
    return CopyG2SOp(config);
}

pub fn CopyAtomTmaLoad(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tmaCopy("CopyBulkTensorTileG2SOp", config.dtype, .gmem, .smem, toNvgpuCopyOptions(config));
}

pub fn CopyAtomTmaStore(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tmaCopy("CopyBulkTensorTileS2GOp", config.dtype, .smem, .gmem, toNvgpuCopyOptions(config));
}

pub fn CopyAtomLdMatrix(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpLdMatrix("LdMatrix8x8x16bOp", config.dtype, config.bits_per_copy);
}

pub fn CopyAtomStMatrix(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpStMatrix("StMatrix8x8x16bOp", config.dtype, config.bits_per_copy);
}

pub fn LdMatrix8x8x16bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpLdMatrix("LdMatrix8x8x16bOp", config.dtype, 16 * @as(u16, config.num_matrices));
}

pub fn LdMatrix8x16x8bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpLdMatrix("LdMatrix8x16x8bOp", config.dtype, 8 * @as(u16, config.num_matrices));
}

pub fn LdMatrix16x8x8bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpLdMatrix("LdMatrix16x8x8bOp", config.dtype, 8 * @as(u16, config.num_matrices));
}

pub fn LdMatrix16x16x8bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpLdMatrix("LdMatrix16x16x8bOp", config.dtype, 8 * @as(u16, config.num_matrices));
}

pub fn StMatrix8x8x16bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpStMatrix("StMatrix8x8x16bOp", config.dtype, 16 * @as(u16, config.num_matrices));
}

pub fn StMatrix16x8x8bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.warpStMatrix("StMatrix16x8x8bOp", config.dtype, 8 * @as(u16, config.num_matrices));
}

pub fn Ld16x64bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Load("Ld16x64bOp", config.dtype, 64);
}
pub fn Ld16x128bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Load("Ld16x128bOp", config.dtype, 128);
}
pub fn Ld16x256bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Load("Ld16x256bOp", config.dtype, 256);
}
pub fn Ld16x32bx2Op(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Load("Ld16x32bx2Op", config.dtype, 64);
}
pub fn Ld32x32bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Load("Ld32x32bOp", config.dtype, 32);
}
pub fn St16x64bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Store("St16x64bOp", config.dtype, 64);
}
pub fn St16x128bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Store("St16x128bOp", config.dtype, 128);
}
pub fn St16x256bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Store("St16x256bOp", config.dtype, 256);
}
pub fn St16x32bx2Op(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Store("St16x32bx2Op", config.dtype, 64);
}
pub fn St32x32bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05Store("St32x32bOp", config.dtype, 32);
}
pub fn Cp128x256bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05S2T("Cp128x256bOp", config.dtype, 256);
}
pub fn Cp128x128bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05S2T("Cp128x128bOp", config.dtype, 128);
}
pub fn Cp4x256bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05S2T("Cp4x256bOp", config.dtype, 256);
}
pub fn Cp4x32x128bOp(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05S2T("Cp4x32x128bOp", config.dtype, 128);
}
pub fn Cp2x64x128b0213Op(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05S2T("Cp2x64x128b0213Op", config.dtype, 128);
}
pub fn Cp2x64x128b0123Op(config: CopyConfig) Error!atom.CopyAtom {
    return nvgpu.tcgen05S2T("Cp2x64x128b0123Op", config.dtype, 128);
}

pub fn countExactArchRecords() usize {
    return arch_exact.source_arch_record_count;
}

pub fn countConstructibleRecords() usize {
    var count: usize = 0;
    for (arch_exact.records) |record| {
        if (makeKind(record.name) != null) count += 1;
    }
    return count;
}

fn makeKind(name: []const u8) ?atom.OpKind {
    if (std.mem.indexOf(u8, name, "Mma") != null) return .mma;
    if (std.mem.indexOf(u8, name, "Copy") != null or startsWith(name, "Ld") or startsWith(name, "St") or startsWith(name, "Cp")) return .copy;
    return null;
}

test "arch_ops: source-derived constructors create copy and mma atoms" {
    const copy = try CopyAtomCpAsync(.{ .dtype = typing.Float32, .bits_per_copy = 32 });
    try std.testing.expectEqual(atom.OpKind.copy, copy.atom.kind());
    const mma = try MmaAtomSM90(.{ .a_dtype = typing.Float16, .acc_dtype = typing.Float32 });
    try std.testing.expectEqual(atom.OpKind.mma, mma.atom.kind());
}

test "arch_ops: exact class-name dispatch covers tcgen05 and warp matrix families" {
    const ld = try makeCopyBySourceName("Ld16x128bOp", .sm100, .{ .dtype = typing.Float32, .bits_per_copy = 128 });
    try std.testing.expectEqual(atom.OpKind.copy, ld.atom.kind());
    const st = try StMatrix8x8x16bOp(.{ .dtype = typing.Float16, .bits_per_copy = 16 });
    try std.testing.expectEqual(atom.OpKind.copy, st.atom.kind());
    const mx = try makeMmaBySourceName("MmaMXF4Op", .sm100, .{});
    try std.testing.expectEqual(atom.OpKind.mma, mx.atom.kind());
}

test "arch_ops: manifest is wired into constructibility audit" {
    try std.testing.expect(countExactArchRecords() >= 380);
    try std.testing.expect(countConstructibleRecords() >= 60);
}
