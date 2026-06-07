const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const atom = @import("atom.zig");

pub const Error = atom.Error || typing.Error || error{
    UnknownOperation,
    InvalidNumericType,
};

pub const OperandMajorMode = enum { MN, K };
pub const OutputMajorMode = enum { M, N };
pub const OperandSource = enum { smem, tmem, rmem, gmem };
pub const CtaGroup = enum { one, two };
pub const MemoryOrder = enum { weak, relaxed, acquire, release, acq_rel, sc, mmio, constant, @"volatile" };
pub const MemoryScope = enum { cta, cluster, gpu, sys };
pub const L2PrefetchSize = enum { none, reserved, size_64b, size_128b, size_256b };
pub const CacheEvictionPriority = enum { evict_normal, evict_first, evict_last, evict_unchanged, no_allocate };
pub const LoadCacheMode = enum { always, global, streaming, last_use, none };
pub const StoreCacheMode = enum { write_back, global, streaming, write_through, none };
pub const SharedSpace = enum { cta, cluster };

pub const MmaRuntimeFields = &.{ atom.RuntimeField.accumulate, .negate_a, .negate_b };
pub const BlockScaledRuntimeFields = &.{
    atom.RuntimeField.accumulate,
    .negate_a,
    .negate_b,
    .sfa,
    .sfb,
};
pub const ScaleFactorRuntimeFields = &.{ atom.RuntimeField.sfa, .sfb };
pub const CachePolicyRuntimeFields = &.{atom.RuntimeField.cache_policy};
pub const TmaRuntimeFields = &.{
    atom.RuntimeField.tma_barrier,
    .multicast_mask,
    .cache_policy,
};
pub const ByteMaskRuntimeFields = &.{atom.RuntimeField.byte_mask};

pub const SourceOpRecord = struct {
    module: []const u8,
    name: []const u8,
    kind: atom.OpKind,
    line: u32,
    abstract: bool = false,
};

pub const source_ops = [_]SourceOpRecord{
    .{ .module = "nvgpu.common", .name = "MmaUniversalOp", .kind = .mma, .line = 166 },
    .{
        .module = "nvgpu.common",
        .name = "CopyUniversalOp",
        .kind = .copy,
        .line = 361,
    },
    .{ .module = "nvgpu.common", .name = "CopyG2ROp", .kind = .copy, .line = 453 },
    .{ .module = "nvgpu.common", .name = "CopyR2GOp", .kind = .copy, .line = 539 },
    .{ .module = "nvgpu.common", .name = "CopyS2ROp", .kind = .copy, .line = 627 },
    .{ .module = "nvgpu.common", .name = "CopyR2SOp", .kind = .copy, .line = 681 },
    .{ .module = "nvgpu.cpasync.copy", .name = "CopyG2SOp", .kind = .copy, .line = 70 },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkTensorTileG2SOp",
        .kind = .copy,
        .line = 155,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkTensorIm2ColG2SOp",
        .kind = .copy,
        .line = 317,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkTensorIm2ColG2SMulticastOp",
        .kind = .copy,
        .line = 451,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkTensorIm2ColS2GOp",
        .kind = .copy,
        .line = 590,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkTensorTileG2SMulticastOp",
        .kind = .copy,
        .line = 680,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkTensorTileS2GOp",
        .kind = .copy,
        .line = 843,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyReduceBulkTensorTileS2GOp",
        .kind = .copy,
        .line = 952,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkG2SOp",
        .kind = .copy,
        .line = 1088,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkG2SMulticastOp",
        .kind = .copy,
        .line = 1178,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkS2GOp",
        .kind = .copy,
        .line = 1277,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkS2GByteMaskOp",
        .kind = .copy,
        .line = 1328,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyBulkS2SOp",
        .kind = .copy,
        .line = 1409,
    },
    .{
        .module = "nvgpu.cpasync.copy",
        .name = "CopyDsmemStoreOp",
        .kind = .copy,
        .line = 1498,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Ld16x64bOp",
        .kind = .copy,
        .line = 171,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Ld16x128bOp",
        .kind = .copy,
        .line = 220,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Ld16x256bOp",
        .kind = .copy,
        .line = 283,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Ld16x32bx2Op",
        .kind = .copy,
        .line = 346,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Ld32x32bOp",
        .kind = .copy,
        .line = 391,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "LdRed16x32bx2Op",
        .kind = .copy,
        .line = 436,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "LdRed32x32bOp",
        .kind = .copy,
        .line = 487,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "St16x64bOp",
        .kind = .copy,
        .line = 592,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "St16x128bOp",
        .kind = .copy,
        .line = 637,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "St16x256bOp",
        .kind = .copy,
        .line = 677,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "St16x32bx2Op",
        .kind = .copy,
        .line = 717,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "St32x32bOp",
        .kind = .copy,
        .line = 748,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Cp128x256bOp",
        .kind = .copy,
        .line = 821,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Cp128x128bOp",
        .kind = .copy,
        .line = 882,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Cp4x256bOp",
        .kind = .copy,
        .line = 929,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Cp4x32x128bOp",
        .kind = .copy,
        .line = 976,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Cp2x64x128b0213Op",
        .kind = .copy,
        .line = 1023,
    },
    .{
        .module = "nvgpu.tcgen05.copy",
        .name = "Cp2x64x128b0123Op",
        .kind = .copy,
        .line = 1070,
    },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "Tcgen05MmaOp",
        .kind = .mma,
        .line = 68,
        .abstract = true,
    },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "MmaOp",
        .kind = .mma,
        .line = 176,
        .abstract = true,
    },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "BlockScaledMmaOp",
        .kind = .mma,
        .line = 375,
        .abstract = true,
    },
    .{ .module = "nvgpu.tcgen05.mma", .name = "MmaTF32Op", .kind = .mma, .line = 609 },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "MmaF16BF16Op",
        .kind = .mma,
        .line = 709,
    },
    .{ .module = "nvgpu.tcgen05.mma", .name = "MmaI8Op", .kind = .mma, .line = 824 },
    .{ .module = "nvgpu.tcgen05.mma", .name = "MmaFP8Op", .kind = .mma, .line = 933 },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "MmaF8F6F4Op",
        .kind = .mma,
        .line = 1048,
    },
    .{ .module = "nvgpu.tcgen05.mma", .name = "MmaMXF8Op", .kind = .mma, .line = 1171 },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "MmaMXF8F6F4Op",
        .kind = .mma,
        .line = 1286,
    },
    .{ .module = "nvgpu.tcgen05.mma", .name = "MmaMXF4Op", .kind = .mma, .line = 1411 },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "MmaMXF4NVF4Op",
        .kind = .mma,
        .line = 1518,
    },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "SM103MmaMXF4Op",
        .kind = .mma,
        .line = 1632,
    },
    .{
        .module = "nvgpu.tcgen05.mma",
        .name = "SM103MmaMXF4NVF4Op",
        .kind = .mma,
        .line = 1737,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "BaseOp",
        .kind = .copy,
        .line = 25,
        .abstract = true,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "LdMatrix8x8x16bOp",
        .kind = .copy,
        .line = 56,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "LdMatrix8x16x8bOp",
        .kind = .copy,
        .line = 100,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "LdMatrix16x8x8bOp",
        .kind = .copy,
        .line = 156,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "LdMatrix16x16x8bOp",
        .kind = .copy,
        .line = 209,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "StMatrix8x8x16bOp",
        .kind = .copy,
        .line = 262,
    },
    .{
        .module = "nvgpu.warp.copy",
        .name = "StMatrix16x8x8bOp",
        .kind = .copy,
        .line = 305,
    },
    .{
        .module = "nvgpu.warp.mma",
        .name = "WarpMmaOp",
        .kind = .mma,
        .line = 48,
        .abstract = true,
    },
    .{ .module = "nvgpu.warp.mma", .name = "MmaF16BF16Op", .kind = .mma, .line = 57 },
    .{ .module = "nvgpu.warp.mma", .name = "MmaFP8Op", .kind = .mma, .line = 139 },
    .{
        .module = "nvgpu.warp.mma",
        .name = "MmaSM120BlockScaledOp",
        .kind = .mma,
        .line = 217,
        .abstract = true,
    },
    .{ .module = "nvgpu.warp.mma", .name = "MmaMXF4Op", .kind = .mma, .line = 388 },
    .{ .module = "nvgpu.warp.mma", .name = "MmaMXF4NVF4Op", .kind = .mma, .line = 445 },
    .{ .module = "nvgpu.warp.mma", .name = "MmaMXF8Op", .kind = .mma, .line = 502 },
    .{ .module = "nvgpu.warp.mma", .name = "MmaMXF8F6F4Op", .kind = .mma, .line = 569 },
    .{
        .module = "nvgpu.warpgroup.mma",
        .name = "WarpGroupMmaOp",
        .kind = .mma,
        .line = 53,
        .abstract = true,
    },
    .{
        .module = "nvgpu.warpgroup.mma",
        .name = "MmaOp",
        .kind = .mma,
        .line = 142,
        .abstract = true,
    },
    .{
        .module = "nvgpu.warpgroup.mma",
        .name = "MmaF16BF16Op",
        .kind = .mma,
        .line = 319,
    },
    .{ .module = "nvgpu.warpgroup.mma", .name = "MmaF8Op", .kind = .mma, .line = 408 },
    .{ .module = "nvgpu.warpgroup.mma", .name = "MmaI8Op", .kind = .mma, .line = 497 },
};

pub fn findSourceOp(module: []const u8, name: []const u8) ?SourceOpRecord {
    for (source_ops) |r| {
        if (std.mem.eql(u8, r.module, module) and std.mem.eql(u8, r.name, name))
            return r;
    }
    return null;
}

pub fn hasSourceOp(module: []const u8, name: []const u8) bool {
    return findSourceOp(module, name) != null;
}

pub const CopyOptions = struct {
    num_bits_per_copy: u16 = 32,
    memory_order: MemoryOrder = .weak,
    memory_scope: MemoryScope = .cta,
    l2_prefetch_size: L2PrefetchSize = .none,
    l1c_evict_priority: CacheEvictionPriority = .evict_normal,
    load_cache_mode: LoadCacheMode = .always,
    store_cache_mode: StoreCacheMode = .write_back,
    shared_space: SharedSpace = .cta,
    invariant: bool = false,
};

pub const MmaOptions = struct {
    instruction_shape_mnk: layout.Tree = layout.Tree.fromComptime(.{ 16, 8, 8 }),
    a_src: OperandSource = .smem,
    a_major_mode: OperandMajorMode = .K,
    b_major_mode: OperandMajorMode = .K,
    cta_group: CtaGroup = .one,
    output_major_mode: OutputMajorMode = .N,
};

pub fn universalMma(abacc_dtype: typing.Numeric) Error!atom.MmaAtom {
    if (!(std.mem.eql(u8, abacc_dtype.name, typing.Float16.name) or std.mem.eql(u8, abacc_dtype.name, typing.Float32.name) or std.mem.eql(u8, abacc_dtype.name, typing.Float64.name)))
        return Error.InvalidNumericType;
    const shape = layout.Tree.fromComptime(.{ 1, 1, 1 });
    const desc = atom.OpDescriptor.mmaTyped(
        "MmaUniversalOp",
        "generic",
        "universal_fma",
        shape,
        abacc_dtype,
        abacc_dtype,
        abacc_dtype,
        &.{},
    );
    return atom.makeMmaAtom(
        desc,
        try defaultMmaTrait("MmaUniversalTrait", shape, &.{}),
    );
}

pub fn warpMmaF16BF16(
    ab_dtype: typing.Numeric,
    acc_dtype: typing.Numeric,
    opts: MmaOptions,
) Error!atom.MmaAtom {
    if (!(std.mem.eql(u8, ab_dtype.name, typing.Float16.name) or std.mem.eql(u8, ab_dtype.name, typing.BFloat16.name)))
        return Error.InvalidNumericType;
    const desc = atom.OpDescriptor.mmaTyped(
        "MmaF16BF16Op",
        "sm80+",
        "warp.mma",
        opts.instruction_shape_mnk,
        ab_dtype,
        ab_dtype,
        acc_dtype,
        &.{},
    );
    return atom.makeMmaAtom(
        desc,
        try defaultMmaTrait("MmaF16BF16Trait", opts.instruction_shape_mnk, &.{}),
    );
}

pub fn warpMmaFP8(
    ab_dtype: typing.Numeric,
    acc_dtype: typing.Numeric,
    opts: MmaOptions,
) Error!atom.MmaAtom {
    if (ab_dtype.width != 8 or !ab_dtype.isFloat()) return Error.InvalidNumericType;
    const desc = atom.OpDescriptor.mmaTyped(
        "MmaFP8Op",
        "sm89+",
        "warp.mma",
        opts.instruction_shape_mnk,
        ab_dtype,
        ab_dtype,
        acc_dtype,
        &.{},
    );
    return atom.makeMmaAtom(
        desc,
        try defaultMmaTrait("MmaFP8Trait", opts.instruction_shape_mnk, &.{}),
    );
}

pub fn warpgroupMmaF16BF16(
    ab_dtype: typing.Numeric,
    acc_dtype: typing.Numeric,
    opts: MmaOptions,
) Error!atom.MmaAtom {
    if (!(std.mem.eql(u8, ab_dtype.name, typing.Float16.name) or std.mem.eql(u8, ab_dtype.name, typing.BFloat16.name)))
        return Error.InvalidNumericType;
    const desc = atom.OpDescriptor.mmaTyped(
        "MmaF16BF16Op",
        "sm90+",
        "warpgroup.mma",
        opts.instruction_shape_mnk,
        ab_dtype,
        ab_dtype,
        acc_dtype,
        MmaRuntimeFields,
    );
    return atom.makeMmaAtom(
        desc,
        try defaultMmaTrait(
            "WarpGroupMmaF16BF16Trait",
            opts.instruction_shape_mnk,
            MmaRuntimeFields,
        ),
    );
}

pub fn tcgen05MmaF16BF16(
    ab_dtype: typing.Numeric,
    acc_dtype: typing.Numeric,
    opts: MmaOptions,
) Error!atom.MmaAtom {
    if (!(std.mem.eql(u8, ab_dtype.name, typing.Float16.name) or std.mem.eql(u8, ab_dtype.name, typing.BFloat16.name)))
        return Error.InvalidNumericType;
    const desc = atom.OpDescriptor.mmaTyped(
        "MmaF16BF16Op",
        "sm100+",
        "tcgen05.mma",
        opts.instruction_shape_mnk,
        ab_dtype,
        ab_dtype,
        acc_dtype,
        MmaRuntimeFields,
    );
    return atom.makeMmaAtom(
        desc,
        try defaultMmaTrait(
            "Tcgen05MmaF16BF16Trait",
            opts.instruction_shape_mnk,
            MmaRuntimeFields,
        ),
    );
}

pub fn tcgen05BlockScaledMma(
    name: []const u8,
    a_dtype: typing.Numeric,
    b_dtype: typing.Numeric,
    sf_dtype: typing.Numeric,
    opts: MmaOptions,
) Error!atom.MmaAtom {
    _ = sf_dtype;
    const desc = atom.OpDescriptor.mmaTyped(
        name,
        "sm100+",
        "tcgen05.block_scaled_mma",
        opts.instruction_shape_mnk,
        a_dtype,
        b_dtype,
        typing.Float32,
        BlockScaledRuntimeFields,
    );
    return atom.makeMmaAtom(
        desc,
        try defaultMmaTrait(
            "Tcgen05BlockScaledMmaTrait",
            opts.instruction_shape_mnk,
            BlockScaledRuntimeFields,
        ),
    );
}

pub fn copyUniversal(dtype: typing.Numeric, opts: CopyOptions) Error!atom.CopyAtom {
    return makeCopy(
        "CopyUniversalOp",
        "generic",
        "simt.sync.copy",
        dtype,
        .generic,
        .generic,
        opts.num_bits_per_copy,
        &.{},
    );
}

pub fn copyG2R(dtype: typing.Numeric, opts: CopyOptions) Error!atom.CopyAtom {
    return makeCopy(
        "CopyG2ROp",
        "generic",
        "g2r",
        dtype,
        .gmem,
        .generic,
        opts.num_bits_per_copy,
        CachePolicyRuntimeFields,
    );
}

pub fn copyR2G(dtype: typing.Numeric, opts: CopyOptions) Error!atom.CopyAtom {
    return makeCopy(
        "CopyR2GOp",
        "generic",
        "r2g",
        dtype,
        .generic,
        .gmem,
        opts.num_bits_per_copy,
        CachePolicyRuntimeFields,
    );
}

pub fn copyS2R(dtype: typing.Numeric, opts: CopyOptions) Error!atom.CopyAtom {
    return makeCopy(
        "CopyS2ROp",
        "generic",
        "s2r",
        dtype,
        .smem,
        .generic,
        opts.num_bits_per_copy,
        &.{},
    );
}

pub fn copyR2S(dtype: typing.Numeric, opts: CopyOptions) Error!atom.CopyAtom {
    return makeCopy(
        "CopyR2SOp",
        "generic",
        "r2s",
        dtype,
        .generic,
        .smem,
        opts.num_bits_per_copy,
        &.{},
    );
}

pub fn cpAsyncG2S(dtype: typing.Numeric, opts: CopyOptions) Error!atom.CopyAtom {
    return makeCopy(
        "CopyG2SOp",
        "sm80+",
        "cpasync.g2s",
        dtype,
        .gmem,
        .smem,
        opts.num_bits_per_copy,
        &.{},
    );
}

pub fn tmaCopy(
    name: []const u8,
    dtype: typing.Numeric,
    source: typing.AddressSpace,
    destination: typing.AddressSpace,
    opts: CopyOptions,
) Error!atom.CopyAtom {
    return makeCopy(
        name,
        "sm90+",
        "tma",
        dtype,
        source,
        destination,
        opts.num_bits_per_copy,
        TmaRuntimeFields,
    );
}

pub fn warpLdMatrix(
    name: []const u8,
    dtype: typing.Numeric,
    bits_per_copy: u16,
) Error!atom.CopyAtom {
    return makeCopy(
        name,
        "sm75+",
        "warp.ldmatrix",
        dtype,
        .smem,
        .generic,
        bits_per_copy,
        &.{},
    );
}

pub fn warpStMatrix(
    name: []const u8,
    dtype: typing.Numeric,
    bits_per_copy: u16,
) Error!atom.CopyAtom {
    return makeCopy(
        name,
        "sm90+",
        "warp.stmatrix",
        dtype,
        .generic,
        .smem,
        bits_per_copy,
        &.{},
    );
}

pub fn tcgen05Load(
    name: []const u8,
    dtype: typing.Numeric,
    bits_per_copy: u16,
) Error!atom.CopyAtom {
    return makeCopy(
        name,
        "sm100+",
        "tcgen05.ld",
        dtype,
        .tmem,
        .generic,
        bits_per_copy,
        &.{},
    );
}

pub fn tcgen05Store(
    name: []const u8,
    dtype: typing.Numeric,
    bits_per_copy: u16,
) Error!atom.CopyAtom {
    return makeCopy(
        name,
        "sm100+",
        "tcgen05.st",
        dtype,
        .generic,
        .tmem,
        bits_per_copy,
        &.{},
    );
}

pub fn tcgen05S2T(
    name: []const u8,
    dtype: typing.Numeric,
    bits_per_copy: u16,
) Error!atom.CopyAtom {
    return makeCopy(
        name,
        "sm100+",
        "tcgen05.cp",
        dtype,
        .smem,
        .tmem,
        bits_per_copy,
        &.{},
    );
}

fn makeCopy(
    name: []const u8,
    arch: []const u8,
    family: []const u8,
    dtype: typing.Numeric,
    source: typing.AddressSpace,
    dest: typing.AddressSpace,
    bits: u16,
    fields: []const atom.RuntimeField,
) Error!atom.CopyAtom {
    if (bits == 0) return Error.InvalidCopyBits;
    const desc = atom.OpDescriptor.copyTyped(
        name,
        arch,
        family,
        dtype,
        source,
        dest,
        bits,
        fields,
    );
    return atom.makeCopyAtom(desc, try defaultCopyTrait(name, dtype, bits, fields));
}

fn defaultMmaTrait(
    name: []const u8,
    shape: layout.Tree,
    fields: []const atom.RuntimeField,
) Error!atom.Trait {
    const thr = layout.makeCompactLayout(.{32});
    const tv = layout.makeCompactLayout(.{ 32, 1 });
    return atom.Trait{
        .name = name,
        .thr_id = thr,
        .shape_mnk = shape,
        .tv_layout_a = tv,
        .tv_layout_b = tv,
        .tv_layout_c = tv,
        .admissible_fields = fields,
        .type_name = "!cute_nvgpu.mma_trait",
    };
}

fn defaultCopyTrait(
    name: []const u8,
    dtype: typing.Numeric,
    bits_per_copy: u16,
    fields: []const atom.RuntimeField,
) Error!atom.Trait {
    const values = @max(@as(u16, 1), bits_per_copy / @max(@as(u16, 1), dtype.width));
    const tv_shape = try layout.Tree.initTuple(&.{
        try layout.Tree.initLeaf(32),
        try layout.Tree.initLeaf(@intCast(values)),
    });
    const tv = try layout.Layout.makeCompact(tv_shape);
    return atom.Trait{
        .name = name,
        .thr_id = layout.makeCompactLayout(.{32}),
        .layout_src_tv = tv,
        .layout_dst_tv = tv,
        .admissible_fields = fields,
        .type_name = "!cute_nvgpu.copy_trait",
    };
}

test "nvgpu: source-grounded op catalog covers concrete and abstract classes" {
    try std.testing.expect(source_ops.len >= 70);
    try std.testing.expect(hasSourceOp("nvgpu.warpgroup.mma", "MmaF16BF16Op"));
    const rec = findSourceOp(
        "nvgpu.tcgen05.mma",
        "BlockScaledMmaOp",
    ) orelse return error.TestExpectedEqual;
    try std.testing.expect(rec.abstract);
}

test "nvgpu: universal, warpgroup, tcgen05 mma constructors produce runtime-state traits" {
    const uni = try universalMma(typing.Float32);
    try std.testing.expectEqual(atom.OpKind.mma, uni.atom.kind());

    var wg = try warpgroupMmaF16BF16(typing.Float16, typing.Float32, .{});
    try wg.set(.accumulate, .{ .bool = true });
    try std.testing.expect((try wg.get(.accumulate)).eql(.{ .bool = true }));
    try std.testing.expectError(
        atom.Error.UnsupportedField,
        wg.set(.sfa, .{ .symbol = "%scale" }),
    );

    var tc = try tcgen05BlockScaledMma(
        "MmaMXF8Op",
        typing.Float8E4M3,
        typing.Float8E4M3,
        typing.Float8E8M0FNU,
        .{},
    );
    try tc.set(.sfa, .{ .symbol = "%sfa" });
    try std.testing.expect((try tc.get(.sfa)).eql(.{ .symbol = "%sfa" }));
}

test "nvgpu: copy constructors map source/destination spaces and runtime fields" {
    var g2r = try copyG2R(typing.Int32, .{ .num_bits_per_copy = 64 });
    try std.testing.expectEqual(typing.AddressSpace.gmem, g2r.atom.op.source_space.?);
    try std.testing.expectEqual(@as(u16, 64), g2r.atom.op.num_bits_per_copy.?);
    try g2r.set(.cache_policy, .{ .i64 = 1 });
    try std.testing.expect((try g2r.get(.cache_policy)).eql(.{ .i64 = 1 }));

    const ld = try warpLdMatrix("LdMatrix8x8x16bOp", typing.Float16, 128);
    try std.testing.expectEqual(typing.AddressSpace.smem, ld.atom.op.source_space.?);
    const tma = try tmaCopy(
        "CopyBulkTensorTileG2SOp",
        typing.Float16,
        .gmem,
        .smem,
        .{ .num_bits_per_copy = 128 },
    );
    try std.testing.expectEqualStrings("tma", tma.atom.op.family);
}
