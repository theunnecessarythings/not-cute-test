const atom = @import("atom.zig");
const layout = @import("layout.zig");
const mlir = @import("mlir.zig");
const nvgpu = @import("nvgpu.zig");
const runtime = @import("runtime.zig");
const std = @import("std");
const typing = @import("typing.zig");

pub const Error = mlir.Error || runtime.Error || nvgpu.Error || atom.Error || typing.Error || error{
    InvalidArchOperation,
    InvalidBarrierPhase,
    InvalidFieldCombination,
    InvalidInstructionConfig,
    InvalidMemorySpace,
    InvalidNvvmWrapper,
    InvalidOperandCount,
    InvalidShape,
    UnknownArchOperation,
    UnsupportedArch,
    UnsupportedArchitecture,
    UnsupportedInstructionShape,
    UnsupportedOperandType,
};

pub const SmemAllocation = struct {
    bytes: usize,
    alignment: usize = 16,
    dtype: typing.Numeric = typing.Uint8,

    pub fn init(bytes: usize, alignment: usize, dtype: typing.Numeric) Error!SmemAllocation {
        if (bytes == 0 or alignment == 0) return Error.InvalidMemorySpace;
        return .{ .bytes = bytes, .alignment = alignment, .dtype = dtype };
    }

    pub fn emitAlloc(self: SmemAllocation, builder: anytype) Error!mlir.Value {
        _ = self;
        return builder.genericOp(
            "cute.arch.alloc_smem",
            &.{},
            &.{ .{ .key = "bytes", .value = "dynamic" }, .{ .key = "alignment", .value = "dynamic" } },
            &.{},
            &.{mlir.Type.raw("!cute.ptr")},
        );
    }
};

pub const TmemAllocation = struct {
    columns: u16,
    alignment: u16 = 32,

    pub fn init(columns: u16, alignment: u16) Error!TmemAllocation {
        if (columns == 0 or alignment == 0) return Error.InvalidMemorySpace;
        return .{ .columns = columns, .alignment = alignment };
    }
};

pub const Barrier = struct {
    address: mlir.Operand,
    ty: mlir.Type = mlir.Type.raw("!cute.ptr"),

    pub fn emitInit(self: Barrier, builder: anytype, thread_count: mlir.Operand) Error!void {
        try builder.operationNoResult(.{
            .name = "cute.arch.mbarrier_init",
            .operands = &.{ self.address, thread_count },
            .operand_types = &.{ self.ty, mlir.Type.i(32) },
            .result_types = &.{},
        });
    }

    pub fn emitArrive(self: Barrier, builder: anytype) Error!mlir.Value {
        return builder.genericOp(
            "cute.arch.mbarrier_arrive",
            &.{self.address},
            &.{},
            &.{self.ty},
            &.{mlir.Type.i(64)},
        );
    }

    pub fn emitWait(self: Barrier, builder: anytype, phase: mlir.Operand) Error!void {
        try builder.operationNoResult(.{
            .name = "cute.arch.mbarrier_wait",
            .operands = &.{ self.address, phase },
            .operand_types = &.{ self.ty, mlir.Type.i(32) },
            .result_types = &.{},
        });
    }

    pub fn emitExpectTx(self: Barrier, builder: anytype, bytes: mlir.Operand) Error!void {
        try builder.operationNoResult(.{
            .name = "cute.arch.mbarrier_expect_tx",
            .operands = &.{ self.address, bytes },
            .operand_types = &.{ self.ty, mlir.Type.i(32) },
            .result_types = &.{},
        });
    }
};

pub const ElectOne = struct {
    pub fn emit(builder: anytype) Error!mlir.Value {
        return builder.genericOp("cute.arch.elect_one", &.{}, &.{}, &.{}, &.{mlir.Type.i(1)});
    }
};

pub const ClusterQuery = enum { cta_rank, cluster_shape_x, cluster_shape_y, cluster_shape_z };

pub fn issueClcQuery(builder: anytype, query: ClusterQuery) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.issue_clc_query",
        &.{},
        &.{.{ .key = "query", .value = clcName(query) }},
        &.{},
        &.{mlir.Type.i(32)},
    );
}

pub fn clcResponse(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.arch.clc_response", &.{}, &.{}, &.{}, &.{mlir.Type.i(32)});
}

pub fn getDynSmem(builder: anytype) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.get_dyn_smem",
        &.{},
        &.{},
        &.{},
        &.{mlir.Type.raw("!cute.ptr")},
    );
}

pub fn getDynSmemSize(builder: anytype) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.get_dyn_smem_size",
        &.{},
        &.{},
        &.{},
        &.{mlir.Type.i(32)},
    );
}

pub fn mapDsmemPtr(builder: anytype, ptr: mlir.Operand) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.map_dsmem_ptr",
        &.{ptr},
        &.{},
        &.{mlir.Type.raw("!cute.ptr")},
        &.{mlir.Type.raw("!cute.ptr")},
    );
}

pub fn getMaxTmemAllocCols(builder: anytype) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.get_max_tmem_alloc_cols",
        &.{},
        &.{},
        &.{},
        &.{mlir.Type.i(32)},
    );
}

pub fn getMinTmemAllocCols(builder: anytype) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.get_min_tmem_alloc_cols",
        &.{},
        &.{},
        &.{},
        &.{mlir.Type.i(32)},
    );
}

pub fn allocTmem(builder: anytype, columns: mlir.Operand) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.alloc_tmem",
        &.{columns},
        &.{},
        &.{mlir.Type.i(32)},
        &.{mlir.Type.raw("!cute.ptr")},
    );
}

pub fn retrieveTmemPtr(builder: anytype, handle: mlir.Operand) Error!mlir.Value {
    return builder.genericOp(
        "cute.arch.retrieve_tmem_ptr",
        &.{handle},
        &.{},
        &.{mlir.Type.i(32)},
        &.{mlir.Type.raw("!cute.ptr")},
    );
}

pub fn deallocTmem(builder: anytype, ptr: mlir.Operand) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.arch.dealloc_tmem",
        .operands = &.{ptr},
        .operand_types = &.{mlir.Type.raw("!cute.ptr")},
        .result_types = &.{},
    });
}

pub fn relinquishTmemAllocPermit(builder: anytype) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.arch.relinquish_tmem_alloc_permit",
        .operands = &.{},
        .operand_types = &.{},
        .result_types = &.{},
    });
}

pub const NumericConversion = enum {
    cvtI8Bf16,
    cvtI4Bf16,
    sext_unpacked_i4_i8,

    pub fn opName(self: NumericConversion) []const u8 {
        return switch (self) {
            .cvtI8Bf16 => "cute.arch.cvt_i8_bf16_intrinsic",
            .cvtI4Bf16 => "cute.arch.cvt_i4_bf16_intrinsic",
            .sext_unpacked_i4_i8 => "cute.arch.sext_unpacked_i4_i8_intrinsic",
        };
    }
};

pub fn numericConvert(
    builder: anytype,
    conversion: NumericConversion,
    value: mlir.Operand,
    input_type: mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return builder.genericOp(
        conversion.opName(),
        &.{value},
        &.{},
        &.{input_type},
        &.{result_type},
    );
}

pub fn inlineAsm(
    builder: anytype,
    asm_text: []const u8,
    constraints: []const u8,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_types: []const mlir.Type,
) Error!mlir.ValueRange {
    return mlir.llvm.inlineAsm(
        builder,
        asm_text,
        constraints,
        operands,
        operand_types,
        result_types,
        true,
    );
}

fn clcName(q: ClusterQuery) []const u8 {
    return switch (q) {
        .cta_rank => "cta_rank",
        .cluster_shape_x => "cluster_shape_x",
        .cluster_shape_y => "cluster_shape_y",
        .cluster_shape_z => "cluster_shape_z",
    };
}

test "arch: emits barrier, smem, tmem, elect, and conversion ops" {
    var b: mlir.Builder(4096) = .{};
    _ = try getDynSmem(&b);
    const barrier: Barrier = .{ .address = .arg(0) };
    try barrier.emitInit(&b, .arg(1));
    _ = try barrier.emitArrive(&b);
    _ = try ElectOne.emit(&b);
    _ = try allocTmem(&b, .arg(2));
    _ = try numericConvert(&b, .cvtI8Bf16, .arg(3), mlir.Type.i(32), mlir.Type.bf16());
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "mbarrier_init") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "alloc_tmem") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cvt_i8_bf16") != null);
}

pub const SmArch = enum(u16) {
    sm70 = 70,
    sm75 = 75,
    sm80 = 80,
    sm89 = 89,
    sm90 = 90,
    sm100 = 100,
    sm103 = 103,
    sm120 = 120,

    pub fn cc(self: SmArch) u16 {
        return @intFromEnum(self);
    }

    pub fn atLeast(self: SmArch, min: SmArch) bool {
        return self.cc() >= min.cc();
    }

    pub fn mlirChip(self: SmArch) []const u8 {
        return switch (self) {
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
};

pub const ArchClass = enum { generic, warp, cpasync, warpgroup, tcgen05, sm120_warp };
pub const OpFamily = enum {
    universal,
    gmem_rmem,
    rmem_gmem,
    smem_rmem,
    rmem_smem,
    cpasync,
    tma,
    tmem_ldst,
    smem_tmem,
    warp_mma,
    warpgroup_mma,
    tcgen05_mma,
    block_scaled_mma,
};
pub const OperandMajorMode = nvgpu.OperandMajorMode;
pub const OperandSource = nvgpu.OperandSource;
pub const CtaGroup = nvgpu.CtaGroup;
pub const OutputMajorMode = nvgpu.OutputMajorMode;

pub const DTypeClass = enum {
    any,
    int,
    signed_int,
    unsigned_int,
    float,
    f16_bf16,
    fp8,
    f4_f6_f8,
    mxf4,
    mxf8,
    tf32,
};

pub const OpSpec = struct {
    source_module: []const u8,
    source_name: []const u8,
    kind: atom.OpKind,
    arch_class: ArchClass,
    family: OpFamily,
    min_arch: SmArch,
    concrete: bool = true,
    dtype_class: DTypeClass = .any,
    value_bits: ?u16 = null,
    source_space: ?typing.AddressSpace = null,
    destination_space: ?typing.AddressSpace = null,
    allowed_fields: []const atom.RuntimeField = &.{},

    pub fn sourceRecord(self: OpSpec) ?nvgpu.SourceOpRecord {
        return nvgpu.findSourceOp(self.source_module, self.source_name);
    }

    pub fn supportsArch(self: OpSpec, arch: SmArch) bool {
        return arch.atLeast(self.min_arch);
    }

    pub fn validateDType(self: OpSpec, dtype: typing.Numeric) Error!void {
        if (!dtypeMatches(self.dtype_class, dtype)) return Error.UnsupportedOperandType;
    }

    pub fn ensureSourceGrounded(self: OpSpec) Error!void {
        const rec = self.sourceRecord() orelse return Error.UnknownArchOperation;
        if (rec.kind != self.kind) return Error.UnknownArchOperation;
        if (self.concrete and rec.abstract) return Error.UnknownArchOperation;
    }

    pub fn makeCopyAtom(self: OpSpec, arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
        try self.ensureSourceGrounded();
        if (self.kind != .copy) return Error.InvalidInstructionConfig;
        if (!self.supportsArch(arch)) return Error.UnsupportedArchitecture;
        try self.validateDType(dtype);
        return atom.makeCopyAtom(
            .{
                .name = self.source_name,
                .kind = .copy,
                .arch = arch.mlirChip(),
                .family = familyName(self.family),
                .value_type = dtype,
                .num_bits_per_copy = self.value_bits orelse dtype.width,
                .source_space = self.source_space orelse .generic,
                .destination_space = self.destination_space orelse .generic,
                .allowed_fields = self.allowed_fields,
            },
            defaultCopyTrait(
                self.source_name,
                dtype,
                self.value_bits orelse dtype.width,
                self.allowed_fields,
            ) catch |err| return err,
        );
    }

    pub fn makeMmaAtom(
        self: OpSpec,
        arch: SmArch,
        a: typing.Numeric,
        b: typing.Numeric,
        acc: typing.Numeric,
        shape_mnk: layout.Tree,
    ) Error!atom.MmaAtom {
        try self.ensureSourceGrounded();
        if (self.kind != .mma) return Error.InvalidInstructionConfig;
        if (!self.supportsArch(arch)) return Error.UnsupportedArchitecture;
        try self.validateDType(a);
        try self.validateDType(b);
        try validateMmaShape(self.family, shape_mnk);
        return atom.makeMmaAtom(
            .{
                .name = self.source_name,
                .kind = .mma,
                .arch = arch.mlirChip(),
                .family = familyName(self.family),
                .instruction_shape_mnk = shape_mnk,
                .a_type = a,
                .b_type = b,
                .c_type = acc,
                .allowed_fields = self.allowed_fields,
            },
            defaultMmaTrait(
                self.source_name,
                shape_mnk,
                self.allowed_fields,
            ) catch |err| return err,
        );
    }
};

pub const TmaDescriptor = struct {
    op_name: []const u8,
    cta_group: CtaGroup = .one,
    multicast: bool = false,
    im2col: bool = false,
    source: typing.AddressSpace = .gmem,
    destination: typing.AddressSpace = .smem,
    cache_policy_field: bool = true,

    pub fn spec(self: TmaDescriptor) Error!OpSpec {
        const s = findSpec("nvgpu.cpasync.copy", self.op_name) orelse return Error.UnknownArchOperation;
        return s;
    }
};

pub const SmemLayoutAtomKind = enum {
    MN_INTER,
    MN_SW32,
    MN_SW64,
    MN_SW128,
    MN_SW128_32B,
    K_INTER,
    K_SW32,
    K_SW64,
    K_SW128,

    pub fn irName(self: SmemLayoutAtomKind) []const u8 {
        return switch (self) {
            .MN_INTER => "mn_inter",
            .MN_SW32 => "mn_sw32",
            .MN_SW64 => "mn_sw64",
            .MN_SW128 => "mn_sw128",
            .MN_SW128_32B => "mn_sw128_32b",
            .K_INTER => "k_inter",
            .K_SW32 => "k_sw32",
            .K_SW64 => "k_sw64",
            .K_SW128 => "k_sw128",
        };
    }
};

pub const TmemLoadReduction = enum { max, maxabs, min, minabs };
pub const Repetition = enum { x1, x2, x4, x8, x16, x32, x64, x128 };
pub const Pack = enum { none, pack_16b_in_32b };
pub const Unpack = enum { none, unpack_32b_in_16b };

pub const RuntimeFieldGroup = struct {
    name: []const u8,
    fields: []const atom.RuntimeField,
};

pub const runtime_field_groups = [_]RuntimeFieldGroup{
    .{ .name = "warpgroup_mma", .fields = nvgpu.MmaRuntimeFields },
    .{ .name = "tcgen05_mma", .fields = nvgpu.MmaRuntimeFields },
    .{ .name = "block_scaled", .fields = nvgpu.BlockScaledRuntimeFields },
    .{ .name = "scale_factor", .fields = nvgpu.ScaleFactorRuntimeFields },
    .{ .name = "cache_policy", .fields = nvgpu.CachePolicyRuntimeFields },
    .{ .name = "tma", .fields = nvgpu.TmaRuntimeFields },
    .{ .name = "byte_mask", .fields = nvgpu.ByteMaskRuntimeFields },
};

pub const op_specs = [_]OpSpec{
    .{
        .source_module = "nvgpu.common",
        .source_name = "MmaUniversalOp",
        .kind = .mma,
        .arch_class = .generic,
        .family = .universal,
        .min_arch = .sm70,
        .dtype_class = .float,
    },
    .{
        .source_module = "nvgpu.common",
        .source_name = "CopyUniversalOp",
        .kind = .copy,
        .arch_class = .generic,
        .family = .universal,
        .min_arch = .sm70,
    },
    .{
        .source_module = "nvgpu.common",
        .source_name = "CopyG2ROp",
        .kind = .copy,
        .arch_class = .generic,
        .family = .gmem_rmem,
        .min_arch = .sm70,
        .source_space = .gmem,
        .destination_space = .generic,
        .allowed_fields = nvgpu.CachePolicyRuntimeFields,
    },
    .{
        .source_module = "nvgpu.common",
        .source_name = "CopyR2GOp",
        .kind = .copy,
        .arch_class = .generic,
        .family = .rmem_gmem,
        .min_arch = .sm70,
        .source_space = .generic,
        .destination_space = .gmem,
        .allowed_fields = nvgpu.CachePolicyRuntimeFields,
    },
    .{
        .source_module = "nvgpu.common",
        .source_name = "CopyS2ROp",
        .kind = .copy,
        .arch_class = .generic,
        .family = .smem_rmem,
        .min_arch = .sm70,
        .source_space = .smem,
        .destination_space = .generic,
    },
    .{
        .source_module = "nvgpu.common",
        .source_name = "CopyR2SOp",
        .kind = .copy,
        .arch_class = .generic,
        .family = .rmem_smem,
        .min_arch = .sm70,
        .source_space = .generic,
        .destination_space = .smem,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyG2SOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm80,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkTensorTileG2SOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkTensorIm2ColG2SOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkTensorIm2ColG2SMulticastOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkTensorIm2ColS2GOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .gmem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkTensorTileG2SMulticastOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkTensorTileS2GOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .gmem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyReduceBulkTensorTileS2GOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .tma,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .gmem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkG2SOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm90,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.CachePolicyRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkG2SMulticastOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm90,
        .source_space = .gmem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkS2GOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .gmem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkS2GByteMaskOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .gmem,
        .value_bits = 128,
        .allowed_fields = nvgpu.ByteMaskRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyBulkS2SOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.cpasync.copy",
        .source_name = "CopyDsmemStoreOp",
        .kind = .copy,
        .arch_class = .cpasync,
        .family = .cpasync,
        .min_arch = .sm90,
        .source_space = .smem,
        .destination_space = .smem,
        .value_bits = 128,
        .allowed_fields = nvgpu.TmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warp.copy",
        .source_name = "LdMatrix8x8x16bOp",
        .kind = .copy,
        .arch_class = .warp,
        .family = .smem_rmem,
        .min_arch = .sm75,
        .source_space = .smem,
        .destination_space = .generic,
        .value_bits = 128,
        .dtype_class = .f16_bf16,
    },
    .{
        .source_module = "nvgpu.warp.copy",
        .source_name = "LdMatrix8x16x8bOp",
        .kind = .copy,
        .arch_class = .warp,
        .family = .smem_rmem,
        .min_arch = .sm75,
        .source_space = .smem,
        .destination_space = .generic,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.warp.copy",
        .source_name = "LdMatrix16x8x8bOp",
        .kind = .copy,
        .arch_class = .warp,
        .family = .smem_rmem,
        .min_arch = .sm75,
        .source_space = .smem,
        .destination_space = .generic,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.warp.copy",
        .source_name = "LdMatrix16x16x8bOp",
        .kind = .copy,
        .arch_class = .warp,
        .family = .smem_rmem,
        .min_arch = .sm75,
        .source_space = .smem,
        .destination_space = .generic,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.warp.copy",
        .source_name = "StMatrix8x8x16bOp",
        .kind = .copy,
        .arch_class = .warp,
        .family = .rmem_smem,
        .min_arch = .sm90,
        .source_space = .generic,
        .destination_space = .smem,
        .value_bits = 128,
        .dtype_class = .f16_bf16,
    },
    .{
        .source_module = "nvgpu.warp.copy",
        .source_name = "StMatrix16x8x8bOp",
        .kind = .copy,
        .arch_class = .warp,
        .family = .rmem_smem,
        .min_arch = .sm90,
        .source_space = .generic,
        .destination_space = .smem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.warp.mma",
        .source_name = "MmaF16BF16Op",
        .kind = .mma,
        .arch_class = .warp,
        .family = .warp_mma,
        .min_arch = .sm80,
        .dtype_class = .f16_bf16,
    },
    .{
        .source_module = "nvgpu.warp.mma",
        .source_name = "MmaFP8Op",
        .kind = .mma,
        .arch_class = .warp,
        .family = .warp_mma,
        .min_arch = .sm89,
        .dtype_class = .fp8,
    },
    .{
        .source_module = "nvgpu.warp.mma",
        .source_name = "MmaMXF4Op",
        .kind = .mma,
        .arch_class = .sm120_warp,
        .family = .block_scaled_mma,
        .min_arch = .sm120,
        .dtype_class = .mxf4,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warp.mma",
        .source_name = "MmaMXF4NVF4Op",
        .kind = .mma,
        .arch_class = .sm120_warp,
        .family = .block_scaled_mma,
        .min_arch = .sm120,
        .dtype_class = .mxf4,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warp.mma",
        .source_name = "MmaMXF8Op",
        .kind = .mma,
        .arch_class = .sm120_warp,
        .family = .block_scaled_mma,
        .min_arch = .sm120,
        .dtype_class = .mxf8,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warp.mma",
        .source_name = "MmaMXF8F6F4Op",
        .kind = .mma,
        .arch_class = .sm120_warp,
        .family = .block_scaled_mma,
        .min_arch = .sm120,
        .dtype_class = .f4_f6_f8,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warpgroup.mma",
        .source_name = "MmaF16BF16Op",
        .kind = .mma,
        .arch_class = .warpgroup,
        .family = .warpgroup_mma,
        .min_arch = .sm90,
        .dtype_class = .f16_bf16,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warpgroup.mma",
        .source_name = "MmaF8Op",
        .kind = .mma,
        .arch_class = .warpgroup,
        .family = .warpgroup_mma,
        .min_arch = .sm90,
        .dtype_class = .fp8,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.warpgroup.mma",
        .source_name = "MmaI8Op",
        .kind = .mma,
        .arch_class = .warpgroup,
        .family = .warpgroup_mma,
        .min_arch = .sm90,
        .dtype_class = .int,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaTF32Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .tcgen05_mma,
        .min_arch = .sm100,
        .dtype_class = .tf32,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaF16BF16Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .tcgen05_mma,
        .min_arch = .sm100,
        .dtype_class = .f16_bf16,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaI8Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .tcgen05_mma,
        .min_arch = .sm100,
        .dtype_class = .int,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaFP8Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .tcgen05_mma,
        .min_arch = .sm100,
        .dtype_class = .fp8,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaF8F6F4Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .tcgen05_mma,
        .min_arch = .sm100,
        .dtype_class = .f4_f6_f8,
        .allowed_fields = nvgpu.MmaRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaMXF8Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .block_scaled_mma,
        .min_arch = .sm100,
        .dtype_class = .mxf8,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaMXF8F6F4Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .block_scaled_mma,
        .min_arch = .sm100,
        .dtype_class = .f4_f6_f8,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaMXF4Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .block_scaled_mma,
        .min_arch = .sm100,
        .dtype_class = .mxf4,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "MmaMXF4NVF4Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .block_scaled_mma,
        .min_arch = .sm100,
        .dtype_class = .mxf4,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "SM103MmaMXF4Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .block_scaled_mma,
        .min_arch = .sm103,
        .dtype_class = .mxf4,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.mma",
        .source_name = "SM103MmaMXF4NVF4Op",
        .kind = .mma,
        .arch_class = .tcgen05,
        .family = .block_scaled_mma,
        .min_arch = .sm103,
        .dtype_class = .mxf4,
        .allowed_fields = nvgpu.BlockScaledRuntimeFields,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Ld16x64bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 64,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Ld16x128bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Ld16x256bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 256,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Ld16x32bx2Op",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 64,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Ld32x32bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "LdRed16x32bx2Op",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 64,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "LdRed32x32bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .tmem,
        .destination_space = .generic,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "St16x64bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .generic,
        .destination_space = .tmem,
        .value_bits = 64,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "St16x128bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .generic,
        .destination_space = .tmem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "St16x256bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .generic,
        .destination_space = .tmem,
        .value_bits = 256,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "St16x32bx2Op",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .generic,
        .destination_space = .tmem,
        .value_bits = 64,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "St32x32bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .tmem_ldst,
        .min_arch = .sm100,
        .source_space = .generic,
        .destination_space = .tmem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Cp128x256bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .smem_tmem,
        .min_arch = .sm100,
        .source_space = .smem,
        .destination_space = .tmem,
        .value_bits = 256,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Cp128x128bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .smem_tmem,
        .min_arch = .sm100,
        .source_space = .smem,
        .destination_space = .tmem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Cp4x256bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .smem_tmem,
        .min_arch = .sm100,
        .source_space = .smem,
        .destination_space = .tmem,
        .value_bits = 256,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Cp4x32x128bOp",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .smem_tmem,
        .min_arch = .sm100,
        .source_space = .smem,
        .destination_space = .tmem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Cp2x64x128b0213Op",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .smem_tmem,
        .min_arch = .sm100,
        .source_space = .smem,
        .destination_space = .tmem,
        .value_bits = 128,
    },
    .{
        .source_module = "nvgpu.tcgen05.copy",
        .source_name = "Cp2x64x128b0123Op",
        .kind = .copy,
        .arch_class = .tcgen05,
        .family = .smem_tmem,
        .min_arch = .sm100,
        .source_space = .smem,
        .destination_space = .tmem,
        .value_bits = 128,
    },
};

pub fn familyName(family: OpFamily) []const u8 {
    return switch (family) {
        .universal => "universal",
        .gmem_rmem => "gmem_rmem",
        .rmem_gmem => "rmem_gmem",
        .smem_rmem => "smem_rmem",
        .rmem_smem => "rmem_smem",
        .cpasync => "cpasync",
        .tma => "tma",
        .tmem_ldst => "tmem_ldst",
        .smem_tmem => "smem_tmem",
        .warp_mma => "warp_mma",
        .warpgroup_mma => "warpgroup_mma",
        .tcgen05_mma => "tcgen05_mma",
        .block_scaled_mma => "block_scaled_mma",
    };
}

pub fn dtypeMatches(class: DTypeClass, dtype: typing.Numeric) bool {
    return switch (class) {
        .any => true,
        .int => dtype.kind == .signed_int or dtype.kind == .unsigned_int,
        .signed_int => dtype.kind == .signed_int,
        .unsigned_int => dtype.kind == .unsigned_int,
        .float => dtype.isFloat(),
        .f16_bf16 => numericEq(dtype, typing.Float16) or numericEq(dtype, typing.BFloat16),
        .fp8 => dtype.width == 8 and dtype.isFloat(),
        .f4_f6_f8 => dtype.width == 4 or dtype.width == 6 or dtype.width == 8,
        .mxf4 => numericEq(dtype, typing.Float4E2M1FN),
        .mxf8 => numericEq(dtype, typing.Float8E4M3) or numericEq(dtype, typing.Float8E5M2) or
            numericEq(dtype, typing.Float8E8M0FNU),
        .tf32 => numericEq(dtype, typing.TFloat32) or numericEq(dtype, typing.Float32),
    };
}

fn numericEq(a: typing.Numeric, b: typing.Numeric) bool {
    return std.mem.eql(u8, a.name, b.name);
}

pub fn findSpec(module: []const u8, name: []const u8) ?OpSpec {
    for (op_specs) |s| {
        if (std.mem.eql(u8, s.source_module, module) and std.mem.eql(u8, s.source_name, name)) return s;
    }
    return null;
}

pub fn countSpecsForArch(arch: SmArch) usize {
    var n: usize = 0;
    for (op_specs) |s| {
        if (s.supportsArch(arch)) n += 1;
    }
    return n;
}

pub fn countSpecsByClass(class: ArchClass) usize {
    var n: usize = 0;
    for (op_specs) |s| {
        if (s.arch_class == class) n += 1;
    }
    return n;
}

pub fn sourceCoverageCount() usize {
    var n: usize = 0;
    for (op_specs) |s| {
        if (s.sourceRecord() != null) n += 1;
    }
    return n;
}

pub fn validateCatalog() Error!void {
    for (op_specs) |s| try s.ensureSourceGrounded();
}

pub const CatalogMmaConfig = struct {
    arch: SmArch,
    op_name: []const u8,
    module: []const u8,
    a: typing.Numeric,
    b: typing.Numeric,
    acc: typing.Numeric,
    shape_mnk: layout.Tree,
    a_src: OperandSource = .smem,
    a_major_mode: OperandMajorMode = .K,
    b_major_mode: OperandMajorMode = .K,
    cta_group: CtaGroup = .one,
    output_major_mode: OutputMajorMode = .N,

    pub fn makeAtom(self: CatalogMmaConfig) Error!atom.MmaAtom {
        const s = findSpec(self.module, self.op_name) orelse return Error.UnknownArchOperation;
        _ = self.a_src;
        _ = self.a_major_mode;
        _ = self.b_major_mode;
        _ = self.cta_group;
        _ = self.output_major_mode;
        return s.makeMmaAtom(self.arch, self.a, self.b, self.acc, self.shape_mnk);
    }
};

pub const CatalogCopyConfig = struct {
    arch: SmArch,
    op_name: []const u8,
    module: []const u8,
    dtype: typing.Numeric,

    pub fn makeAtom(self: CatalogCopyConfig) Error!atom.CopyAtom {
        const s = findSpec(self.module, self.op_name) orelse return Error.UnknownArchOperation;
        return s.makeCopyAtom(self.arch, self.dtype);
    }
};

pub fn warpF16Mma(
    arch: SmArch,
    shape_mnk: layout.Tree,
    dtype: typing.Numeric,
    acc: typing.Numeric,
) Error!atom.MmaAtom {
    return (CatalogMmaConfig{
        .arch = arch,
        .module = "nvgpu.warp.mma",
        .op_name = "MmaF16BF16Op",
        .a = dtype,
        .b = dtype,
        .acc = acc,
        .shape_mnk = shape_mnk,
    }).makeAtom();
}

pub fn warpgroupF16Mma(
    arch: SmArch,
    shape_mnk: layout.Tree,
    dtype: typing.Numeric,
    acc: typing.Numeric,
) Error!atom.MmaAtom {
    return (CatalogMmaConfig{
        .arch = arch,
        .module = "nvgpu.warpgroup.mma",
        .op_name = "MmaF16BF16Op",
        .a = dtype,
        .b = dtype,
        .acc = acc,
        .shape_mnk = shape_mnk,
    }).makeAtom();
}

pub fn tcgen05F16Mma(
    arch: SmArch,
    shape_mnk: layout.Tree,
    dtype: typing.Numeric,
    acc: typing.Numeric,
) Error!atom.MmaAtom {
    return (CatalogMmaConfig{
        .arch = arch,
        .module = "nvgpu.tcgen05.mma",
        .op_name = "MmaF16BF16Op",
        .a = dtype,
        .b = dtype,
        .acc = acc,
        .shape_mnk = shape_mnk,
    }).makeAtom();
}

pub fn cpAsyncG2S(arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
    return (CatalogCopyConfig{
        .arch = arch,
        .module = "nvgpu.cpasync.copy",
        .op_name = "CopyG2SOp",
        .dtype = dtype,
    }).makeAtom();
}

pub fn tmaTileG2S(arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
    return (CatalogCopyConfig{
        .arch = arch,
        .module = "nvgpu.cpasync.copy",
        .op_name = "CopyBulkTensorTileG2SOp",
        .dtype = dtype,
    }).makeAtom();
}

pub fn tcgen05Ld16x128b(arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
    return (CatalogCopyConfig{
        .arch = arch,
        .module = "nvgpu.tcgen05.copy",
        .op_name = "Ld16x128bOp",
        .dtype = dtype,
    }).makeAtom();
}

pub fn smemLayoutAtom(kind: SmemLayoutAtomKind, element_bits: u16) Error!layout.Layout {
    const swizzle: i64 = switch (kind) {
        .MN_INTER, .K_INTER => 1,
        .MN_SW32, .K_SW32 => 32,
        .MN_SW64, .K_SW64 => 64,
        .MN_SW128, .K_SW128, .MN_SW128_32B => 128,
    };
    if (element_bits == 0) return Error.InvalidInstructionConfig;
    const cols = @max(
        @as(i64, 1),
        @divTrunc(@as(i64, swizzle), @as(i64, @intCast(element_bits))),
    );
    const shape = try layout.Tree.initTuple(&.{
        try layout.Tree.initLeaf(8),
        try layout.Tree.initLeaf(cols),
    });
    return layout.Layout.makeCompact(shape);
}

pub fn writeArchSummary(out: anytype, arch: SmArch) Error!void {
    try out.append("#not_cute.arch<chip = \"");
    try out.append(arch.mlirChip());
    try out.append("\", ops = ");
    try out.appendUnsigned(@intCast(countSpecsForArch(arch)));
    try out.append(">");
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
    return .{
        .name = name,
        .thr_id = layout.makeCompactLayout(.{32}),
        .layout_src_tv = tv,
        .layout_dst_tv = tv,
        .admissible_fields = fields,
        .type_name = "!cute_nvgpu.copy_trait",
    };
}

fn defaultMmaTrait(
    name: []const u8,
    shape: layout.Tree,
    fields: []const atom.RuntimeField,
) Error!atom.Trait {
    const tv = layout.makeCompactLayout(.{ 32, 1 });
    return .{
        .name = name,
        .thr_id = layout.makeCompactLayout(.{32}),
        .shape_mnk = shape,
        .tv_layout_a = tv,
        .tv_layout_b = tv,
        .tv_layout_c = tv,
        .admissible_fields = fields,
        .type_name = "!cute_nvgpu.mma_trait",
    };
}

pub fn validateMmaShape(family: OpFamily, shape_mnk: layout.Tree) Error!void {
    if (shape_mnk.rank() != 3) return Error.UnsupportedInstructionShape;
    try shape_mnk.assertPositive();
    const flat = try shape_mnk.flattenLeaves();
    if (flat.len != 3) return Error.UnsupportedInstructionShape;
    const mv = flat.at(0);
    const nv = flat.at(1);
    const kv = flat.at(2);
    switch (family) {
        .warp_mma => if (!(mv == 16 and (nv == 8 or nv == 16) and
            (kv == 8 or kv == 16 or kv == 32)))
            return Error.UnsupportedInstructionShape,
        .warpgroup_mma => if (!(@mod(mv, 64) == 0 and @mod(nv, 8) == 0 and @mod(kv, 8) == 0))
            return Error.UnsupportedInstructionShape,
        .tcgen05_mma, .block_scaled_mma => if (!(@mod(mv, 64) == 0 and
            @mod(nv, 8) == 0 and @mod(kv, 8) == 0))
            return Error.UnsupportedInstructionShape,
        else => {},
    }
}

pub fn shapeMNK(m: i64, n: i64, k: i64) Error!layout.Tree {
    return layout.Tree.initTuple(&.{
        try layout.Tree.initLeaf(m),
        try layout.Tree.initLeaf(n),
        try layout.Tree.initLeaf(k),
    });
}

pub const CutlassBridgeFinding = struct {
    nvidia_cutlass_imports_dsl: bool,
    nvidia_cutlass_dsl_imports_dsl: bool,
    discovered_cutlass_ir: bool,
    verify_examples_passed_here: bool,
    note: []const u8,
};

pub const sandbox_bridge_finding = CutlassBridgeFinding{
    .nvidia_cutlass_imports_dsl = false,
    .nvidia_cutlass_dsl_imports_dsl = true,
    .discovered_cutlass_ir = true,
    .verify_examples_passed_here = false,
    .note = "pip install nvidia-cutlass installed cutlass_cppgen/cutlass_library, not cutlass._mlir; nvidia-cutlass-dsl installed cutlass._mlir and _cutlass_ir. Default golden examples use CUTLASS parser-aligned MLIR; the fake tensor type remains only in negative-parser fixtures.",
};

test "arch_catalog: architecture chips and catalog are source grounded" {
    try std.testing.expectEqualStrings("sm_90", SmArch.sm90.mlirChip());
    try std.testing.expect(SmArch.sm100.atLeast(.sm90));
    try validateCatalog();
    try std.testing.expect(sourceCoverageCount() == op_specs.len);
    try std.testing.expect(countSpecsByClass(.tcgen05) >= 20);
}

test "arch_catalog: architecture gates reject unsupported operations" {
    const tma = findSpec("nvgpu.cpasync.copy", "CopyBulkTensorTileG2SOp") orelse
        return error.TestExpectedEqual;
    try std.testing.expect(!tma.supportsArch(.sm80));
    try std.testing.expect(tma.supportsArch(.sm90));
    try std.testing.expectError(
        Error.UnsupportedArchitecture,
        tma.makeCopyAtom(.sm80, typing.Float16),
    );
    const ok = try tma.makeCopyAtom(.sm90, typing.Float16);
    try std.testing.expectEqual(typing.AddressSpace.gmem, ok.atom.op.source_space.?);
}

test "arch_catalog: dtype validation follows upstream families" {
    const wgmma = findSpec("nvgpu.warpgroup.mma", "MmaF16BF16Op") orelse
        return error.TestExpectedEqual;
    try wgmma.validateDType(typing.Float16);
    try wgmma.validateDType(typing.BFloat16);
    try std.testing.expectError(Error.UnsupportedOperandType, wgmma.validateDType(typing.Int8));

    const i8_spec = findSpec("nvgpu.warpgroup.mma", "MmaI8Op") orelse return error.TestExpectedEqual;
    try i8_spec.validateDType(typing.Int8);
    try i8_spec.validateDType(typing.Uint8);
    try std.testing.expectError(
        Error.UnsupportedOperandType,
        i8_spec.validateDType(typing.Float16),
    );
}

test "arch_catalog: concrete MMA constructors validate arch, type, shape and runtime fields" {
    const shape = try shapeMNK(64, 8, 16);
    var wg = try warpgroupF16Mma(.sm90, shape, typing.Float16, typing.Float32);
    try wg.set(.accumulate, .{ .bool = true });
    try std.testing.expect((try wg.get(.accumulate)).eql(.{ .bool = true }));
    try std.testing.expectError(
        Error.UnsupportedArchitecture,
        warpgroupF16Mma(.sm80, shape, typing.Float16, typing.Float32),
    );
    try std.testing.expectError(
        Error.UnsupportedOperandType,
        warpgroupF16Mma(.sm90, shape, typing.Int8, typing.Int32),
    );
    try std.testing.expectError(
        Error.UnsupportedInstructionShape,
        warpgroupF16Mma(.sm90, try shapeMNK(32, 8, 16), typing.Float16, typing.Float32),
    );
}

test "arch_catalog: copy constructors cover cp.async, TMA, and tcgen05 spaces" {
    const cp = try cpAsyncG2S(.sm80, typing.Int32);
    try std.testing.expectEqual(typing.AddressSpace.gmem, cp.atom.op.source_space.?);
    try std.testing.expectEqual(typing.AddressSpace.smem, cp.atom.op.destination_space.?);

    var tma = try tmaTileG2S(.sm90, typing.Float16);
    try tma.set(.tma_barrier, .{ .symbol = "%mbar" });
    try std.testing.expect((try tma.get(.tma_barrier)).eql(.{ .symbol = "%mbar" }));

    const ld = try tcgen05Ld16x128b(.sm100, typing.Float16);
    try std.testing.expectEqual(typing.AddressSpace.tmem, ld.atom.op.source_space.?);
    try std.testing.expectEqual(@as(u16, 128), ld.atom.op.num_bits_per_copy.?);
}

test "arch_catalog: smem layout atom and summary writer are deterministic" {
    const l = try smemLayoutAtom(.MN_SW64, 16);
    const flat = try l.shape.flattenLeaves();
    try std.testing.expectEqual(@as(i128, 8), flat.at(0));
    try std.testing.expectEqualStrings("mn_sw64", SmemLayoutAtomKind.MN_SW64.irName());
    var buf: mlir.TextBuffer(256) = .{};
    try writeArchSummary(&buf, .sm90);
    try std.testing.expect(
        std.mem.startsWith(u8, buf.slice(), "#not_cute.arch<chip = \"sm_90\""),
    );
}

test "arch_catalog: sandbox bridge finding records real package behavior" {
    try std.testing.expect(!sandbox_bridge_finding.nvidia_cutlass_imports_dsl);
    try std.testing.expect(sandbox_bridge_finding.nvidia_cutlass_dsl_imports_dsl);
    try std.testing.expect(sandbox_bridge_finding.discovered_cutlass_ir);
    try std.testing.expect(!sandbox_bridge_finding.verify_examples_passed_here);
}

pub const CopyOptions = nvgpu.CopyOptions;
pub const MmaOptions = nvgpu.MmaOptions;
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
        if (@intFromEnum(arch) >= @intFromEnum(SmArch.sm100))
            return nvgpu.tcgen05MmaF16BF16(config.a_dtype, config.acc_dtype, opts);
        if (@intFromEnum(arch) >= @intFromEnum(SmArch.sm90))
            return nvgpu.warpgroupMmaF16BF16(config.a_dtype, config.acc_dtype, opts);
        return nvgpu.warpMmaF16BF16(config.a_dtype, config.acc_dtype, opts);
    }
    if (std.mem.eql(u8, name, "MmaFP8Op") or std.mem.eql(u8, name, "MmaF8Op"))
        return nvgpu.warpMmaFP8(config.a_dtype, config.acc_dtype, opts);
    if (std.mem.eql(u8, name, "MmaTF32Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaTF32Op",
            typing.TFloat32,
            typing.TFloat32,
            typing.Float8E8M0FNU,
            opts,
        );
    if (std.mem.eql(u8, name, "MmaI8Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaI8Op",
            typing.Int8,
            typing.Int8,
            typing.Float8E8M0FNU,
            opts,
        );
    if (std.mem.eql(u8, name, "MmaF8F6F4Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaF8F6F4Op",
            config.a_dtype,
            config.b_dtype,
            config.sf_dtype,
            opts,
        );
    if (std.mem.eql(u8, name, "MmaMXF8Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaMXF8Op",
            config.a_dtype,
            config.b_dtype,
            config.sf_dtype,
            opts,
        );
    if (std.mem.eql(u8, name, "MmaMXF8F6F4Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaMXF8F6F4Op",
            config.a_dtype,
            config.b_dtype,
            config.sf_dtype,
            opts,
        );
    if (std.mem.eql(u8, name, "MmaMXF4Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaMXF4Op",
            typing.Float4E2M1FN,
            typing.Float4E2M1FN,
            config.sf_dtype,
            opts,
        );
    if (std.mem.eql(u8, name, "MmaMXF4NVF4Op"))
        return nvgpu.tcgen05BlockScaledMma(
            "MmaMXF4NVF4Op",
            typing.Float4E2M1FN,
            typing.Float4E2M1FN,
            config.sf_dtype,
            opts,
        );
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
    if (startsWith(name, "CopyBulkTensor") or startsWith(name, "CopyReduceBulkTensor"))
        return nvgpu.tmaCopy(name, config.dtype, .gmem, .smem, opts);
    if (startsWith(name, "CopyBulkG2S"))
        return nvgpu.tmaCopy(name, config.dtype, .gmem, .smem, opts);
    if (startsWith(name, "CopyBulkS2G"))
        return nvgpu.tmaCopy(name, config.dtype, .smem, .gmem, opts);
    if (startsWith(name, "CopyBulkS2S"))
        return nvgpu.tmaCopy(name, config.dtype, .smem, .smem, opts);
    if (startsWith(name, "CopyDsmemStore")) return nvgpu.tmaCopy(name, config.dtype, .smem, .smem, opts);
    if (startsWith(name, "LdMatrix"))
        return nvgpu.warpLdMatrix(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "StMatrix"))
        return nvgpu.warpStMatrix(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "Ld") or startsWith(name, "LdRed"))
        return nvgpu.tcgen05Load(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "St"))
        return nvgpu.tcgen05Store(name, config.dtype, config.bits_per_copy);
    if (startsWith(name, "Cp"))
        return nvgpu.tcgen05S2T(name, config.dtype, config.bits_per_copy);
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

pub const nvvm = struct {
    pub const ReductionOp = enum { add, min, max, and_, or_, xor };
    pub const VoteOp = enum { any, all, uni, ballot };
    pub const FenceProxyKind = enum { async_shared, async_tmem };
    pub const MatchKind = enum { any, all };
    pub const ShiftKind = enum { left, right, wrap };

    pub const IntrinsicCall = struct {
        name: []const u8,
        operands: []const mlir.Operand = &.{},
        operand_types: []const mlir.Type = &.{},
        result_type: ?mlir.Type = null,
        attrs: []const mlir.Attribute = &.{},

        pub fn emit(self: IntrinsicCall, builder: anytype) Error!?mlir.Value {
            if (self.operands.len != self.operand_types.len)
                return Error.InvalidOperandCount;
            if (self.result_type) |rt| {
                return try builder.genericOp(
                    self.name,
                    self.operands,
                    self.attrs,
                    self.operand_types,
                    &.{rt},
                );
            }
            try builder.operationNoResult(.{
                .name = self.name,
                .operands = self.operands,
                .attrs = self.attrs,
                .operand_types = self.operand_types,
                .result_types = &.{},
            });
            return null;
        }
    };

    fn opName(comptime source_name: []const u8) []const u8 {
        return "cutlass.nvvm." ++ source_name;
    }
    fn emitValue(
        builder: anytype,
        comptime source_name: []const u8,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        const maybe = try (IntrinsicCall{
            .name = opName(source_name),
            .operands = operands,
            .operand_types = operand_types,
            .result_type = result_type,
        }).emit(builder);
        return maybe.?;
    }
    fn emitVoid(
        builder: anytype,
        comptime source_name: []const u8,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        _ = try (IntrinsicCall{
            .name = opName(source_name),
            .operands = operands,
            .operand_types = operand_types,
        }).emit(builder);
    }

    pub fn emit(
        builder: anytype,
        source_name: []const u8,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: ?mlir.Type,
    ) Error!?mlir.Value {
        var buf: mlir.TextBuffer(128) = .{};
        try buf.append("cutlass.nvvm.");
        try buf.append(source_name);
        return (IntrinsicCall{
            .name = buf.slice(),
            .operands = operands,
            .operand_types = operand_types,
            .result_type = result_type,
        }).emit(builder);
    }

    pub fn laneIdx(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "lane_idx", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn warpIdx(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "warp_idx", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn physicalWarpId(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "physical_warp_id", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn threadIdx(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "thread_idx", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn blockDim(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "block_dim", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn blockIdx(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "block_idx", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn gridDim(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "grid_dim", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn clusterIdx(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "cluster_idx", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn clusterDim(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "cluster_dim", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn blockInClusterIdx(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "block_in_cluster_idx", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn blockInClusterDim(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "block_in_cluster_dim", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn clusterSize(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "cluster_size", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn blockIdxInCluster(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "block_idx_in_cluster", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn dynamicSmemSize(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "dynamic_smem_size", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn shuffleSyncOp(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "shuffle_sync_op", operands, operand_types, result_type);
    }

    pub fn warpReduction(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "warp_reduction", operands, operand_types, result_type);
    }

    pub fn barrier(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "barrier", operands, operand_types);
    }

    pub fn barrierArrive(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "barrier_arrive", operands, operand_types);
    }

    pub fn syncThreads(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "sync_threads", operands, operand_types);
    }

    pub fn syncWarp(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "sync_warp", operands, operand_types, result_type);
    }

    pub fn fenceAcqRelCta(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "fence_acq_rel_cta", operands, operand_types);
    }

    pub fn fenceAcqRelCluster(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "fence_acq_rel_cluster", operands, operand_types);
    }

    pub fn fenceAcqRelGpu(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "fence_acq_rel_gpu", operands, operand_types);
    }

    pub fn fenceAcqRelSys(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "fence_acq_rel_sys", operands, operand_types);
    }

    pub fn cpAsyncCommitGroup(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cp_async_commit_group", operands, operand_types);
    }

    pub fn cpAsyncWaitGroup(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cp_async_wait_group", operands, operand_types);
    }

    pub fn cpAsyncBulkCommitGroup(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cp_async_bulk_commit_group", operands, operand_types);
    }

    pub fn cpAsyncBulkWaitGroup(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cp_async_bulk_wait_group", operands, operand_types);
    }

    pub fn clusterWait(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cluster_wait", operands, operand_types);
    }

    pub fn clusterArrive(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cluster_arrive", operands, operand_types);
    }

    pub fn clusterArriveRelaxed(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "cluster_arrive_relaxed", operands, operand_types);
    }

    pub fn fenceProxy(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "fence_proxy", operands, operand_types);
    }

    pub fn voteSyncOp(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "vote_sync_op", operands, operand_types, result_type);
    }

    pub fn voteBallotSync(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "vote_ballot_sync", operands, operand_types, result_type);
    }

    pub fn voteAnySync(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "vote_any_sync", operands, operand_types, result_type);
    }

    pub fn voteAllSync(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "vote_all_sync", operands, operand_types, result_type);
    }

    pub fn voteUniSync(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "vote_uni_sync", operands, operand_types, result_type);
    }

    pub fn popc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "popc", operands, operand_types, result_type);
    }

    pub fn fenceViewAsyncTmemOp(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "fence_view_async_tmem_op",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn fenceViewAsyncShared(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "fence_view_async_shared", operands, operand_types);
    }

    pub fn setmaxregisterIncrease(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "setmaxregister_increase", operands, operand_types);
    }

    pub fn setmaxregisterDecrease(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "setmaxregister_decrease", operands, operand_types);
    }

    pub fn warpgroupRegAlloc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "warpgroup_reg_alloc", operands, operand_types);
    }

    pub fn warpgroupRegDealloc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "warpgroup_reg_dealloc", operands, operand_types);
    }

    pub fn calcPackedF32x2Op(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "calc_packed_f32x2_op",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn fmax(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "fmax", operands, operand_types, result_type);
    }

    pub fn fmin(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "fmin", operands, operand_types, result_type);
    }

    pub fn rcpApprox(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "rcp_approx", operands, operand_types, result_type);
    }

    pub fn exp2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "exp2", operands, operand_types, result_type);
    }

    pub fn cvtI8Bf16(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "cvt_i8_bf16", operands, operand_types, result_type);
    }

    pub fn cvtI8x2ToBf16x2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i8x2_to_bf16x2",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtI8x4ToBf16x4(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i8x4_to_bf16x4",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtF32x2Bf16x2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "cvt_f32x2_bf16x2", operands, operand_types, result_type);
    }

    pub fn cvtF32Bf16(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "cvt_f32_bf16", operands, operand_types, result_type);
    }

    pub fn cvtI8x4ToF32x4(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i8x4_to_f32x4",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtI8x2ToF32x2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i8x2_to_f32x2",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn prmt(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "prmt", operands, operand_types, result_type);
    }

    pub fn cvtI4Bf16(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "cvt_i4_bf16", operands, operand_types, result_type);
    }

    pub fn cvtI4ToBf16WithShuffleImpl(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i4_to_bf16_with_shuffle_impl",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtI4ToBf16Impl(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i4_to_bf16_impl",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtI4x2ToBf16x2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i4x2_to_bf16x2",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtI4x4ToBf16x4(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i4x4_to_bf16x4",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtI4x8ToBf16x8(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_i4x8_to_bf16x8",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn sextUnpackedI4x4ToI8x4(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "sext_unpacked_i4x4_to_i8x4",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn log2OfPow2Int(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "log2_of_pow2_int", operands, operand_types, result_type);
    }

    pub fn exp(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "exp", operands, operand_types, result_type);
    }

    pub fn expPackedF32x2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "exp_packed_f32x2", operands, operand_types, result_type);
    }

    pub fn griddepcontrolWait(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "griddepcontrol_wait", operands, operand_types);
    }

    pub fn griddepcontrolLaunchDependents(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(
            builder,
            "griddepcontrol_launch_dependents",
            operands,
            operand_types,
        );
    }

    pub fn warpReduxSync(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "warp_redux_sync", operands, operand_types, result_type);
    }

    pub fn atomicAdd(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_add", operands, operand_types, result_type);
    }

    pub fn atomicAnd(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_and", operands, operand_types, result_type);
    }

    pub fn atomicOr(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_or", operands, operand_types, result_type);
    }

    pub fn atomicXor(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_xor", operands, operand_types, result_type);
    }

    pub fn atomicMax(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_max", operands, operand_types, result_type);
    }

    pub fn atomicMin(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_min", operands, operand_types, result_type);
    }

    pub fn atomicExch(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_exch", operands, operand_types, result_type);
    }

    pub fn atomicFmax(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_fmax", operands, operand_types, result_type);
    }

    pub fn atomicMaxFloat32(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "atomic_max_float32",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn atomicCas(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "atomic_cas", operands, operand_types, result_type);
    }

    pub fn store(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "store", operands, operand_types);
    }

    pub fn load(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "load", operands, operand_types, result_type);
    }

    pub fn red(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "red", operands, operand_types);
    }

    pub fn cvtF4e2m1F16(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "cvt_f4e2m1_f16", operands, operand_types, result_type);
    }

    pub fn cvtF4e2m1x2ToF16x2(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_f4e2m1x2_to_f16x2",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtF4e2m1x4ToF16x4(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_f4e2m1x4_to_f16x4",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn cvtF4e2m1x8ToF16x8(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(
            builder,
            "cvt_f4e2m1x8_to_f16x8",
            operands,
            operand_types,
            result_type,
        );
    }

    pub fn smid(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "smid", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn nsmid(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "nsmid", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn clock(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "clock", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn clock64(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "clock64", &.{}, &.{}, mlir.Type.i(64));
    }

    pub fn matchSync(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "match_sync", operands, operand_types, result_type);
    }

    pub fn clz(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "clz", operands, operand_types, result_type);
    }

    pub fn bfind(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "bfind", operands, operand_types, result_type);
    }

    pub fn brev(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "brev", operands, operand_types, result_type);
    }

    pub fn bfe(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "bfe", operands, operand_types, result_type);
    }

    pub fn bfi(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "bfi", operands, operand_types, result_type);
    }

    pub fn mulHi(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "mul_hi", operands, operand_types, result_type);
    }

    pub fn mulWide(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "mul_wide", operands, operand_types, result_type);
    }

    pub fn mul24(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "mul24", operands, operand_types, result_type);
    }

    pub fn mad24(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "mad24", operands, operand_types, result_type);
    }

    pub fn addCc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "add_cc", operands, operand_types, result_type);
    }

    pub fn addc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "addc", operands, operand_types, result_type);
    }

    pub fn subCc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "sub_cc", operands, operand_types, result_type);
    }

    pub fn subc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "subc", operands, operand_types, result_type);
    }

    pub fn madCc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "mad_cc", operands, operand_types, result_type);
    }

    pub fn madc(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "madc", operands, operand_types, result_type);
    }

    pub fn activemask(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "activemask", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn lanemaskLt(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "lanemask_lt", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn lanemaskLe(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "lanemask_le", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn lanemaskEq(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "lanemask_eq", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn lanemaskGe(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "lanemask_ge", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn lanemaskGt(builder: anytype) Error!mlir.Value {
        return emitValue(builder, "lanemask_gt", &.{}, &.{}, mlir.Type.i(32));
    }

    pub fn addSatInt(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "add_sat_int", operands, operand_types, result_type);
    }

    pub fn subSatInt(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "sub_sat_int", operands, operand_types, result_type);
    }

    pub fn lop3(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "lop3", operands, operand_types, result_type);
    }

    pub fn shf(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
        result_type: mlir.Type,
    ) Error!mlir.Value {
        return emitValue(builder, "shf", operands, operand_types, result_type);
    }

    pub fn prefetch(
        builder: anytype,
        operands: []const mlir.Operand,
        operand_types: []const mlir.Type,
    ) Error!void {
        return emitVoid(builder, "prefetch", operands, operand_types);
    }

    test "arch_nvvm: source-name wrappers emit deterministic intrinsic calls" {
        var b: mlir.Builder(4096) = .{};
        _ = try laneIdx(&b);
        _ = try shuffleSyncOp(
            &b,
            &.{ .arg(0), .arg(1) },
            &.{ mlir.Type.i(32), mlir.Type.i(32) },
            mlir.Type.i(32),
        );
        try barrier(&b, &.{}, &.{});
        try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cutlass.nvvm.lane_idx") != null);
        try std.testing.expect(
            std.mem.indexOf(u8, b.slice(), "cutlass.nvvm.shuffle_sync_op") != null,
        );
        try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cutlass.nvvm.barrier") != null);
    }
};
