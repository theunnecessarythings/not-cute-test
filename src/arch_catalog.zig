const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const atom = @import("atom.zig");
const nvgpu = @import("nvgpu.zig");
const mlir = @import("mlir_text.zig");

pub const Error = atom.Error || typing.Error || mlir.Error || error{
    UnsupportedArchitecture,
    UnsupportedOperandType,
    UnsupportedInstructionShape,
    UnknownArchOperation,
    InvalidInstructionConfig,
};

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
pub const OpFamily = enum { universal, gmem_rmem, rmem_gmem, smem_rmem, rmem_smem, cpasync, tma, tmem_ldst, smem_tmem, warp_mma, warpgroup_mma, tcgen05_mma, block_scaled_mma };
pub const OperandMajorMode = nvgpu.OperandMajorMode;
pub const OperandSource = nvgpu.OperandSource;
pub const CtaGroup = nvgpu.CtaGroup;
pub const OutputMajorMode = nvgpu.OutputMajorMode;

pub const DTypeClass = enum { any, int, signed_int, unsigned_int, float, f16_bf16, fp8, f4_f6_f8, mxf4, mxf8, tf32 };

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
        return atom.makeCopyAtom(.{
            .name = self.source_name,
            .kind = .copy,
            .arch = arch.mlirChip(),
            .family = familyName(self.family),
            .value_type = dtype,
            .num_bits_per_copy = self.value_bits orelse dtype.width,
            .source_space = self.source_space orelse .generic,
            .destination_space = self.destination_space orelse .generic,
            .allowed_fields = self.allowed_fields,
        }, defaultCopyTrait(self.source_name, dtype, self.value_bits orelse dtype.width, self.allowed_fields) catch |err| return err);
    }

    pub fn makeMmaAtom(self: OpSpec, arch: SmArch, a: typing.Numeric, b: typing.Numeric, acc: typing.Numeric, shape_mnk: layout.Tree) Error!atom.MmaAtom {
        try self.ensureSourceGrounded();
        if (self.kind != .mma) return Error.InvalidInstructionConfig;
        if (!self.supportsArch(arch)) return Error.UnsupportedArchitecture;
        try self.validateDType(a);
        try self.validateDType(b);
        try validateMmaShape(self.family, shape_mnk);
        return atom.makeMmaAtom(.{
            .name = self.source_name,
            .kind = .mma,
            .arch = arch.mlirChip(),
            .family = familyName(self.family),
            .instruction_shape_mnk = shape_mnk,
            .a_type = a,
            .b_type = b,
            .c_type = acc,
            .allowed_fields = self.allowed_fields,
        }, defaultMmaTrait(self.source_name, shape_mnk, self.allowed_fields) catch |err| return err);
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
    .{ .source_module = "nvgpu.common", .source_name = "MmaUniversalOp", .kind = .mma, .arch_class = .generic, .family = .universal, .min_arch = .sm70, .dtype_class = .float },
    .{ .source_module = "nvgpu.common", .source_name = "CopyUniversalOp", .kind = .copy, .arch_class = .generic, .family = .universal, .min_arch = .sm70 },
    .{ .source_module = "nvgpu.common", .source_name = "CopyG2ROp", .kind = .copy, .arch_class = .generic, .family = .gmem_rmem, .min_arch = .sm70, .source_space = .gmem, .destination_space = .generic, .allowed_fields = nvgpu.CachePolicyRuntimeFields },
    .{ .source_module = "nvgpu.common", .source_name = "CopyR2GOp", .kind = .copy, .arch_class = .generic, .family = .rmem_gmem, .min_arch = .sm70, .source_space = .generic, .destination_space = .gmem, .allowed_fields = nvgpu.CachePolicyRuntimeFields },
    .{ .source_module = "nvgpu.common", .source_name = "CopyS2ROp", .kind = .copy, .arch_class = .generic, .family = .smem_rmem, .min_arch = .sm70, .source_space = .smem, .destination_space = .generic },
    .{ .source_module = "nvgpu.common", .source_name = "CopyR2SOp", .kind = .copy, .arch_class = .generic, .family = .rmem_smem, .min_arch = .sm70, .source_space = .generic, .destination_space = .smem },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyG2SOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm80, .source_space = .gmem, .destination_space = .smem, .value_bits = 128 },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkTensorTileG2SOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .gmem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkTensorIm2ColG2SOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .gmem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkTensorIm2ColG2SMulticastOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .gmem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkTensorIm2ColS2GOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .smem, .destination_space = .gmem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkTensorTileG2SMulticastOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .gmem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkTensorTileS2GOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .smem, .destination_space = .gmem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyReduceBulkTensorTileS2GOp", .kind = .copy, .arch_class = .cpasync, .family = .tma, .min_arch = .sm90, .source_space = .smem, .destination_space = .gmem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkG2SOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm90, .source_space = .gmem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.CachePolicyRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkG2SMulticastOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm90, .source_space = .gmem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkS2GOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm90, .source_space = .smem, .destination_space = .gmem, .value_bits = 128 },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkS2GByteMaskOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm90, .source_space = .smem, .destination_space = .gmem, .value_bits = 128, .allowed_fields = nvgpu.ByteMaskRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyBulkS2SOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm90, .source_space = .smem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.cpasync.copy", .source_name = "CopyDsmemStoreOp", .kind = .copy, .arch_class = .cpasync, .family = .cpasync, .min_arch = .sm90, .source_space = .smem, .destination_space = .smem, .value_bits = 128, .allowed_fields = nvgpu.TmaRuntimeFields },
    .{ .source_module = "nvgpu.warp.copy", .source_name = "LdMatrix8x8x16bOp", .kind = .copy, .arch_class = .warp, .family = .smem_rmem, .min_arch = .sm75, .source_space = .smem, .destination_space = .generic, .value_bits = 128, .dtype_class = .f16_bf16 },
    .{ .source_module = "nvgpu.warp.copy", .source_name = "LdMatrix8x16x8bOp", .kind = .copy, .arch_class = .warp, .family = .smem_rmem, .min_arch = .sm75, .source_space = .smem, .destination_space = .generic, .value_bits = 128 },
    .{ .source_module = "nvgpu.warp.copy", .source_name = "LdMatrix16x8x8bOp", .kind = .copy, .arch_class = .warp, .family = .smem_rmem, .min_arch = .sm75, .source_space = .smem, .destination_space = .generic, .value_bits = 128 },
    .{ .source_module = "nvgpu.warp.copy", .source_name = "LdMatrix16x16x8bOp", .kind = .copy, .arch_class = .warp, .family = .smem_rmem, .min_arch = .sm75, .source_space = .smem, .destination_space = .generic, .value_bits = 128 },
    .{ .source_module = "nvgpu.warp.copy", .source_name = "StMatrix8x8x16bOp", .kind = .copy, .arch_class = .warp, .family = .rmem_smem, .min_arch = .sm90, .source_space = .generic, .destination_space = .smem, .value_bits = 128, .dtype_class = .f16_bf16 },
    .{ .source_module = "nvgpu.warp.copy", .source_name = "StMatrix16x8x8bOp", .kind = .copy, .arch_class = .warp, .family = .rmem_smem, .min_arch = .sm90, .source_space = .generic, .destination_space = .smem, .value_bits = 128 },
    .{ .source_module = "nvgpu.warp.mma", .source_name = "MmaF16BF16Op", .kind = .mma, .arch_class = .warp, .family = .warp_mma, .min_arch = .sm80, .dtype_class = .f16_bf16 },
    .{ .source_module = "nvgpu.warp.mma", .source_name = "MmaFP8Op", .kind = .mma, .arch_class = .warp, .family = .warp_mma, .min_arch = .sm89, .dtype_class = .fp8 },
    .{ .source_module = "nvgpu.warp.mma", .source_name = "MmaMXF4Op", .kind = .mma, .arch_class = .sm120_warp, .family = .block_scaled_mma, .min_arch = .sm120, .dtype_class = .mxf4, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.warp.mma", .source_name = "MmaMXF4NVF4Op", .kind = .mma, .arch_class = .sm120_warp, .family = .block_scaled_mma, .min_arch = .sm120, .dtype_class = .mxf4, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.warp.mma", .source_name = "MmaMXF8Op", .kind = .mma, .arch_class = .sm120_warp, .family = .block_scaled_mma, .min_arch = .sm120, .dtype_class = .mxf8, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.warp.mma", .source_name = "MmaMXF8F6F4Op", .kind = .mma, .arch_class = .sm120_warp, .family = .block_scaled_mma, .min_arch = .sm120, .dtype_class = .f4_f6_f8, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.warpgroup.mma", .source_name = "MmaF16BF16Op", .kind = .mma, .arch_class = .warpgroup, .family = .warpgroup_mma, .min_arch = .sm90, .dtype_class = .f16_bf16, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.warpgroup.mma", .source_name = "MmaF8Op", .kind = .mma, .arch_class = .warpgroup, .family = .warpgroup_mma, .min_arch = .sm90, .dtype_class = .fp8, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.warpgroup.mma", .source_name = "MmaI8Op", .kind = .mma, .arch_class = .warpgroup, .family = .warpgroup_mma, .min_arch = .sm90, .dtype_class = .int, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaTF32Op", .kind = .mma, .arch_class = .tcgen05, .family = .tcgen05_mma, .min_arch = .sm100, .dtype_class = .tf32, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaF16BF16Op", .kind = .mma, .arch_class = .tcgen05, .family = .tcgen05_mma, .min_arch = .sm100, .dtype_class = .f16_bf16, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaI8Op", .kind = .mma, .arch_class = .tcgen05, .family = .tcgen05_mma, .min_arch = .sm100, .dtype_class = .int, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaFP8Op", .kind = .mma, .arch_class = .tcgen05, .family = .tcgen05_mma, .min_arch = .sm100, .dtype_class = .fp8, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaF8F6F4Op", .kind = .mma, .arch_class = .tcgen05, .family = .tcgen05_mma, .min_arch = .sm100, .dtype_class = .f4_f6_f8, .allowed_fields = nvgpu.MmaRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaMXF8Op", .kind = .mma, .arch_class = .tcgen05, .family = .block_scaled_mma, .min_arch = .sm100, .dtype_class = .mxf8, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaMXF8F6F4Op", .kind = .mma, .arch_class = .tcgen05, .family = .block_scaled_mma, .min_arch = .sm100, .dtype_class = .f4_f6_f8, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaMXF4Op", .kind = .mma, .arch_class = .tcgen05, .family = .block_scaled_mma, .min_arch = .sm100, .dtype_class = .mxf4, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "MmaMXF4NVF4Op", .kind = .mma, .arch_class = .tcgen05, .family = .block_scaled_mma, .min_arch = .sm100, .dtype_class = .mxf4, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "SM103MmaMXF4Op", .kind = .mma, .arch_class = .tcgen05, .family = .block_scaled_mma, .min_arch = .sm103, .dtype_class = .mxf4, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.mma", .source_name = "SM103MmaMXF4NVF4Op", .kind = .mma, .arch_class = .tcgen05, .family = .block_scaled_mma, .min_arch = .sm103, .dtype_class = .mxf4, .allowed_fields = nvgpu.BlockScaledRuntimeFields },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Ld16x64bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 64 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Ld16x128bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Ld16x256bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 256 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Ld16x32bx2Op", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 64 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Ld32x32bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "LdRed16x32bx2Op", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 64 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "LdRed32x32bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .tmem, .destination_space = .generic, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "St16x64bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .generic, .destination_space = .tmem, .value_bits = 64 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "St16x128bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .generic, .destination_space = .tmem, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "St16x256bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .generic, .destination_space = .tmem, .value_bits = 256 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "St16x32bx2Op", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .generic, .destination_space = .tmem, .value_bits = 64 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "St32x32bOp", .kind = .copy, .arch_class = .tcgen05, .family = .tmem_ldst, .min_arch = .sm100, .source_space = .generic, .destination_space = .tmem, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Cp128x256bOp", .kind = .copy, .arch_class = .tcgen05, .family = .smem_tmem, .min_arch = .sm100, .source_space = .smem, .destination_space = .tmem, .value_bits = 256 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Cp128x128bOp", .kind = .copy, .arch_class = .tcgen05, .family = .smem_tmem, .min_arch = .sm100, .source_space = .smem, .destination_space = .tmem, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Cp4x256bOp", .kind = .copy, .arch_class = .tcgen05, .family = .smem_tmem, .min_arch = .sm100, .source_space = .smem, .destination_space = .tmem, .value_bits = 256 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Cp4x32x128bOp", .kind = .copy, .arch_class = .tcgen05, .family = .smem_tmem, .min_arch = .sm100, .source_space = .smem, .destination_space = .tmem, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Cp2x64x128b0213Op", .kind = .copy, .arch_class = .tcgen05, .family = .smem_tmem, .min_arch = .sm100, .source_space = .smem, .destination_space = .tmem, .value_bits = 128 },
    .{ .source_module = "nvgpu.tcgen05.copy", .source_name = "Cp2x64x128b0123Op", .kind = .copy, .arch_class = .tcgen05, .family = .smem_tmem, .min_arch = .sm100, .source_space = .smem, .destination_space = .tmem, .value_bits = 128 },
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
        .mxf8 => numericEq(dtype, typing.Float8E4M3) or numericEq(dtype, typing.Float8E5M2) or numericEq(dtype, typing.Float8E8M0FNU),
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

pub const MmaConfig = struct {
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

    pub fn makeAtom(self: MmaConfig) Error!atom.MmaAtom {
        const s = findSpec(self.module, self.op_name) orelse return Error.UnknownArchOperation;
        _ = self.a_src;
        _ = self.a_major_mode;
        _ = self.b_major_mode;
        _ = self.cta_group;
        _ = self.output_major_mode;
        return s.makeMmaAtom(self.arch, self.a, self.b, self.acc, self.shape_mnk);
    }
};

pub const CopyConfig = struct {
    arch: SmArch,
    op_name: []const u8,
    module: []const u8,
    dtype: typing.Numeric,

    pub fn makeAtom(self: CopyConfig) Error!atom.CopyAtom {
        const s = findSpec(self.module, self.op_name) orelse return Error.UnknownArchOperation;
        return s.makeCopyAtom(self.arch, self.dtype);
    }
};

pub fn warpF16Mma(arch: SmArch, shape_mnk: layout.Tree, dtype: typing.Numeric, acc: typing.Numeric) Error!atom.MmaAtom {
    return (MmaConfig{ .arch = arch, .module = "nvgpu.warp.mma", .op_name = "MmaF16BF16Op", .a = dtype, .b = dtype, .acc = acc, .shape_mnk = shape_mnk }).makeAtom();
}

pub fn warpgroupF16Mma(arch: SmArch, shape_mnk: layout.Tree, dtype: typing.Numeric, acc: typing.Numeric) Error!atom.MmaAtom {
    return (MmaConfig{ .arch = arch, .module = "nvgpu.warpgroup.mma", .op_name = "MmaF16BF16Op", .a = dtype, .b = dtype, .acc = acc, .shape_mnk = shape_mnk }).makeAtom();
}

pub fn tcgen05F16Mma(arch: SmArch, shape_mnk: layout.Tree, dtype: typing.Numeric, acc: typing.Numeric) Error!atom.MmaAtom {
    return (MmaConfig{ .arch = arch, .module = "nvgpu.tcgen05.mma", .op_name = "MmaF16BF16Op", .a = dtype, .b = dtype, .acc = acc, .shape_mnk = shape_mnk }).makeAtom();
}

pub fn cpAsyncG2S(arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
    return (CopyConfig{ .arch = arch, .module = "nvgpu.cpasync.copy", .op_name = "CopyG2SOp", .dtype = dtype }).makeAtom();
}

pub fn tmaTileG2S(arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
    return (CopyConfig{ .arch = arch, .module = "nvgpu.cpasync.copy", .op_name = "CopyBulkTensorTileG2SOp", .dtype = dtype }).makeAtom();
}

pub fn tcgen05Ld16x128b(arch: SmArch, dtype: typing.Numeric) Error!atom.CopyAtom {
    return (CopyConfig{ .arch = arch, .module = "nvgpu.tcgen05.copy", .op_name = "Ld16x128bOp", .dtype = dtype }).makeAtom();
}

pub fn smemLayoutAtom(kind: SmemLayoutAtomKind, element_bits: u16) Error!layout.Layout {
    const swizzle: i64 = switch (kind) {
        .MN_INTER, .K_INTER => 1,
        .MN_SW32, .K_SW32 => 32,
        .MN_SW64, .K_SW64 => 64,
        .MN_SW128, .K_SW128, .MN_SW128_32B => 128,
    };
    if (element_bits == 0) return Error.InvalidInstructionConfig;
    const cols = @max(@as(i64, 1), @divTrunc(@as(i64, swizzle), @as(i64, @intCast(element_bits))));
    const shape = try layout.Tree.initTuple(&.{ try layout.Tree.initLeaf(8), try layout.Tree.initLeaf(cols) });
    return layout.Layout.makeCompact(shape);
}

pub fn writeArchSummary(out: anytype, arch: SmArch) Error!void {
    try out.append("#not_cute.arch<chip = \"");
    try out.append(arch.mlirChip());
    try out.append("\", ops = ");
    try out.appendUnsigned(@intCast(countSpecsForArch(arch)));
    try out.append(">");
}

fn defaultCopyTrait(name: []const u8, dtype: typing.Numeric, bits_per_copy: u16, fields: []const atom.RuntimeField) Error!atom.Trait {
    const values = @max(@as(u16, 1), bits_per_copy / @max(@as(u16, 1), dtype.width));
    const tv_shape = try layout.Tree.initTuple(&.{ try layout.Tree.initLeaf(32), try layout.Tree.initLeaf(@intCast(values)) });
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

fn defaultMmaTrait(name: []const u8, shape: layout.Tree, fields: []const atom.RuntimeField) Error!atom.Trait {
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
        .warp_mma => if (!(mv == 16 and (nv == 8 or nv == 16) and (kv == 8 or kv == 16 or kv == 32))) return Error.UnsupportedInstructionShape,
        .warpgroup_mma => if (!(@mod(mv, 64) == 0 and @mod(nv, 8) == 0 and @mod(kv, 8) == 0)) return Error.UnsupportedInstructionShape,
        .tcgen05_mma, .block_scaled_mma => if (!(@mod(mv, 64) == 0 and @mod(nv, 8) == 0 and @mod(kv, 8) == 0)) return Error.UnsupportedInstructionShape,
        else => {},
    }
}

pub fn shapeMNK(m: i64, n: i64, k: i64) Error!layout.Tree {
    return layout.Tree.initTuple(&.{ try layout.Tree.initLeaf(m), try layout.Tree.initLeaf(n), try layout.Tree.initLeaf(k) });
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
    const tma = findSpec("nvgpu.cpasync.copy", "CopyBulkTensorTileG2SOp") orelse return error.TestExpectedEqual;
    try std.testing.expect(!tma.supportsArch(.sm80));
    try std.testing.expect(tma.supportsArch(.sm90));
    try std.testing.expectError(Error.UnsupportedArchitecture, tma.makeCopyAtom(.sm80, typing.Float16));
    const ok = try tma.makeCopyAtom(.sm90, typing.Float16);
    try std.testing.expectEqual(typing.AddressSpace.gmem, ok.atom.op.source_space.?);
}

test "arch_catalog: dtype validation follows upstream families" {
    const wgmma = findSpec("nvgpu.warpgroup.mma", "MmaF16BF16Op") orelse return error.TestExpectedEqual;
    try wgmma.validateDType(typing.Float16);
    try wgmma.validateDType(typing.BFloat16);
    try std.testing.expectError(Error.UnsupportedOperandType, wgmma.validateDType(typing.Int8));

    const i8_spec = findSpec("nvgpu.warpgroup.mma", "MmaI8Op") orelse return error.TestExpectedEqual;
    try i8_spec.validateDType(typing.Int8);
    try i8_spec.validateDType(typing.Uint8);
    try std.testing.expectError(Error.UnsupportedOperandType, i8_spec.validateDType(typing.Float16));
}

test "arch_catalog: concrete MMA constructors validate arch, type, shape and runtime fields" {
    const shape = try shapeMNK(64, 8, 16);
    var wg = try warpgroupF16Mma(.sm90, shape, typing.Float16, typing.Float32);
    try wg.set(.accumulate, .{ .bool = true });
    try std.testing.expect((try wg.get(.accumulate)).eql(.{ .bool = true }));
    try std.testing.expectError(Error.UnsupportedArchitecture, warpgroupF16Mma(.sm80, shape, typing.Float16, typing.Float32));
    try std.testing.expectError(Error.UnsupportedOperandType, warpgroupF16Mma(.sm90, shape, typing.Int8, typing.Int32));
    try std.testing.expectError(Error.UnsupportedInstructionShape, warpgroupF16Mma(.sm90, try shapeMNK(32, 8, 16), typing.Float16, typing.Float32));
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
    try std.testing.expect(std.mem.startsWith(u8, buf.slice(), "#not_cute.arch<chip = \"sm_90\""));
}

test "arch_catalog: sandbox bridge finding records real package behavior" {
    try std.testing.expect(!sandbox_bridge_finding.nvidia_cutlass_imports_dsl);
    try std.testing.expect(sandbox_bridge_finding.nvidia_cutlass_dsl_imports_dsl);
    try std.testing.expect(sandbox_bridge_finding.discovered_cutlass_ir);
    try std.testing.expect(!sandbox_bridge_finding.verify_examples_passed_here);
}
