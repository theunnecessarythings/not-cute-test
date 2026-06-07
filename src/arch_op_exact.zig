const std = @import("std");
const mlir = @import("mlir_text.zig");
const arch_exact = @import("arch_exact.zig");
const arch_ops = @import("arch_ops.zig");

pub const Error = mlir.Error || error{
    UnsupportedArchitecture,
    UnsupportedElementType,
    UnsupportedOperationShape,
    UnsupportedBytes,
    InvalidMemorySpace,
    InvalidRuntimeField,
    InvalidAtomConfig,
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

    pub fn atLeast(self: SmArch, min: SmArch) bool {
        return @intFromEnum(self) >= @intFromEnum(min);
    }
    pub fn mlirName(self: SmArch) []const u8 {
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

pub const Element = enum {
    f16,
    bf16,
    tf32,
    f32,
    f64,
    s8,
    u8,
    s4,
    u4,
    e4m3,
    e5m2,
    b1,

    pub fn mlir(self: Element) []const u8 {
        return switch (self) {
            .f16 => "f16",
            .bf16 => "bf16",
            .tf32 => "tf32",
            .f32 => "f32",
            .f64 => "f64",
            .s8 => "i8",
            .u8 => "ui8",
            .s4 => "i4",
            .u4 => "ui4",
            .e4m3 => "f8E4M3FN",
            .e5m2 => "f8E5M2",
            .b1 => "i1",
        };
    }
    pub fn isFloat(self: Element) bool {
        return switch (self) {
            .f16, .bf16, .tf32, .f32, .f64, .e4m3, .e5m2 => true,
            else => false,
        };
    }
    pub fn isInt(self: Element) bool {
        return !self.isFloat();
    }
};

pub const MemorySpace = enum { gmem, smem, tmem, rmem, generic };
pub const RuntimeField = enum { cache_policy, byte_mask, multicast_mask, mbarrier, scale_d, scale_a, scale_b, negate_a, negate_b, accumulate, fence };

pub const MmaShape = struct {
    m: u16,
    n: u16,
    k: u16,

    pub fn validate(self: MmaShape) Error!void {
        if (self.m == 0 or self.n == 0 or self.k == 0) return Error.UnsupportedOperationShape;
    }

    pub fn write(self: MmaShape, out: anytype) !void {
        try out.appendUnsigned(self.m);
        try out.append("x");
        try out.appendUnsigned(self.n);
        try out.append("x");
        try out.appendUnsigned(self.k);
    }
};

pub const MmaFamily = enum { sm70_mma, sm75_mma, sm80_mma, sm90_wgmma, sm100_tcgen05, universal_fma };
pub const CopyFamily = enum { universal_copy, cp_async, tma_load, tma_store, ldmatrix, stmatrix, tcgen05_ld, tcgen05_st };

pub const MmaConfig = struct {
    family: MmaFamily,
    arch: SmArch,
    shape: MmaShape,
    a: Element,
    b: Element,
    c: Element,
    d: Element,
    runtime_fields: []const RuntimeField = &.{},

    pub fn validate(self: MmaConfig) Error!void {
        try self.shape.validate();
        switch (self.family) {
            .sm70_mma => if (!self.arch.atLeast(.sm70)) return Error.UnsupportedArchitecture,
            .sm75_mma => if (!self.arch.atLeast(.sm75)) return Error.UnsupportedArchitecture,
            .sm80_mma => if (!self.arch.atLeast(.sm80)) return Error.UnsupportedArchitecture,
            .sm90_wgmma => if (!self.arch.atLeast(.sm90)) return Error.UnsupportedArchitecture,
            .sm100_tcgen05 => if (!self.arch.atLeast(.sm100)) return Error.UnsupportedArchitecture,
            .universal_fma => {},
        }
        if (self.a != self.b) return Error.UnsupportedElementType;
        if (self.c != self.d) return Error.UnsupportedElementType;
        switch (self.family) {
            .sm80_mma, .sm90_wgmma, .sm100_tcgen05 => if (!(self.a.isFloat() or self.a.isInt())) return Error.UnsupportedElementType,
            else => {},
        }
        for (self.runtime_fields) |f| switch (f) {
            .scale_a, .scale_b, .scale_d, .negate_a, .negate_b => if (self.family != .sm100_tcgen05 and self.family != .sm90_wgmma) return Error.InvalidRuntimeField,
            .accumulate => {},
            else => return Error.InvalidRuntimeField,
        };
    }

    pub fn writeMlirType(self: MmaConfig, out: anytype) Error!void {
        try self.validate();
        switch (self.family) {
            .universal_fma => {
                try out.append("!cute_nvgpu.atom.universal_fma<");
                try self.shape.write(out);
                try out.append(", (");
                try out.append(self.a.mlir());
                try out.append(", ");
                try out.append(self.b.mlir());
                try out.append(") -> ");
                try out.append(self.d.mlir());
                try out.append(" >");
            },
            else => {
                try out.append("!cute_nvgpu.atom.");
                try out.append(@tagName(self.family));
                try out.append("<");
                try self.shape.write(out);
                try out.append(", ");
                try out.append(self.a.mlir());
                try out.append(", ");
                try out.append(self.d.mlir());
                try out.append(", ");
                try out.append(self.arch.mlirName());
                try out.append(">");
            },
        }
    }
};

pub const CopyConfig = struct {
    family: CopyFamily,
    arch: SmArch,
    element: Element,
    bits: u16,
    src: MemorySpace,
    dst: MemorySpace,
    runtime_fields: []const RuntimeField = &.{},

    pub fn validate(self: CopyConfig) Error!void {
        if (self.bits == 0 or self.bits % 8 != 0) return Error.UnsupportedBytes;
        switch (self.family) {
            .universal_copy => {},
            .cp_async => {
                if (!self.arch.atLeast(.sm80)) return Error.UnsupportedArchitecture;
                if (self.src != .gmem or self.dst != .smem) return Error.InvalidMemorySpace;
                if (!(self.bits == 32 or self.bits == 64 or self.bits == 128)) return Error.UnsupportedBytes;
            },
            .tma_load, .tma_store => if (!self.arch.atLeast(.sm90)) return Error.UnsupportedArchitecture,
            .ldmatrix, .stmatrix => if (!self.arch.atLeast(.sm75)) return Error.UnsupportedArchitecture,
            .tcgen05_ld, .tcgen05_st => if (!self.arch.atLeast(.sm100)) return Error.UnsupportedArchitecture,
        }
        for (self.runtime_fields) |f| switch (f) {
            .cache_policy, .byte_mask => if (self.family != .cp_async and self.family != .universal_copy) return Error.InvalidRuntimeField,
            .multicast_mask, .mbarrier => if (self.family != .tma_load and self.family != .tma_store) return Error.InvalidRuntimeField,
            .fence => if (self.family != .tcgen05_ld and self.family != .tcgen05_st) return Error.InvalidRuntimeField,
            else => return Error.InvalidRuntimeField,
        };
    }

    pub fn writeMlirType(self: CopyConfig, out: anytype) Error!void {
        try self.validate();
        switch (self.family) {
            .universal_copy => {
                try out.append("!cute_nvgpu.atom.universal_copy<");
                try out.append(self.element.mlir());
                try out.append(", ");
                try out.appendUnsigned(self.bits);
                try out.append(" b>");
            },
            else => {
                try out.append("!cute_nvgpu.atom.");
                try out.append(@tagName(self.family));
                try out.append("<");
                try out.append(self.element.mlir());
                try out.append(", ");
                try out.appendUnsigned(self.bits);
                try out.append(" b, ");
                try out.append(self.arch.mlirName());
                try out.append(">");
            },
        }
    }
};

pub const ExactOpSummary = struct {
    manifest_records: usize,
    copy_records: usize,
    mma_records: usize,
    source_exact_rules: usize,

    pub fn fromManifest() ExactOpSummary {
        var copy_count: usize = 0;
        var mma_count: usize = 0;
        var rule_count: usize = 0;
        for (arch_exact.records) |r| {
            if (std.mem.indexOf(u8, r.name, "Copy") != null or std.mem.indexOf(u8, r.name, "copy") != null) copy_count += 1;
            if (std.mem.indexOf(u8, r.name, "Mma") != null or std.mem.indexOf(u8, r.name, "mma") != null or std.mem.indexOf(u8, r.name, "MMA") != null) mma_count += 1;
            if (r.rule_len != 0) rule_count += 1;
        }
        return .{ .manifest_records = arch_exact.records.len, .copy_records = copy_count, .mma_records = mma_count, .source_exact_rules = rule_count };
    }
};

pub fn universalFma(dtype: Element) MmaConfig {
    return .{ .family = .universal_fma, .arch = .sm70, .shape = .{ .m = 1, .n = 1, .k = 1 }, .a = dtype, .b = dtype, .c = dtype, .d = dtype };
}

pub fn cpAsync(arch: SmArch, bits: u16) CopyConfig {
    return .{ .family = .cp_async, .arch = arch, .element = .u8, .bits = bits, .src = .gmem, .dst = .smem, .runtime_fields = &.{ .cache_policy, .byte_mask } };
}

pub fn tmaLoad(arch: SmArch, element: Element, bits: u16) CopyConfig {
    return .{ .family = .tma_load, .arch = arch, .element = element, .bits = bits, .src = .gmem, .dst = .smem, .runtime_fields = &.{ .mbarrier, .multicast_mask } };
}

pub fn wgmmaF16(arch: SmArch, shape: MmaShape) MmaConfig {
    return .{ .family = .sm90_wgmma, .arch = arch, .shape = shape, .a = .f16, .b = .f16, .c = .f32, .d = .f32, .runtime_fields = &.{.accumulate} };
}

pub fn tcgen05F16(shape: MmaShape) MmaConfig {
    return .{ .family = .sm100_tcgen05, .arch = .sm100, .shape = shape, .a = .f16, .b = .f16, .c = .f32, .d = .f32, .runtime_fields = &.{ .accumulate, .scale_d } };
}

test "arch_op_exact: exact copy validation catches architecture and memory mistakes" {
    var good = cpAsync(.sm80, 128);
    try good.validate();
    var bad_arch = cpAsync(.sm75, 128);
    try std.testing.expectError(Error.UnsupportedArchitecture, bad_arch.validate());
    var bad_space = cpAsync(.sm80, 128);
    bad_space.dst = .gmem;
    try std.testing.expectError(Error.InvalidMemorySpace, bad_space.validate());
}

test "arch_op_exact: MMA type spelling and source manifest counts are available" {
    const mma = wgmmaF16(.sm90, .{ .m = 64, .n = 64, .k = 16 });
    var out: mlir.TextBuffer(512) = .{};
    try mma.writeMlirType(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "sm90_wgmma") != null);
    const summary = ExactOpSummary.fromManifest();
    try std.testing.expect(summary.manifest_records >= 300);
    try std.testing.expect(summary.copy_records >= 50);
    try std.testing.expect(summary.mma_records >= 20);
}

test "arch_op_exact: tcgen05 runtime-field validation" {
    const cfg = tcgen05F16(.{ .m = 128, .n = 128, .k = 64 });
    try cfg.validate();
    var invalid = universalFma(.f32);
    invalid.runtime_fields = &.{.scale_d};
    try std.testing.expectError(Error.InvalidRuntimeField, invalid.validate());
}

pub fn universalCopy(element: Element, bits: u16) CopyConfig {
    return .{ .family = .universal_copy, .arch = .sm70, .element = element, .bits = bits, .src = .generic, .dst = .generic };
}

pub fn cpAsyncCg(bits: u16) CopyConfig {
    return .{ .family = .cp_async, .arch = .sm80, .element = .u8, .bits = bits, .src = .gmem, .dst = .smem, .runtime_fields = &.{.cache_policy} };
}

pub fn cpAsyncCa(bits: u16) CopyConfig {
    return .{ .family = .cp_async, .arch = .sm80, .element = .u8, .bits = bits, .src = .gmem, .dst = .smem, .runtime_fields = &.{ .cache_policy, .byte_mask } };
}

pub fn tmaStore(arch: SmArch, element: Element, bits: u16) CopyConfig {
    return .{ .family = .tma_store, .arch = arch, .element = element, .bits = bits, .src = .smem, .dst = .gmem, .runtime_fields = &.{.mbarrier} };
}

pub fn ldMatrix(arch: SmArch, element: Element, bits: u16) CopyConfig {
    return .{ .family = .ldmatrix, .arch = arch, .element = element, .bits = bits, .src = .smem, .dst = .rmem };
}

pub fn stMatrix(arch: SmArch, element: Element, bits: u16) CopyConfig {
    return .{ .family = .stmatrix, .arch = arch, .element = element, .bits = bits, .src = .rmem, .dst = .smem };
}

pub fn tcgen05Load(bits: u16) CopyConfig {
    return .{ .family = .tcgen05_ld, .arch = .sm100, .element = .u8, .bits = bits, .src = .tmem, .dst = .rmem, .runtime_fields = &.{.fence} };
}

pub fn tcgen05Store(bits: u16) CopyConfig {
    return .{ .family = .tcgen05_st, .arch = .sm100, .element = .u8, .bits = bits, .src = .rmem, .dst = .tmem, .runtime_fields = &.{.fence} };
}

pub fn sm80MmaF16(shape: MmaShape) MmaConfig {
    return .{ .family = .sm80_mma, .arch = .sm80, .shape = shape, .a = .f16, .b = .f16, .c = .f32, .d = .f32, .runtime_fields = &.{.accumulate} };
}

pub fn sm80MmaTF32(shape: MmaShape) MmaConfig {
    return .{ .family = .sm80_mma, .arch = .sm80, .shape = shape, .a = .tf32, .b = .tf32, .c = .f32, .d = .f32, .runtime_fields = &.{.accumulate} };
}

pub fn sm80MmaI8(shape: MmaShape) MmaConfig {
    return .{ .family = .sm80_mma, .arch = .sm80, .shape = shape, .a = .s8, .b = .s8, .c = .s8, .d = .s8, .runtime_fields = &.{.accumulate} };
}

pub fn sm89MmaFP8(shape: MmaShape) MmaConfig {
    return .{ .family = .sm80_mma, .arch = .sm89, .shape = shape, .a = .e4m3, .b = .e4m3, .c = .f32, .d = .f32, .runtime_fields = &.{.accumulate} };
}

pub fn sm90WgmmaF8(shape: MmaShape) MmaConfig {
    return .{ .family = .sm90_wgmma, .arch = .sm90, .shape = shape, .a = .e4m3, .b = .e4m3, .c = .f32, .d = .f32, .runtime_fields = &.{ .accumulate, .scale_a, .scale_b } };
}

pub fn sm90WgmmaI8(shape: MmaShape) MmaConfig {
    return .{ .family = .sm90_wgmma, .arch = .sm90, .shape = shape, .a = .s8, .b = .s8, .c = .s8, .d = .s8, .runtime_fields = &.{.accumulate} };
}

pub fn sm100Tcgen05FP8(shape: MmaShape) MmaConfig {
    return .{ .family = .sm100_tcgen05, .arch = .sm100, .shape = shape, .a = .e4m3, .b = .e4m3, .c = .f32, .d = .f32, .runtime_fields = &.{ .accumulate, .scale_a, .scale_b, .scale_d } };
}

pub fn sourceNamedCopy(module: []const u8, name: []const u8, arch: SmArch) Error!CopyConfig {
    if (std.mem.indexOf(u8, module, "cpasync") != null or std.mem.indexOf(u8, name, "CopyG2S") != null) return cpAsync(arch, 128);
    if (std.mem.indexOf(u8, module, "tma") != null and std.mem.indexOf(u8, name, "Store") != null) return tmaStore(arch, .f32, 128);
    if (std.mem.indexOf(u8, module, "tma") != null) return tmaLoad(arch, .f32, 128);
    if (std.mem.indexOf(u8, name, "LdMatrix") != null or std.mem.indexOf(u8, name, "Ldsm") != null) return ldMatrix(arch, .f16, 128);
    if (std.mem.indexOf(u8, name, "StMatrix") != null or std.mem.indexOf(u8, name, "Stsm") != null) return stMatrix(arch, .f16, 128);
    if (std.mem.indexOf(u8, module, "tcgen05") != null and std.mem.indexOf(u8, name, "Store") != null) return tcgen05Store(128);
    if (std.mem.indexOf(u8, module, "tcgen05") != null) return tcgen05Load(128);
    return universalCopy(.f32, 32);
}

pub fn sourceNamedMma(module: []const u8, name: []const u8, arch: SmArch, shape: MmaShape) Error!MmaConfig {
    _ = module;
    if (std.mem.indexOf(u8, name, "TF32") != null) return sm80MmaTF32(shape);
    if (std.mem.indexOf(u8, name, "I8") != null) return if (arch.atLeast(.sm90)) sm90WgmmaI8(shape) else sm80MmaI8(shape);
    if (std.mem.indexOf(u8, name, "FP8") != null or std.mem.indexOf(u8, name, "F8") != null) return if (arch.atLeast(.sm100)) sm100Tcgen05FP8(shape) else sm90WgmmaF8(shape);
    if (arch.atLeast(.sm100)) return tcgen05F16(shape);
    if (arch.atLeast(.sm90)) return wgmmaF16(arch, shape);
    return sm80MmaF16(shape);
}

test "arch_op_exact: expanded copy constructor families validate" {
    try universalCopy(.f32, 32).validate();
    try cpAsyncCg(32).validate();
    try cpAsyncCa(128).validate();
    try tmaLoad(.sm90, .f16, 128).validate();
    try tmaStore(.sm90, .f16, 128).validate();
    try ldMatrix(.sm75, .f16, 128).validate();
    try stMatrix(.sm75, .f16, 128).validate();
    try tcgen05Load(128).validate();
    try tcgen05Store(128).validate();
    var bad = ldMatrix(.sm70, .f16, 128);
    try std.testing.expectError(Error.UnsupportedArchitecture, bad.validate());
}

test "arch_op_exact: expanded MMA constructor families validate" {
    const s = MmaShape{ .m = 16, .n = 8, .k = 16 };
    try sm80MmaF16(s).validate();
    try sm80MmaTF32(s).validate();
    try sm80MmaI8(s).validate();
    try sm89MmaFP8(s).validate();
    try sm90WgmmaF8(.{ .m = 64, .n = 64, .k = 16 }).validate();
    try sm90WgmmaI8(.{ .m = 64, .n = 64, .k = 32 }).validate();
    try sm100Tcgen05FP8(.{ .m = 128, .n = 128, .k = 64 }).validate();
    const src = try sourceNamedMma("nvgpu.warpgroup.mma", "MmaF8Op", .sm90, .{ .m = 64, .n = 64, .k = 16 });
    try src.validate();
}

test "arch_op_exact: source-name copy dispatch covers catalog shapes" {
    const cp = try sourceNamedCopy("nvgpu.cpasync", "CopyG2SOp", .sm80);
    try cp.validate();
    const tma = try sourceNamedCopy("nvgpu.tma", "CopyBulkTensorTileG2SOp", .sm90);
    try tma.validate();
    const tc = try sourceNamedCopy("nvgpu.tcgen05.copy", "Ld16x128bOp", .sm100);
    try tc.validate();
}
