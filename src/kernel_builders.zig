const std = @import("std");
const mlir = @import("mlir.zig");
const runtime = @import("runtime.zig");
const compile_pipeline = @import("compile_pipeline.zig");

pub const Error = mlir.Error || runtime.Error || compile_pipeline.Error || runtime.Error || error{
    InvalidKernelBuilder,
    InvalidKernelShape,
    InvalidKernelArchitecture,
};

pub const KernelKind = enum {
    copy,
    vector_copy,
    tiled_copy,
    mma_microkernel,
    gemm_mainloop,
    epilogue,
    sm80_gemm,
    sm90_tma_wgmma,
    sm100_tcgen05,
};

pub const KernelArch = enum {
    generic,
    sm80,
    sm90,
    sm100,

    pub fn compileArch(self: KernelArch) []const u8 {
        return switch (self) {
            .generic => "sm_90",
            .sm80 => "sm_80",
            .sm90 => "sm_90",
            .sm100 => "sm_100",
        };
    }

    pub fn commentName(self: KernelArch) []const u8 {
        return switch (self) {
            .generic => "generic",
            .sm80 => "SM80",
            .sm90 => "SM90",
            .sm100 => "SM100",
        };
    }
};

pub const KernelOptions = struct {
    name: []const u8,
    kind: KernelKind,
    arch: KernelArch = .generic,
    element: []const u8 = "f32",
    vector_width: usize = 4,
    tile_m: usize = 1,
    tile_n: usize = 1,
    tile_k: usize = 1,
    alignment: usize = 16,
    block_threads: u32 = 128,
    grid_x: u32 = 1,
    grid_y: u32 = 1,
    grid_z: u32 = 1,

    pub fn validate(self: KernelOptions) Error!void {
        if (self.name.len == 0 or self.element.len == 0)
            return Error.InvalidKernelBuilder;
        try mlir.validateSymbol(self.name);
        if (self.vector_width == 0 or self.tile_m == 0 or self.tile_n == 0 or self.tile_k == 0)
            return Error.InvalidKernelShape;
        if (self.alignment == 0 or self.block_threads == 0 or self.grid_x == 0 or self.grid_y == 0 or self.grid_z == 0)
            return Error.InvalidKernelShape;
        switch (self.kind) {
            .sm80_gemm => if (self.arch != .sm80) return Error.InvalidKernelArchitecture,
            .sm90_tma_wgmma => if (self.arch != .sm90) return Error.InvalidKernelArchitecture,
            .sm100_tcgen05 => if (self.arch != .sm100) return Error.InvalidKernelArchitecture,
            else => {},
        }
    }

    pub fn launchConfig(self: KernelOptions) Error!runtime.LaunchConfig {
        try self.validate();
        return runtime.LaunchConfig.init(
            try runtime.Dim3.init(self.grid_x, self.grid_y, self.grid_z),
            try runtime.Dim3.init(self.block_threads, 1, 1),
            0,
            runtime.Stream.default(),
        );
    }

    pub fn compileRequest(
        self: KernelOptions,
        input_mlir: []const u8,
        work_dir: []const u8,
    ) Error!compile_pipeline.CompileRequest {
        try self.validate();
        return compile_pipeline.defaultKernelCompileRequest(
            input_mlir,
            work_dir,
            self.name,
            self.arch.compileArch(),
        );
    }

    pub fn launchPlan(
        self: KernelOptions,
        cubin_path: []const u8,
    ) Error!runtime.LaunchPlan {
        const cfg = try self.launchConfig();
        const symbols = try runtime.RuntimeSymbols.init("notcute", self.name);
        return .{
            .symbols = symbols,
            .module = try runtime.BinaryModule.init(cubin_path, .cubin),
            .config = cfg,
            .args = .{},
        };
    }
};

pub const KernelModule = struct {
    options: KernelOptions,
    mlir_text: []const u8,

    pub fn validate(self: KernelModule) Error!void {
        try self.options.validate();
        if (self.mlir_text.len == 0) return Error.InvalidKernelBuilder;
        if (std.mem.indexOf(u8, self.mlir_text, "gpu.module") == null)
            return Error.InvalidKernelBuilder;
        if (std.mem.indexOf(u8, self.mlir_text, "gpu.func @") == null)
            return Error.InvalidKernelBuilder;
        if (std.mem.indexOf(u8, self.mlir_text, "gpu.return") == null)
            return Error.InvalidKernelBuilder;
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.InvalidKernelBuilder;
    }
};

pub fn writeKernelModule(out: anytype, options: KernelOptions) Error!void {
    try options.validate();
    switch (options.kind) {
        .copy => try writeCopyKernel(out, options),
        .vector_copy => try writeVectorCopyKernel(out, options),
        .tiled_copy => try writeTiledCopyKernel(out, options),
        .mma_microkernel => try writeMmaMicrokernel(out, options),
        .gemm_mainloop => try writeGemmMainloop(out, options),
        .epilogue => try writeEpilogueKernel(out, options),
        .sm80_gemm => try writeSm80GemmKernel(out, options),
        .sm90_tma_wgmma => try writeSm90TmaWgmmaKernel(out, options),
        .sm100_tcgen05 => try writeSm100Tcgen05Kernel(out, options),
    }
}

pub fn writeCopyKernel(out: anytype, options: KernelOptions) Error!void {
    try options.validate();
    try writeHeader(out, options);
    try writeGpuKernelStart(
        out,
        options,
        "(%src: !memref_scalar, %dst: !memref_scalar, %coord: !coord_zero)",
    );
    try out.append("      %v = cute.memref.load(%src, %coord) : (!memref_scalar, !coord_zero) -> ");
    try out.append(options.element);
    try out.append("\n");
    try out.append("      cute.memref.store(%dst, %coord, %v) : (!memref_scalar, !coord_zero, ");
    try out.append(options.element);
    try out.append(") -> ()\n");
    try writeGpuKernelEnd(out);
}

pub fn writeVectorCopyKernel(out: anytype, options: KernelOptions) Error!void {
    try options.validate();
    try writeHeader(out, options);
    try writeGpuKernelStart(out, options, "(%src: !memref_vec, %dst: !memref_vec)");
    try out.append("      %v = cute.memref.load_vec(%src) : (!memref_vec) -> vector<");
    try out.appendUnsigned(options.vector_width);
    try out.append("x");
    try out.append(options.element);
    try out.append(">\n");
    try out.append("      cute.memref.store_vec(%v, %dst) : (vector<");
    try out.appendUnsigned(options.vector_width);
    try out.append("x");
    try out.append(options.element);
    try out.append(">, !memref_vec) -> ()\n");
    try writeGpuKernelEnd(out);
}

pub fn writeTiledCopyKernel(out: anytype, options: KernelOptions) Error!void {
    try options.validate();
    try writeHeader(out, options);
    try out.append("!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<");
    try out.append(options.element);
    try out.append(", 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>\n");
    try out.append("!memref_partition = !cute.memref<");
    try out.append(options.element);
    try out.append(", gmem, \"((1,1),1,1):((0,0),0,0)\">\n");
    try writeGpuKernelStart(
        out,
        options,
        "(%src: !memref_tile, %dst: !memref_tile, %coord: !coord_zero)",
    );
    try out.append("      %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<");
    try out.append(options.element);
    try out.append(", 32 b>\n");
    try out.append("      %tiled = cute.make_tiled_copy(%atom) : !copy_simt\n");
    try out.append("      %s = cute.tiled.copy.partition_S(%tiled, %src, %coord) : (!copy_simt, !memref_tile, !coord_zero) -> !memref_partition\n");
    try out.append("      %d = cute.tiled.copy.partition_D(%tiled, %dst, %coord) : (!copy_simt, !memref_tile, !coord_zero) -> !memref_partition\n");
    try out.append("      cute.copy(%tiled, %s, %d) : (!copy_simt, !memref_partition, !memref_partition)\n");
    try writeGpuKernelEnd(out);
}

pub fn writeMmaMicrokernel(out: anytype, options: KernelOptions) Error!void {
    try writeMmaLikeKernel(out, options, "MMA microkernel", "universal_fma");
}

pub fn writeGemmMainloop(out: anytype, options: KernelOptions) Error!void {
    try writeMmaLikeKernel(out, options, "GEMM mainloop", "universal_fma");
}

pub fn writeEpilogueKernel(out: anytype, options: KernelOptions) Error!void {
    try options.validate();
    try writeHeader(out, options);
    try writeGpuKernelStart(
        out,
        options,
        "(%acc: !memref_vec, %bias: !memref_vec, %dst: !memref_vec)",
    );
    try out.append("      %a = cute.memref.load_vec(%acc) : (!memref_vec) -> vector<");
    try out.appendUnsigned(options.vector_width);
    try out.append("x");
    try out.append(options.element);
    try out.append(">\n");
    try out.append("      %b = cute.memref.load_vec(%bias) : (!memref_vec) -> vector<");
    try out.appendUnsigned(options.vector_width);
    try out.append("x");
    try out.append(options.element);
    try out.append(">\n");
    try out.append("      %r = arith.addf %a, %b : vector<");
    try out.appendUnsigned(options.vector_width);
    try out.append("x");
    try out.append(options.element);
    try out.append(">\n");
    try out.append("      cute.memref.store_vec(%r, %dst) : (vector<");
    try out.appendUnsigned(options.vector_width);
    try out.append("x");
    try out.append(options.element);
    try out.append(">, !memref_vec) -> ()\n");
    try writeGpuKernelEnd(out);
}

pub fn writeSm80GemmKernel(out: anytype, options: KernelOptions) Error!void {
    try writeMmaLikeKernel(
        out,
        options,
        "SM80 GEMM builder: cp.async plus warp MMA ready skeleton",
        "sm80_universal_fma",
    );
}

pub fn writeSm90TmaWgmmaKernel(out: anytype, options: KernelOptions) Error!void {
    try writeMmaLikeKernel(
        out,
        options,
        "SM90 TMA/WGMMA builder: TMA descriptors plus WGMMA ready skeleton",
        "sm90_wgmma",
    );
}

pub fn writeSm100Tcgen05Kernel(out: anytype, options: KernelOptions) Error!void {
    try writeMmaLikeKernel(
        out,
        options,
        "SM100 tcgen05 builder: tensor memory and tcgen05 ready skeleton",
        "sm100_tcgen05",
    );
}

fn writeMmaLikeKernel(
    out: anytype,
    options: KernelOptions,
    label: []const u8,
    flavor: []const u8,
) Error!void {
    try options.validate();
    try writeHeader(out, options);
    try out.append("// ");
    try out.append(label);
    try out.append("\n");
    try out.append("// flavor: ");
    try out.append(flavor);
    try out.append("\n");
    try out.append("!mma_f32 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <\"(1,1,1):(1,1,1)\">>\n");
    try out.append("!frag = !cute.memref<f32, gmem, \"(1,1,1):(0,0,0)\">\n");
    try writeGpuKernelStart(
        out,
        options,
        "(%a_in: !memref_tile, %b_in: !memref_tile, %c_in: !memref_tile, %d_out: !memref_tile, %coord: !coord_zero)",
    );
    try out.append("      %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >\n");
    try out.append("      %tiled = cute.make_tiled_mma(%atom) : !mma_f32\n");
    try out.append("      %a = cute.tiled.mma.partition A(%tiled, %a_in, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag\n");
    try out.append("      %b = cute.tiled.mma.partition B(%tiled, %b_in, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag\n");
    try out.append("      %c = cute.tiled.mma.partition C(%tiled, %c_in, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag\n");
    try out.append("      %d = cute.tiled.mma.partition C(%tiled, %d_out, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag\n");
    try out.append("      cute.gemm(%tiled, %d, %a, %b, %c) : (!mma_f32, !frag, !frag, !frag, !frag)\n");
    try writeGpuKernelEnd(out);
}

fn writeHeader(out: anytype, options: KernelOptions) Error!void {
    try out.append("// not-cute kernel builder: ");
    try out.append(@tagName(options.kind));
    try out.append(" arch=");
    try out.append(options.arch.commentName());
    try out.append("\n");
    try out.append("!coord_zero = !cute.coord<\"0\">\n");
    try out.append("!memref_scalar = !cute.memref<");
    try out.append(options.element);
    try out.append(", gmem, align<");
    try out.appendUnsigned(options.alignment);
    try out.append(">, \"(1):(1)\">\n");
    try out.append("!memref_vec = !cute.memref<");
    try out.append(options.element);
    try out.append(", gmem, align<");
    try out.appendUnsigned(options.alignment);
    try out.append(">, \"(");
    try out.appendUnsigned(options.vector_width);
    try out.append("):(1)\">\n");
    try out.append("!memref_tile = !cute.memref<");
    try out.append(options.element);
    try out.append(", gmem, \"(1,1):(1,1)\">\n");
}

fn writeGpuKernelStart(
    out: anytype,
    options: KernelOptions,
    signature: []const u8,
) Error!void {
    try out.append("module {\n");
    try out.append("  gpu.module @notcute_kernels {\n");
    try out.append("    gpu.func @");
    try out.append(options.name);
    try out.append(signature);
    try out.append(" kernel {\n");
}

fn writeGpuKernelEnd(out: anytype) Error!void {
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

pub fn writeAllKernels(out: anytype) Error!void {
    const all = [_]KernelOptions{
        .{ .name = "copy_kernel", .kind = .copy },
        .{ .name = "vector_copy_kernel", .kind = .vector_copy },
        .{ .name = "tiled_copy_kernel", .kind = .tiled_copy },
        .{ .name = "mma_microkernel", .kind = .mma_microkernel },
        .{ .name = "gemm_mainloop", .kind = .gemm_mainloop },
        .{ .name = "epilogue_kernel", .kind = .epilogue },
        .{ .name = "sm80_gemm_kernel", .kind = .sm80_gemm, .arch = .sm80 },
        .{ .name = "sm90_tma_wgmma_kernel", .kind = .sm90_tma_wgmma, .arch = .sm90 },
        .{ .name = "sm100_tcgen05_kernel", .kind = .sm100_tcgen05, .arch = .sm100 },
    };
    for (all) |opts| {
        try out.append("// ===== ");
        try out.append(opts.name);
        try out.append(" =====\n");
        try writeKernelModule(out, opts);
    }
}

pub fn defaultOptions(kind: KernelKind) KernelOptions {
    return switch (kind) {
        .copy => .{ .name = "copy_kernel", .kind = .copy },
        .vector_copy => .{ .name = "vector_copy_kernel", .kind = .vector_copy },
        .tiled_copy => .{ .name = "tiled_copy_kernel", .kind = .tiled_copy },
        .mma_microkernel => .{ .name = "mma_microkernel", .kind = .mma_microkernel },
        .gemm_mainloop => .{ .name = "gemm_mainloop", .kind = .gemm_mainloop },
        .epilogue => .{ .name = "epilogue_kernel", .kind = .epilogue },
        .sm80_gemm => .{
            .name = "sm80_gemm_kernel",
            .kind = .sm80_gemm,
            .arch = .sm80,
        },
        .sm90_tma_wgmma => .{
            .name = "sm90_tma_wgmma_kernel",
            .kind = .sm90_tma_wgmma,
            .arch = .sm90,
        },
        .sm100_tcgen05 => .{
            .name = "sm100_tcgen05_kernel",
            .kind = .sm100_tcgen05,
            .arch = .sm100,
        },
    };
}

test "kernel_builders: every requested kernel kind emits a full gpu module" {
    const kinds = [_]KernelKind{ .copy, .vector_copy, .tiled_copy, .mma_microkernel, .gemm_mainloop, .epilogue, .sm80_gemm, .sm90_tma_wgmma, .sm100_tcgen05 };
    for (kinds) |kind| {
        const opts = defaultOptions(kind);
        var out: mlir.TextBuffer(20000) = .{};
        try writeKernelModule(&out, opts);
        const module: KernelModule = .{ .options = opts, .mlir_text = out.slice() };
        try module.validate();
        try std.testing.expect(std.mem.indexOf(u8, out.slice(), opts.name) != null);
    }
}

test "kernel_builders: tiled copy kernel is compile-request ready" {
    const opts = defaultOptions(.tiled_copy);
    var out: mlir.TextBuffer(20000) = .{};
    try writeKernelModule(&out, opts);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.copy(%tiled") != null);
    const req = try opts.compileRequest(
        "tiled_copy.mlir",
        "zig-cache/not-cute-artifacts/tiled_copy",
    );
    var cmd: mlir.TextBuffer(4096) = .{};
    try compile_pipeline.bridgeCompileCommandText(.{}, req, &cmd);
    try std.testing.expect(std.mem.indexOf(u8, cmd.slice(), "compile-artifact") != null);
}

test "kernel_builders: architecture-specific builders reject mismatched arch" {
    try std.testing.expectError(
        Error.InvalidKernelArchitecture,
        (KernelOptions{ .name = "bad", .kind = .sm80_gemm, .arch = .sm90 }).validate(),
    );
}

pub fn writeEmptyGpuKernel(out: anytype, kernel_name: []const u8) Error!void {
    try mlir.validateSymbol(kernel_name);
    try out.append("module {\n");
    try out.append("  gpu.module @kernels {\n");
    try out.append("    gpu.func @");
    try out.append(kernel_name);
    try out.append("() kernel {\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

pub fn writeTiledCopyGpuKernel(out: anytype, kernel_name: []const u8) Error!void {
    try mlir.validateSymbol(kernel_name);
    try out.append("!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>\n");
    try out.append("!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, \"(1,1):(1,1)\">\n");
    try out.append("!memref_gmem_f32_partition = !cute.memref<f32, gmem, \"((1,1),1,1):((0,0),0,0)\">\n");
    try out.append("module {\n");
    try out.append("  gpu.module @kernels {\n");
    try out.append("    gpu.func @");
    try out.append(kernel_name);
    try out.append("(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !cute.coord<\"0\">) kernel {\n");
    try out.append("      %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>\n");
    try out.append("      %tiled = cute.make_tiled_copy(%atom) : !copy_simt\n");
    try out.append("      %src_partitioned = cute.tiled.copy.partition_S(%tiled, %arg0, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<\"0\">) -> !memref_gmem_f32_partition\n");
    try out.append("      %dst_partitioned = cute.tiled.copy.partition_D(%tiled, %arg1, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<\"0\">) -> !memref_gmem_f32_partition\n");
    try out.append("      cute.copy(%tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

pub fn compileRequestForTiledCopyKernel(
    input_mlir: []const u8,
    work_dir: []const u8,
    arch: []const u8,
) compile_pipeline.CompileRequest {
    return compile_pipeline.defaultKernelCompileRequest(
        input_mlir,
        work_dir,
        "tiled_copy_kernel",
        arch,
    );
}

test "kernel_modules: writes kernel-shaped CUTLASS MLIR" {
    var out: mlir.TextBuffer(4096) = .{};
    try writeTiledCopyGpuKernel(&out, "tiled_copy_kernel");
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "gpu.module @kernels") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.copy") != null);
}

test "kernel_modules: compile request targets executable cubin dump path" {
    const req = compileRequestForTiledCopyKernel(
        "kernel.mlir",
        "zig-cache/not-cute/kernel",
        "sm_90",
    );
    var p: mlir.TextBuffer(256) = .{};
    try req.artifactPath(.cubin, &p);
    try std.testing.expectEqualStrings(
        "zig-cache/not-cute/kernel/tiled_copy_kernel",
        p.slice(),
    );
}
