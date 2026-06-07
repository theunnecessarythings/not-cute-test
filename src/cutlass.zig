const atom = @import("atom.zig");
const copy_mma = @import("copy_mma.zig");
const layout = @import("layout.zig");
const mlir = @import("mlir.zig");
const mlir_harness = @import("mlir.zig");
const nvgpu = @import("nvgpu.zig");
const runtime = @import("runtime.zig");
const std = @import("std");
const tensor = @import("tensor.zig");
const typing = @import("typing.zig");

pub const Error = nvgpu.Error || layout.Error || runtime.Error || mlir.Error || error{
    BridgeFailed,
    InvalidAtomType,
    InvalidBridgeConfig,
    InvalidBridgeMode,
    InvalidBridgeScript,
    InvalidCuteMemorySpace,
    InvalidCuteTypePayload,
    InvalidDiscoveryJson,
    InvalidPackageModule,
    InvalidPythonExecutable,
    InvalidFullTiledFixture,
    InvalidRoutedFixture,
    InvalidSharedLibrary,
    InvalidTensorType,
    Invalidcutlass_emitFixture,
    MissingCutlassIrLibrary,
};

pub const BridgeMode = enum {
    /// No external dependencies.  The current Zig golden/string tests continue
    /// to be the only validation performed.
    pure_zig,
    /// Invoke a Python helper which imports the installed CUTLASS DSL package
    /// and uses cutlass._mlir.ir/passmanager.
    python_bridge,
    /// Discover and fingerprint the native _cutlass_ir CPython extension.  This
    /// records the native payload but deliberately does not dlopen it from Zig.
    native_discovery,
};

pub const PythonBridgeConfig = struct {
    python_exe: []const u8 = "python3",
    bridge_script: []const u8 = "tools/cutlass_mlir_bridge.py",
    package_module: []const u8 = "cutlass",
    enable_verifier: bool = true,
    max_output_bytes: usize = 1 << 20,

    pub fn validate(self: PythonBridgeConfig) Error!void {
        if (self.python_exe.len == 0) return Error.InvalidPythonExecutable;
        if (self.bridge_script.len == 0) return Error.InvalidBridgeScript;
        if (self.package_module.len == 0) return Error.InvalidPackageModule;
        // Package module names are Python identifiers separated by dots.
        var start: usize = 0;
        while (start < self.package_module.len) {
            const dot = std.mem.indexOfScalarPos(
                u8,
                self.package_module,
                start,
                '.',
            ) orelse self.package_module.len;
            if (dot == start) return Error.InvalidPackageModule;
            try mlir.validateSymbol(self.package_module[start..dot]);
            start = dot + 1;
        }
    }
};

pub const DiscoveryRecord = struct {
    python_exe: []const u8,
    package_module: []const u8,
    package_version: []const u8,
    package_file: []const u8,
    mlir_libs_dir: []const u8,
    cutlass_ir_so: []const u8,
    cuda_version: ?[]const u8 = null,

    pub fn validate(self: DiscoveryRecord) Error!void {
        if (self.python_exe.len == 0 or self.package_module.len == 0)
            return Error.InvalidDiscoveryJson;
        if (self.package_file.len == 0 or self.mlir_libs_dir.len == 0)
            return Error.InvalidDiscoveryJson;
        try validateCutlassIrSharedLibrary(self.cutlass_ir_so);
    }

    pub fn fingerprint(self: DiscoveryRecord) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.python_exe);
        hasher.update(self.package_module);
        hasher.update(self.package_version);
        hasher.update(self.package_file);
        hasher.update(self.mlir_libs_dir);
        hasher.update(self.cutlass_ir_so);
        if (self.cuda_version) |v| hasher.update(v);
        return hasher.final();
    }

    pub fn writeJson(self: DiscoveryRecord, out: anytype) Error!void {
        try self.validate();
        try out.append("{\n");
        try writeJsonField(out, "python_exe", self.python_exe, true);
        try writeJsonField(out, "package_module", self.package_module, true);
        try writeJsonField(out, "package_version", self.package_version, true);
        try writeJsonField(out, "package_file", self.package_file, true);
        try writeJsonField(out, "mlir_libs_dir", self.mlir_libs_dir, true);
        try writeJsonField(out, "cutlass_ir_so", self.cutlass_ir_so, true);
        if (self.cuda_version) |v| {
            try writeJsonField(out, "cuda_version", v, false);
        } else {
            try out.append("  \"cuda_version\": null\n");
        }
        try out.append("}\n");
    }
};

pub const BridgePlan = struct {
    mode: BridgeMode = .python_bridge,
    config: PythonBridgeConfig = .{},
    compile_options: runtime.CompileOptions = .{},
    flavor: runtime.CompileFlavor = .cutlass_dsl,

    pub fn validate(self: BridgePlan) Error!void {
        switch (self.mode) {
            .pure_zig => {},
            .python_bridge, .native_discovery => try self.config.validate(),
        }
        try self.compile_options.validate();
    }

    pub fn writePipeline(self: BridgePlan, out: anytype) Error!void {
        try self.compile_options.writePipeline(self.flavor, out);
    }

    pub fn writeSummary(self: BridgePlan, out: anytype) Error!void {
        try self.validate();
        try out.append("mode=");
        try out.append(@tagName(self.mode));
        try out.append(" python=");
        try out.append(self.config.python_exe);
        try out.append(" module=");
        try out.append(self.config.package_module);
        try out.append(" script=");
        try out.append(self.config.bridge_script);
        try out.append(" pipeline=");
        var pipeline: mlir.TextBuffer(4096) = .{};
        try self.writePipeline(&pipeline);
        try out.appendQuotedString(pipeline.slice());
    }
};

pub fn validateCutlassIrSharedLibrary(path: []const u8) Error!void {
    if (path.len == 0) return Error.MissingCutlassIrLibrary;
    if (std.mem.indexOf(u8, path, "_cutlass_ir") == null)
        return Error.InvalidSharedLibrary;
    const good_suffix = std.mem.endsWith(u8, path, ".so") or
        std.mem.indexOf(u8, path, ".so.") != null or
        (std.mem.indexOf(u8, path, ".cpython") != null and std.mem.endsWith(u8, path, ".so"));
    if (!good_suffix) return Error.InvalidSharedLibrary;
}

pub fn validateDiscoveryJson(text: []const u8) !void {
    if (text.len == 0) return Error.InvalidDiscoveryJson;
    const required = [_][]const u8{
        "\"package_module\"",
        "\"package_file\"",
        "\"mlir_libs_dir\"",
        "\"cutlass_ir_so\"",
    };
    for (required) |needle| {
        if (std.mem.indexOf(u8, text, needle) == null)
            return Error.InvalidDiscoveryJson;
    }
    if (std.mem.indexOf(u8, text, "_cutlass_ir") == null)
        return Error.MissingCutlassIrLibrary;
}

pub fn discoveryInvocation(config: PythonBridgeConfig) !mlir.Invocation {
    try config.validate();
    var inv = mlir.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("discover");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--json");
    return inv;
}

pub fn metadataInvocation(config: PythonBridgeConfig) !mlir.Invocation {
    try config.validate();
    var inv = mlir.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("metadata");
    try inv.append("--module");
    try inv.append(config.package_module);
    return inv;
}

pub fn verifyInvocation(
    config: PythonBridgeConfig,
    input_mlir: []const u8,
    pipeline: []const u8,
) !mlir.Invocation {
    try config.validate();
    if (input_mlir.len == 0 or pipeline.len == 0) return Error.InvalidBridgeConfig;
    var inv = mlir.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("verify");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--input");
    try inv.append(input_mlir);
    try inv.append("--pipeline");
    try inv.append(pipeline);
    if (config.enable_verifier) try inv.append("--enable-verifier");
    return inv;
}

pub fn lowerInvocation(
    config: PythonBridgeConfig,
    input_mlir: []const u8,
    output_mlir: []const u8,
    pipeline: []const u8,
) !mlir.Invocation {
    try config.validate();
    if (input_mlir.len == 0 or output_mlir.len == 0 or pipeline.len == 0)
        return Error.InvalidBridgeConfig;
    var inv = mlir.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("lower");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--input");
    try inv.append(input_mlir);
    try inv.append("--output");
    try inv.append(output_mlir);
    try inv.append("--pipeline");
    try inv.append(pipeline);
    if (config.enable_verifier) try inv.append("--enable-verifier");
    return inv;
}

pub fn writeDefaultCutlassPipeline(out: anytype) Error!void {
    const opts: runtime.CompileOptions = .{};
    try opts.writePipeline(.cutlass_dsl, out);
}

pub fn writeLirCutlassPipeline(out: anytype) Error!void {
    const opts: runtime.CompileOptions = .{};
    try opts.writePipeline(.cute_experimental_lir, out);
}

pub fn writeDiscoveryRules(out: anytype) Error!void {
    try out.append("The CUTLASS bridge discovers the installed DSL package by importing `cutlass`, which triggers `_cutlass_ir.populate(_cutlass_ir)`. ");
    try out.append("The helper then looks below `cutlass._mlir` for `_mlir_libs/_cutlass_ir.cpython*.so` and records that path instead of dlopening it from Zig. ");
    try out.append("Verification uses `cutlass._mlir.ir.Module.parse`, `cutlass._mlir.passmanager.PassManager.parse`, `enable_verifier`, and `pm.run(module.operation)`.\n");
}

pub fn writeBridgeUsage(out: anytype) Error!void {
    try out.append("python3 tools/cutlass_mlir_bridge.py discover --json\n");
    try out.append("python3 tools/cutlass_mlir_bridge.py verify --input testdata/golden/examples/gemm_skeleton.mlir --pipeline '<pipeline>' --enable-verifier\n");
    try out.append("zig build verify-cutlass -Dcutlass-python=python3\n");
}

fn writeJsonField(
    out: anytype,
    key: []const u8,
    value: []const u8,
    comma: bool,
) Error!void {
    try out.append("  ");
    try out.appendQuotedString(key);
    try out.append(": ");
    try out.appendQuotedString(value);
    if (comma) try out.append(",");
    try out.append("\n");
}

test "cutlass_bridge validates cutlass shared library candidates" {
    try validateCutlassIrSharedLibrary("/pkg/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-310-x86_64-linux-gnu.so");
    try std.testing.expectError(
        Error.InvalidSharedLibrary,
        validateCutlassIrSharedLibrary("/tmp/libmlir.so"),
    );
    try std.testing.expectError(
        Error.MissingCutlassIrLibrary,
        validateCutlassIrSharedLibrary(""),
    );
}

test "cutlass_bridge builds discovery and verify invocations" {
    const cfg: PythonBridgeConfig = .{
        .python_exe = "python",
        .bridge_script = "tools/cutlass_mlir_bridge.py",
    };
    const discover = try discoveryInvocation(cfg);
    try std.testing.expectEqualStrings("python", discover.args()[0]);
    try std.testing.expectEqualStrings("discover", discover.args()[2]);
    try std.testing.expectEqualStrings("--json", discover.args()[discover.argc - 1]);

    var pipeline: mlir.TextBuffer(4096) = .{};
    try writeDefaultCutlassPipeline(&pipeline);
    const verify = try verifyInvocation(cfg, "case.mlir", pipeline.slice());
    try std.testing.expectEqualStrings("verify", verify.args()[2]);
    try std.testing.expect(std.mem.indexOf(u8, verify.args()[8], "cute-to-nvvm") != null);
}

test "cutlass_bridge bridge plan writes source-grounded pipeline" {
    const plan: BridgePlan = .{};
    var out: mlir.TextBuffer(8192) = .{};
    try plan.writeSummary(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "python_bridge") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute-to-nvvm") != null);
}

test "cutlass_bridge discovery record fingerprints and json validates" {
    const rec: DiscoveryRecord = .{
        .python_exe = "python3",
        .package_module = "cutlass",
        .package_version = "4.5.2",
        .package_file = "/site/cutlass/__init__.py",
        .mlir_libs_dir = "/site/cutlass/_mlir/_mlir_libs",
        .cutlass_ir_so = "/site/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-312-x86_64-linux-gnu.so",
        .cuda_version = "12.8",
    };
    try rec.validate();
    try std.testing.expect(rec.fingerprint() != 0);
    var out: mlir.TextBuffer(2048) = .{};
    try rec.writeJson(&out);
    try validateDiscoveryJson(out.slice());
}

test "cutlass_bridge documents discovery rules" {
    var out: mlir.TextBuffer(2048) = .{};
    try writeDiscoveryRules(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "_cutlass_ir.populate") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "PassManager.parse") != null);
}

pub const BridgeStatus = enum {
    skipped,
    passed,
    failed,
};

pub const OwnedBridgeResult = struct {
    term: std.process.Child.Term,
    stdout: []u8,
    stderr: []u8,

    pub fn deinit(self: OwnedBridgeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }

    pub fn success(self: OwnedBridgeResult) bool {
        return switch (self.term) {
            .exited => |code| code == 0,
            else => false,
        };
    }
};

pub const BridgeRunResult = struct {
    status: BridgeStatus,
    term: ?std.process.Child.Term = null,
};

pub fn runInvocation(
    allocator: std.mem.Allocator,
    inv: mlir.Invocation,
    max_output_bytes: usize,
) !OwnedBridgeResult {
    _ = max_output_bytes;
    const io = std.Io.Threaded.global_single_threaded.io();
    var child = try std.process.spawn(io, .{
        .argv = inv.args(),
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    const term = try child.wait(io);
    return .{
        .term = term,
        .stdout = try allocator.dupe(u8, ""),
        .stderr = try allocator.dupe(u8, ""),
    };
}

pub fn discoverMaybe(
    allocator: std.mem.Allocator,
    config: PythonBridgeConfig,
    enabled: bool,
) !BridgeRunResult {
    if (!enabled) return .{ .status = .skipped };
    const inv = try discoveryInvocation(config);
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    return .{
        .status = if (result.success()) .passed else .failed,
        .term = result.term,
    };
}

pub fn verifyMaybe(
    allocator: std.mem.Allocator,
    config: PythonBridgeConfig,
    input_mlir: []const u8,
    pipeline: []const u8,
    enabled: bool,
) !BridgeRunResult {
    if (!enabled) return .{ .status = .skipped };
    const inv = try verifyInvocation(config, input_mlir, pipeline);
    const result = try runInvocation(allocator, inv, config.max_output_bytes);
    defer result.deinit(allocator);
    return .{
        .status = if (result.success()) .passed else .failed,
        .term = result.term,
    };
}

test "cutlass_bridge exec can skip external bridge" {
    const res = try discoverMaybe(std.testing.allocator, .{}, false);
    try std.testing.expect(res.status == .skipped);
}

/// This module contains generated-MLIR spelling helpers. It keeps the
/// old placeholder goldens out of default verifier paths and exposes a
/// parser-aligned emitter for the tensor/copy/MMA forms that the installed
/// CUTLASS DSL package accepts today.
pub const FixtureKind = enum {
    tensor_scalar,
    tensor_vector,
    copy_atom,
    tiled_copy,
    mma_atom,
};

pub const Fixture = struct {
    name: []const u8,
    kind: FixtureKind,
    mlir_text: []const u8,

    pub fn validate(self: Fixture) Error!void {
        if (self.name.len == 0 or self.mlir_text.len == 0)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "module") == null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_load") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_store") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_copy_") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_mma_") != null)
            return Error.Invalidcutlass_emitFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.mma_make_fragment") != null)
            return Error.Invalidcutlass_emitFixture;
    }
};

pub const f32_gmem_1d = "!cute.memref<f32, gmem, align<16>, \"(4):(1)\">";
pub const f32_gmem_scalar = "!cute.memref<f32, gmem, align<16>, \"(1):(1)\">";
pub const f32_gmem_2d = "!cute.memref<f32, gmem, align<16>, \"(1,1):(1,1)\">";
pub const f32_gmem_partitioned = "!cute.memref<f32, gmem, align<16>, \"((1,1),1,1):((0,0),0,0)\">";
pub const f32_rmem_scalar = "!cute.memref<f32, rmem, \"(1):(1)\">";
pub const coord_1d = "!cute.coord<\"(2)\">";
pub const coord_scalar_zero = "!cute.coord<\"0\">";
pub const universal_copy_f32_32b = "!cute_nvgpu.atom.universal_copy<f32, 32 b>";
pub const universal_fma_f32_1x1x1 = "!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >";
pub const tiled_copy_f32_1x1 = "!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>";

pub const tensor_scalar_fixture =
    \\module {
    \\  func.func @tensor_scalar_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: !cute.coord<"(2)">, %arg2: f32) -> f32 {
    \\    %0 = cute.memref.load(%arg0, %arg1) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">, !cute.coord<"(2)">) -> f32
    \\    cute.memref.store(%arg0, %arg1, %arg2) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">, !cute.coord<"(2)">, f32) -> ()
    \\    return %0 : f32
    \\  }
    \\}
    \\
;

pub const tensor_vector_fixture =
    \\module {
    \\  func.func @tensor_vector_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: vector<4xf32>) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">) -> vector<4xf32>
    \\    %1 = arith.addf %0, %arg1 : vector<4xf32>
    \\    cute.memref.store_vec(%1, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, align<16>, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const copy_atom_fixture =
    \\module {
    \\  func.func @copy_atom_case(%arg0: !cute.memref<f32, gmem, align<16>, "(1):(1)">, %arg1: !cute.memref<f32, gmem, align<16>, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%atom, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, align<16>, "(1):(1)">, !cute.memref<f32, gmem, align<16>, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const tiled_copy_fixture =
    \\!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
    \\module {
    \\  func.func @tiled_copy_case(%arg0: !copy_simt, %arg1: !cute.memref<f32, gmem, align<16>, "(1,1):(1,1)">, %arg2: !cute.coord<"0">) {
    \\    %0 = cute.tiled.copy.partition_S(%arg0, %arg1, %arg2) : (!copy_simt, !cute.memref<f32, gmem, align<16>, "(1,1):(1,1)">, !cute.coord<"0">) -> !cute.memref<f32, gmem, align<16>, "((1,1),1,1):((0,0),0,0)">
    \\    return
    \\  }
    \\}
    \\
;

pub const mma_atom_fixture =
    \\module {
    \\  func.func @mma_atom_case(%arg0: !cute.memref<f32, rmem, "(1):(1)">, %arg1: !cute.memref<f32, rmem, "(1):(1)">, %arg2: !cute.memref<f32, rmem, "(1):(1)">, %arg3: !cute.memref<f32, rmem, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%atom, %arg0, %arg1, %arg2, %arg3) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const fixtures = [_]Fixture{
    .{
        .name = "tensor_scalar_case",
        .kind = .tensor_scalar,
        .mlir_text = tensor_scalar_fixture,
    },
    .{
        .name = "tensor_vector_case",
        .kind = .tensor_vector,
        .mlir_text = tensor_vector_fixture,
    },
    .{ .name = "copy_atom_case", .kind = .copy_atom, .mlir_text = copy_atom_fixture },
    .{
        .name = "tiled_copy_case",
        .kind = .tiled_copy,
        .mlir_text = tiled_copy_fixture,
    },
    .{ .name = "mma_atom_case", .kind = .mma_atom, .mlir_text = mma_atom_fixture },
};

pub fn fixtureByName(name: []const u8) ?Fixture {
    for (fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn writeTreePayload(out: anytype, tree: *const layout.Tree) Error!void {
    try writeTreePayloadSub(out, tree, tree.root);
}

fn writeTreePayloadSub(out: anytype, tree: *const layout.Tree, id: u16) Error!void {
    switch (tree.nodes.at(id)) {
        .leaf => |v| try out.appendSigned(v),
        .tuple => |span| {
            try out.append("(");
            for (0..span.len) |i| {
                if (i != 0) try out.append(",");
                try writeTreePayloadSub(out, tree, tree.children.at(span.start + i));
            }
            try out.append(")");
        },
    }
}

pub fn writeLayoutPayload(out: anytype, value: *const layout.Layout) Error!void {
    try writeTreePayload(out, &value.shape);
    try out.append(":");
    try writeTreePayload(out, &value.stride);
}

pub fn writeMemRefTypeForLayout(
    out: anytype,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace,
    alignment: usize,
    value: *const layout.Layout,
) Error!void {
    var payload: mlir.TextBuffer(512) = .{};
    try writeLayoutPayload(&payload, value);
    try writeMemRefType(
        out,
        if (dtype.width == 1) "i8" else dtype.mlir_type,
        memspace.mlirName(),
        alignment,
        payload.slice(),
    );
}

pub fn memRefTypeForLayout(
    dtype: typing.Numeric,
    memspace: typing.AddressSpace,
    alignment: usize,
    value: *const layout.Layout,
) Error!mlir.TextBuffer(512) {
    var out: mlir.TextBuffer(512) = .{};
    try writeMemRefTypeForLayout(&out, dtype, memspace, alignment, value);
    return out;
}

pub fn writeCoordPayloadFromScalar(out: anytype, offset: layout.Scalar) Error!void {
    try out.appendSigned(offset);
}

pub fn writeCoordTypeFromScalar(out: anytype, offset: layout.Scalar) Error!void {
    var payload: mlir.TextBuffer(96) = .{};
    try writeCoordPayloadFromScalar(&payload, offset);
    try writeCoordType(out, payload.slice());
}

pub fn makeCoordFromScalar(
    builder: anytype,
    offset: layout.Scalar,
) Error!struct { value: mlir.Value, ty: mlir.Type } {
    var ty_buf: mlir.TextBuffer(128) = .{};
    try writeCoordTypeFromScalar(&ty_buf, offset);
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(ty_buf.slice())}, result.id);
    try builder.append("cute.make_coord() : () -> ");
    try builder.append(ty_buf.slice());
    try builder.newline();
    return .{ .value = result, .ty = mlir.Type.raw(ty_buf.slice()) };
}

pub fn emitMemrefLoad(
    builder: anytype,
    memref: mlir.Value,
    coord: mlir.Value,
    memref_ty: mlir.Type,
    coord_ty: mlir.Type,
    elem_ty: mlir.Type,
) Error!mlir.Value {
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{elem_ty}, result.id);
    try builder.append("cute.memref.load(");
    try memref.writeTo(builder);
    try builder.append(", ");
    try coord.writeTo(builder);
    try builder.append(") : (");
    try builder.append(memref_ty.text);
    try builder.append(", ");
    try builder.append(coord_ty.text);
    try builder.append(") -> ");
    try builder.append(elem_ty.text);
    try builder.newline();
    return result;
}

pub fn emitMemrefStore(
    builder: anytype,
    memref: mlir.Value,
    coord: mlir.Value,
    data: mlir.Value,
    memref_ty: mlir.Type,
    coord_ty: mlir.Type,
    elem_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.memref.store(");
    try memref.writeTo(builder);
    try builder.append(", ");
    try coord.writeTo(builder);
    try builder.append(", ");
    try data.writeTo(builder);
    try builder.append(") : (");
    try builder.append(memref_ty.text);
    try builder.append(", ");
    try builder.append(coord_ty.text);
    try builder.append(", ");
    try builder.append(elem_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn emitMemrefLoadVec(
    builder: anytype,
    memref: mlir.Value,
    memref_ty: mlir.Type,
    result_ty: mlir.Type,
) Error!mlir.Value {
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{result_ty}, result.id);
    try builder.append("cute.memref.load_vec(");
    try memref.writeTo(builder);
    try builder.append(") : (");
    try builder.append(memref_ty.text);
    try builder.append(") -> ");
    try builder.append(result_ty.text);
    try builder.newline();
    return result;
}

pub fn emitMemrefStoreVec(
    builder: anytype,
    data: mlir.Value,
    memref: mlir.Value,
    data_ty: mlir.Type,
    memref_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.memref.store_vec(");
    try data.writeTo(builder);
    try builder.append(", ");
    try memref.writeTo(builder);
    try builder.append(") : (");
    try builder.append(data_ty.text);
    try builder.append(", ");
    try builder.append(memref_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn emitMakeUniversalCopyAtom(
    builder: anytype,
    dtype: typing.Numeric,
    bits: usize,
) Error!struct { value: mlir.Value, ty: mlir.Type } {
    var ty_buf: mlir.TextBuffer(256) = .{};
    try writeUniversalCopyAtomType(&ty_buf, dtype.mlir_type, bits);
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(ty_buf.slice())}, result.id);
    try builder.append("cute.make_atom() : () -> ");
    try builder.append(ty_buf.slice());
    try builder.newline();
    return .{ .value = result, .ty = mlir.Type.raw(ty_buf.slice()) };
}

pub fn emitMakeUniversalFmaAtom(
    builder: anytype,
    dtype: typing.Numeric,
    m: usize,
    n: usize,
    k: usize,
) Error!struct { value: mlir.Value, ty: mlir.Type } {
    var ty_buf: mlir.TextBuffer(256) = .{};
    try writeUniversalFmaAtomType(&ty_buf, dtype.mlir_type, m, n, k);
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(ty_buf.slice())}, result.id);
    try builder.append("cute.make_atom() : () -> ");
    try builder.append(ty_buf.slice());
    try builder.newline();
    return .{ .value = result, .ty = mlir.Type.raw(ty_buf.slice()) };
}

pub fn emitCopyAtomCall(
    builder: anytype,
    atom_value: mlir.Value,
    atom_ty: mlir.Type,
    src: mlir.Value,
    dst: mlir.Value,
    src_ty: mlir.Type,
    dst_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.copy_atom_call(");
    try atom_value.writeTo(builder);
    try builder.append(", ");
    try src.writeTo(builder);
    try builder.append(", ");
    try dst.writeTo(builder);
    try builder.append(") : (");
    try builder.append(atom_ty.text);
    try builder.append(", ");
    try builder.append(src_ty.text);
    try builder.append(", ");
    try builder.append(dst_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn emitMmaAtomCall(
    builder: anytype,
    atom_value: mlir.Value,
    atom_ty: mlir.Type,
    d: mlir.Value,
    a: mlir.Value,
    b: mlir.Value,
    c: mlir.Value,
    d_ty: mlir.Type,
    a_ty: mlir.Type,
    b_ty: mlir.Type,
    c_ty: mlir.Type,
) Error!void {
    try builder.writeResultPrefixFor(&.{}, 0);
    try builder.append("cute.mma_atom_call(");
    try atom_value.writeTo(builder);
    try builder.append(", ");
    try d.writeTo(builder);
    try builder.append(", ");
    try a.writeTo(builder);
    try builder.append(", ");
    try b.writeTo(builder);
    try builder.append(", ");
    try c.writeTo(builder);
    try builder.append(") : (");
    try builder.append(atom_ty.text);
    try builder.append(", ");
    try builder.append(d_ty.text);
    try builder.append(", ");
    try builder.append(a_ty.text);
    try builder.append(", ");
    try builder.append(b_ty.text);
    try builder.append(", ");
    try builder.append(c_ty.text);
    try builder.append(") -> ()");
    try builder.newline();
}

pub fn writeCoordType(out: anytype, coord: []const u8) Error!void {
    try validateCutePayload(coord);
    try out.append("!cute.coord<");
    try out.appendQuotedString(coord);
    try out.append(">");
}

pub fn writeMemRefType(
    out: anytype,
    elem: []const u8,
    memory_space: []const u8,
    alignment: usize,
    layout_text: []const u8,
) Error!void {
    try validateElementType(elem);
    try validateMemorySpace(memory_space);
    try validateCutePayload(layout_text);
    if (alignment == 0) return Error.InvalidCuteTypePayload;
    try out.append("!cute.memref<");
    try out.append(elem);
    try out.append(", ");
    try out.append(memory_space);
    if (alignment != 4) {
        try out.append(", align<");
        try out.appendUnsigned(alignment);
        try out.append(">");
    }
    try out.append(", ");
    try out.appendQuotedString(layout_text);
    try out.append(">");
}

pub fn writeUniversalCopyAtomType(
    out: anytype,
    elem: []const u8,
    bits: usize,
) Error!void {
    try validateElementType(elem);
    if (bits == 0) return Error.InvalidAtomType;
    try out.append("!cute_nvgpu.atom.universal_copy<");
    try out.append(elem);
    try out.append(", ");
    try out.appendUnsigned(bits);
    try out.append(" b>");
}

pub fn writeUniversalFmaAtomType(
    out: anytype,
    elem: []const u8,
    m: usize,
    n: usize,
    k: usize,
) Error!void {
    try validateElementType(elem);
    if (m == 0 or n == 0 or k == 0) return Error.InvalidAtomType;
    try out.append("!cute_nvgpu.atom.universal_fma<");
    try out.appendUnsigned(m);
    try out.append("x");
    try out.appendUnsigned(n);
    try out.append("x");
    try out.appendUnsigned(k);
    try out.append(", (");
    try out.append(elem);
    try out.append(", ");
    try out.append(elem);
    try out.append(") -> ");
    try out.append(elem);
    try out.append(" >");
}

pub fn writeTiledCopyType(
    out: anytype,
    copy_atom_type: []const u8,
    layout_copy_tv: []const u8,
    tiler_mn: []const u8,
) Error!void {
    if (std.mem.indexOf(u8, copy_atom_type, "!cute_nvgpu.atom.universal_copy") == null)
        return Error.InvalidAtomType;
    try validateCutePayload(layout_copy_tv);
    try validateTilePayload(tiler_mn);
    try out.append("!cute.tiled_copy<");
    try out.append(copy_atom_type);
    try out.append(", layout_copy_tv = <");
    try out.appendQuotedString(layout_copy_tv);
    try out.append(">, tiler_mn = <");
    try out.appendQuotedString(tiler_mn);
    try out.append(">>");
}

pub fn validateCutePayload(payload: []const u8) Error!void {
    if (payload.len == 0) return Error.InvalidCuteTypePayload;
    for (payload) |c| switch (c) {
        '0'...'9', 'a'...'z', 'A'...'Z', '(', ')', ',', ':', '@', '_', '-', '+', '*', '/', ' ', '.', '[', ']', ';' => {},
        else => return Error.InvalidCuteTypePayload,
    };
    if (std.mem.indexOfScalar(u8, payload, '"') != null)
        return Error.InvalidCuteTypePayload;
}

pub fn validateElementType(elem: []const u8) Error!void {
    const valid = [_][]const u8{ "i1", "i8", "i16", "i32", "i64", "f16", "bf16", "tf32", "f32", "f64" };
    for (valid) |candidate| {
        if (std.mem.eql(u8, elem, candidate)) return;
    }
    return Error.InvalidMlirType;
}

pub fn validateMemorySpace(memory_space: []const u8) Error!void {
    const valid = [_][]const u8{ "generic", "gmem", "smem", "rmem", "tmem" };
    for (valid) |candidate| {
        if (std.mem.eql(u8, memory_space, candidate)) return;
    }
    return Error.InvalidCuteMemorySpace;
}

fn validateTilePayload(payload: []const u8) Error!void {
    if (payload.len == 0) return Error.InvalidCuteTypePayload;
    for (payload) |c| switch (c) {
        '0'...'9', 'a'...'z', 'A'...'Z', '(', ')', '[', ']', ',', ':', ';', '@', '_', '-', '+', '*', '/', ' ', '.' => {},
        else => return Error.InvalidCuteTypePayload,
    };
    if (std.mem.indexOfScalar(u8, payload, '"') != null)
        return Error.InvalidCuteTypePayload;
}

pub fn emitTensorScalarModule(out: anytype) Error!void {
    try out.append(tensor_scalar_fixture);
}

pub fn emitTensorVectorModule(out: anytype) Error!void {
    try out.append(tensor_vector_fixture);
}

pub fn emitCopyAtomModule(out: anytype) Error!void {
    try out.append(copy_atom_fixture);
}

pub fn emitTiledCopyModule(out: anytype) Error!void {
    try out.append(tiled_copy_fixture);
}

pub fn emitMmaAtomModule(out: anytype) Error!void {
    try out.append(mma_atom_fixture);
}

pub fn writeAllFixtures(out: anytype) Error!void {
    for (fixtures) |fixture| {
        try fixture.validate();
        try out.append("// ----- ");
        try out.append(fixture.name);
        try out.append(" -----\n");
        try out.append(fixture.mlir_text);
    }
}

pub fn writeStatus(out: anytype) Error!void {
    try out.append("CUTLASS emission helpers generate tensor/copy/MMA MLIR spelling using parser-aligned Cute syntax. ");
    try out.append("Fixtures use !cute.memref, !cute_nvgpu.atom.universal_copy, !cute.tiled_copy, and !cute_nvgpu.atom.universal_fma; ");
    try out.append("integration audit later replaced the remaining default example/golden placeholders with parser-aligned forms.\n");
}

test "cutlass_emit fixtures are structural and placeholder-free" {
    for (fixtures) |fixture| {
        try fixture.validate();
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "module") != null);
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "!cute.tensor") == null);
    }
}

test "cutlass_emit writes parser-aligned atom and tiled-copy types" {
    var out: mlir.TextBuffer(512) = .{};
    try writeUniversalCopyAtomType(&out, "f32", 32);
    try std.testing.expectEqualStrings(universal_copy_f32_32b, out.slice());
    out.clear();
    try writeUniversalFmaAtomType(&out, "f32", 1, 1, 1);
    try std.testing.expectEqualStrings(universal_fma_f32_1x1x1, out.slice());
    out.clear();
    try writeTiledCopyType(&out, universal_copy_f32_32b, "(1,1):(1,1)", "[1:0;1:0]");
    try std.testing.expectEqualStrings(tiled_copy_f32_1x1, out.slice());
}

test "cutlass_emit tensor spelling uses dot ops and parenthesized operands" {
    try std.testing.expect(std.mem.indexOf(u8, tensor_vector_fixture, "cute.memref.load_vec(%arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, tensor_vector_fixture, "cute.memref.store_vec(%1, %arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, tensor_vector_fixture, "cute.memref_load_vec") == null);
}

test "cutlass_emit copy and MMA fixtures use real CUTLASS atom types" {
    try std.testing.expect(std.mem.indexOf(u8, copy_atom_fixture, universal_copy_f32_32b) != null);
    try std.testing.expect(std.mem.indexOf(u8, tiled_copy_fixture, tiled_copy_f32_1x1) != null);
    try std.testing.expect(std.mem.indexOf(u8, mma_atom_fixture, universal_fma_f32_1x1x1) != null);
    try std.testing.expect(std.mem.indexOf(u8, mma_atom_fixture, "cute.mma_atom_call") != null);
}

test "cutlass_emit status points to parser-aligned follow-up" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "integration audit") != null);
}

pub const RoutedError = Error || copy_mma.Error || tensor.Error || mlir.Error;

pub const RoutedFixtureKind = enum {
    tensor_vector,
    copy_atom,
    mma_atom,
};

pub const RoutedFixture = struct {
    name: []const u8,
    kind: RoutedFixtureKind,
    mlir_text: []const u8,

    pub fn validate(self: RoutedFixture) !void {
        if (self.name.len == 0 or self.mlir_text.len == 0)
            return Error.InvalidRoutedFixture;
        try mlir.validateGeneratedMlir(self.mlir_text);
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_load") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.memref_store") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_copy_") != null)
            return Error.InvalidRoutedFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_mma_") != null)
            return Error.InvalidRoutedFixture;
    }
};

pub const routed_tensor_vector_fixture =
    \\module {
    \\  func.func @routed_tensor_vector(%arg0: !cute.memref<f32, gmem, "(4):(1)">) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, "(4):(1)">) -> vector<4xf32>
    \\    cute.memref.store_vec(%0, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const routed_copy_atom_fixture =
    \\module {
    \\  func.func @routed_copy_atom(%arg0: !cute.memref<f32, gmem, "(1):(1)">, %arg1: !cute.memref<f32, gmem, "(1):(1)">) {
    \\    %0 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%0, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, "(1):(1)">, !cute.memref<f32, gmem, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const routed_mma_atom_fixture =
    \\module {
    \\  func.func @routed_mma_atom(%arg0: !cute.memref<f32, generic, "(1):(1)">, %arg1: !cute.memref<f32, generic, "(1):(1)">, %arg2: !cute.memref<f32, generic, "(1):(1)">, %arg3: !cute.memref<f32, generic, "(1):(1)">) {
    \\    %0 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%0, %arg3, %arg0, %arg1, %arg2) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const routed_fixtures = [_]RoutedFixture{
    .{
        .name = "cutlass_routed_tensor_vector",
        .kind = .tensor_vector,
        .mlir_text = routed_tensor_vector_fixture,
    },
    .{
        .name = "cutlass_routed_copy_atom",
        .kind = .copy_atom,
        .mlir_text = routed_copy_atom_fixture,
    },
    .{
        .name = "cutlass_routed_mma_atom",
        .kind = .mma_atom,
        .mlir_text = routed_mma_atom_fixture,
    },
};

pub fn routedFixtureByName(name: []const u8) ?RoutedFixture {
    for (routed_fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn emitRoutedTensorVectorModule(out: anytype) RoutedError!void {
    var builder: mlir.Builder(4096) = .{};
    const layout_value = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{4}));
    const meta = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(0) },
        layout_value,
        typing.Float32,
        .gmem,
    );
    var memref_ty_buf: mlir.TextBuffer(512) = .{};
    try meta.cutlassTensorTypeText(&memref_ty_buf);
    try builder.beginModule();
    try builder.beginFunc(
        "routed_tensor_vector",
        &.{mlir.Type.raw(memref_ty_buf.slice())},
        null,
    );
    const tv = tensor.TensorValue.init(
        meta,
        mlir.Value.arg(0),
        memref_ty_buf.slice(),
    );
    const loaded = try tv.load(&builder, null, null);
    try tv.store(&builder, loaded, null);
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    const text = try builder.finish();
    try out.append(text);
}

fn makeCopyAtom() RoutedError!atom.CopyAtom {
    const thr = layout.makeCompactLayout(.{1});
    const tv = layout.makeCompactLayout(.{ 1, 1 });
    var tr: atom.Trait = .{ .name = "routed_copy_trait", .thr_id = thr };
    tr = tr.withCopyLayouts(tv, tv);
    const desc = atom.OpDescriptor.copyTyped(
        "CopyUniversalOp",
        "generic",
        "simt.sync.copy",
        typing.Float32,
        .gmem,
        .gmem,
        32,
        &.{},
    );
    return atom.makeCopyAtom(desc, tr);
}

pub fn emitRoutedCopyAtomModule(out: anytype) RoutedError!void {
    var builder: mlir.Builder(4096) = .{};
    const layout_value = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{1}));
    const src_meta = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(0) },
        layout_value,
        typing.Float32,
        .gmem,
    );
    const dst_meta = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(1) },
        layout_value,
        typing.Float32,
        .gmem,
    );
    var ty_buf: mlir.TextBuffer(512) = .{};
    try src_meta.cutlassTensorTypeText(&ty_buf);
    try builder.beginModule();
    try builder.beginFunc(
        "routed_copy_atom",
        &.{ mlir.Type.raw(ty_buf.slice()), mlir.Type.raw(ty_buf.slice()) },
        null,
    );
    const src = tensor.TensorValue.init(
        src_meta,
        mlir.Value.arg(0),
        ty_buf.slice(),
    );
    const dst = tensor.TensorValue.init(
        dst_meta,
        mlir.Value.arg(1),
        ty_buf.slice(),
    );
    _ = try copy_mma.lowerCopyAtom(&builder, try makeCopyAtom(), src, dst, null);
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    const text = try builder.finish();
    try out.append(text);
}

pub fn emitRoutedMmaAtomModule(out: anytype) RoutedError!void {
    var builder: mlir.Builder(8192) = .{};
    const layout_value = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{1}));
    const meta_a = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(0) },
        layout_value,
        typing.Float32,
        .generic,
    );
    const meta_b = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(1) },
        layout_value,
        typing.Float32,
        .generic,
    );
    const meta_c = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(2) },
        layout_value,
        typing.Float32,
        .generic,
    );
    const meta_d = try tensor.TensorMeta.init(
        .{ .mlir_value = mlir.Value.arg(3) },
        layout_value,
        typing.Float32,
        .generic,
    );
    var ty_buf: mlir.TextBuffer(512) = .{};
    try meta_a.cutlassTensorTypeText(&ty_buf);
    try builder.beginModule();
    try builder.beginFunc(
        "routed_mma_atom",
        &.{
            mlir.Type.raw(ty_buf.slice()),
            mlir.Type.raw(ty_buf.slice()),
            mlir.Type.raw(ty_buf.slice()),
            mlir.Type.raw(ty_buf.slice()),
        },
        null,
    );
    const a = tensor.TensorValue.init(meta_a, mlir.Value.arg(0), ty_buf.slice());
    const b = tensor.TensorValue.init(meta_b, mlir.Value.arg(1), ty_buf.slice());
    const c = tensor.TensorValue.init(meta_c, mlir.Value.arg(2), ty_buf.slice());
    const d = tensor.TensorValue.init(meta_d, mlir.Value.arg(3), ty_buf.slice());
    _ = try copy_mma.lowerMmaAtom(
        &builder,
        try nvgpu.universalMma(typing.Float32),
        d,
        a,
        b,
        c,
    );
    try builder.ret(&.{}, &.{});
    try builder.endFunc();
    try builder.endModule();
    const text = try builder.finish();
    try out.append(text);
}

pub fn emitByName(name: []const u8, out: anytype) RoutedError!void {
    if (std.mem.eql(u8, name, "cutlass_routed_tensor_vector"))
        return emitRoutedTensorVectorModule(out);
    if (std.mem.eql(u8, name, "cutlass_routed_copy_atom"))
        return emitRoutedCopyAtomModule(out);
    if (std.mem.eql(u8, name, "cutlass_routed_mma_atom")) return emitRoutedMmaAtomModule(out);
    return Error.InvalidRoutedFixture;
}

pub fn writeAllGenerated(out: anytype) Error!void {
    inline for (.{
        "cutlass_routed_tensor_vector",
        "cutlass_routed_copy_atom",
        "cutlass_routed_mma_atom",
    }) |name| {
        try out.append("// ----- ");
        try out.append(name);
        try out.append(" -----\n");
        try emitByName(name, out);
    }
}

pub fn writeRoutedStatus(out: anytype) Error!void {
    try out.append("Routed CUTLASS emission connects tensor vector load/store and copy/MMA atom lowering to parser-aligned emitters. ");
    try out.append("Generated modules use !cute.memref, cute.memref.load_vec/store_vec, cute.copy_atom_call, and cute.mma_atom_call forms accepted by the installed CUTLASS DSL parser.\n");
}

test "cutlass_routed static routed fixtures are placeholder-free" {
    for (routed_fixtures) |fixture| {
        try fixture.validate();
    }
}

test "cutlass_routed generated tensor module uses parser-aligned memref vector ops" {
    var out: mlir.TextBuffer(4096) = .{};
    try emitRoutedTensorVectorModule(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.memref.load_vec(%arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.memref.store_vec(%0, %arg0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") == null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.memref_load_vec") == null);
}

test "cutlass_routed generated copy module uses parser-aligned atom call" {
    var out: mlir.TextBuffer(4096) = .{};
    try emitRoutedCopyAtomModule(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.make_atom()") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.copy_atom_call(%0, %arg0, %arg1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") == null);
}

test "cutlass_routed generated mma module uses parser-aligned atom call" {
    var out: mlir.TextBuffer(8192) = .{};
    try emitRoutedMmaAtomModule(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute_nvgpu.atom.universal_fma<1x1x1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.mma_atom_call(%0, %arg3, %arg0, %arg1, %arg2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "!cute.tensor") == null);
}

test "cutlass_routed status names routing boundary" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeRoutedStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "connects tensor") != null);
}

pub const FullTiledFixtureKind = enum {
    tiled_copy_full,
    tiled_mma_full,
};

pub const FullTiledFixture = struct {
    name: []const u8,
    kind: FullTiledFixtureKind,
    mlir_text: []const u8,

    pub fn validate(self: FullTiledFixture) Error!void {
        if (self.name.len == 0 or self.mlir_text.len == 0)
            return Error.InvalidFullTiledFixture;
        try mlir.validateGeneratedMlir(self.mlir_text);
        if (std.mem.indexOf(u8, self.mlir_text, "!cute.tensor") != null)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_copy_") != null)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled_mma_") != null)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled.copy.partition_S") == null and self.kind == .tiled_copy_full)
            return Error.InvalidFullTiledFixture;
        if (std.mem.indexOf(u8, self.mlir_text, "cute.tiled.mma.partition") == null and self.kind == .tiled_mma_full)
            return Error.InvalidFullTiledFixture;
    }
};

pub const tiled_copy_alias = "!copy_simt";
pub const tiled_copy_type = "!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>";
pub const memref_gmem_1x1_alias = "!memref_gmem_f32_1x1";
pub const memref_gmem_1x1_type = "!cute.memref<f32, gmem, \"(1,1):(1,1)\">";
pub const memref_gmem_partition_alias = "!memref_gmem_f32_partition";
pub const memref_gmem_partition_type = "!cute.memref<f32, gmem, \"((1,1),1,1):((0,0),0,0)\">";

pub const tiled_mma_alias = "!mma_f32_1x1x1";
pub const tiled_mma_type = "!cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <\"(1,1,1):(1,1,1)\">>";
pub const memref_generic_1x1_alias = "!memref_generic_f32_1x1";
pub const memref_generic_1x1_type = "!cute.memref<f32, generic, \"(1,1):(1,1)\">";
pub const memref_generic_fragment_alias = "!memref_generic_f32_frag";
pub const memref_generic_fragment_type = "!cute.memref<f32, generic, \"(1,1,1):(0,0,0)\">";

pub const full_tiled_copy_fixture =
    \\!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
    \\!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, "(1,1):(1,1)">
    \\!memref_gmem_f32_partition = !cute.memref<f32, gmem, "((1,1),1,1):((0,0),0,0)">
    \\module {
    \\  func.func @tiled_emit_full_tiled_copy(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !cute.coord<"0">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    %tiled = cute.make_tiled_copy(%atom) : !copy_simt
    \\    %src_partitioned = cute.tiled.copy.partition_S(%tiled, %arg0, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    \\    %dst_partitioned = cute.tiled.copy.partition_D(%tiled, %arg1, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    \\    cute.copy(%tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)
    \\    return
    \\  }
    \\}
    \\
;

pub const full_tiled_mma_fixture =
    \\!mma_f32_1x1x1 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(1,1,1):(1,1,1)">>
    \\!memref_generic_f32_1x1 = !cute.memref<f32, generic, "(1,1):(1,1)">
    \\!memref_generic_f32_frag = !cute.memref<f32, generic, "(1,1,1):(0,0,0)">
    \\module {
    \\  func.func @tiled_emit_full_tiled_mma(%arg0: !memref_generic_f32_1x1, %arg1: !memref_generic_f32_1x1, %arg2: !memref_generic_f32_1x1, %arg3: !memref_generic_f32_1x1, %arg4: !cute.coord<"0">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    %tiled = cute.make_tiled_mma(%atom) : !mma_f32_1x1x1
    \\    %a = cute.tiled.mma.partition A(%tiled, %arg0, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %b = cute.tiled.mma.partition B(%tiled, %arg1, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %c = cute.tiled.mma.partition C(%tiled, %arg2, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %d = cute.tiled.mma.partition C(%tiled, %arg3, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    cute.gemm(%tiled, %d, %a, %b, %c) : (!mma_f32_1x1x1, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag)
    \\    return
    \\  }
    \\}
    \\
;

pub const full_tiled_fixtures = [_]FullTiledFixture{
    .{
        .name = "tiled_emit_full_tiled_copy",
        .kind = .tiled_copy_full,
        .mlir_text = full_tiled_copy_fixture,
    },
    .{
        .name = "tiled_emit_full_tiled_mma",
        .kind = .tiled_mma_full,
        .mlir_text = full_tiled_mma_fixture,
    },
};

pub fn fullTiledFixtureByName(name: []const u8) ?FullTiledFixture {
    for (full_tiled_fixtures) |fixture| {
        if (std.mem.eql(u8, fixture.name, name)) return fixture;
    }
    return null;
}

pub fn emitFullTiledCopyModule(out: anytype) Error!void {
    try out.append(full_tiled_copy_fixture);
}

pub fn emitFullTiledMmaModule(out: anytype) Error!void {
    try out.append(full_tiled_mma_fixture);
}

pub fn emitFullTiledByName(name: []const u8, out: anytype) Error!void {
    if (std.mem.eql(u8, name, "tiled_emit_full_tiled_copy"))
        return emitFullTiledCopyModule(out);
    if (std.mem.eql(u8, name, "tiled_emit_full_tiled_mma"))
        return emitFullTiledMmaModule(out);
    return Error.InvalidFullTiledFixture;
}

pub fn writeAllFullTiledGenerated(out: anytype) Error!void {
    for (full_tiled_fixtures) |fixture| {
        try out.append("// ----- ");
        try out.append(fixture.name);
        try out.append(" -----\n");
        try out.append(fixture.mlir_text);
    }
}

pub fn writeFullTiledStatus(out: anytype) Error!void {
    try out.append("Tiled emission adds CUTLASS parser-verified full tiled-copy and tiled-MMA modules. ");
    try out.append("The copy path constructs a tiled copy, partitions source and destination tensors, and calls cute.copy. ");
    try out.append("The MMA path constructs a tiled MMA, partitions A/B/C/D fragments, and calls cute.gemm with the tiled MMA handle.\n");
}

test "tiled_emit full tiled fixtures are structural and placeholder-free" {
    for (full_tiled_fixtures) |fixture| {
        try fixture.validate();
        try std.testing.expect(std.mem.indexOf(u8, fixture.mlir_text, "!cute.tensor") == null);
    }
}

test "tiled_emit full tiled copy composes make_tiled_copy partition S D and cute.copy" {
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.make_tiled_copy(%atom) : !copy_simt") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.tiled.copy.partition_S") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.tiled.copy.partition_D") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_copy_fixture, "cute.copy(%tiled, %src_partitioned, %dst_partitioned)") != null);
}

test "tiled_emit full tiled mma composes make_tiled_mma partitions and cute.gemm" {
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.make_tiled_mma(%atom) : !mma_f32_1x1x1") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.tiled.mma.partition A") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.tiled.mma.partition B") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.tiled.mma.partition C") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_tiled_mma_fixture, "cute.gemm(%tiled, %d, %a, %b, %c)") != null);
}

test "tiled_emit fixture lookup and emission" {
    const copy = fullTiledFixtureByName("tiled_emit_full_tiled_copy") orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(FullTiledFixtureKind.tiled_copy_full, copy.kind);
    var out: mlir.TextBuffer(12000) = .{};
    try emitFullTiledByName("tiled_emit_full_tiled_mma", &out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "@tiled_emit_full_tiled_mma") != null);
}

test "tiled_emit status names parser-verified tiled scope" {
    var out: mlir.TextBuffer(1024) = .{};
    try writeFullTiledStatus(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "full tiled-copy") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.gemm") != null);
}
