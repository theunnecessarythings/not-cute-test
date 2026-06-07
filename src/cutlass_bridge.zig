const std = @import("std");
const mlir = @import("mlir_text.zig");
const mlir_harness = @import("mlir_harness.zig");
const runtime_plan = @import("runtime_plan.zig");

pub const Error = mlir_harness.Error || runtime_plan.Error || error{
    InvalidBridgeConfig,
    InvalidPythonExecutable,
    InvalidPackageModule,
    InvalidBridgeScript,
    InvalidSharedLibrary,
    MissingCutlassIrLibrary,
    InvalidDiscoveryJson,
    InvalidBridgeMode,
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
            const dot = std.mem.indexOfScalarPos(u8, self.package_module, start, '.') orelse self.package_module.len;
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
        if (self.python_exe.len == 0 or self.package_module.len == 0) return Error.InvalidDiscoveryJson;
        if (self.package_file.len == 0 or self.mlir_libs_dir.len == 0) return Error.InvalidDiscoveryJson;
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
    compile_options: runtime_plan.CompileOptions = .{},
    flavor: runtime_plan.CompileFlavor = .cutlass_dsl,

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
    if (std.mem.indexOf(u8, path, "_cutlass_ir") == null) return Error.InvalidSharedLibrary;
    const good_suffix = std.mem.endsWith(u8, path, ".so") or
        std.mem.indexOf(u8, path, ".so.") != null or
        (std.mem.indexOf(u8, path, ".cpython") != null and std.mem.endsWith(u8, path, ".so"));
    if (!good_suffix) return Error.InvalidSharedLibrary;
}

pub fn validateDiscoveryJson(text: []const u8) Error!void {
    if (text.len == 0) return Error.InvalidDiscoveryJson;
    const required = [_][]const u8{
        "\"package_module\"",
        "\"package_file\"",
        "\"mlir_libs_dir\"",
        "\"cutlass_ir_so\"",
    };
    for (required) |needle| {
        if (std.mem.indexOf(u8, text, needle) == null) return Error.InvalidDiscoveryJson;
    }
    if (std.mem.indexOf(u8, text, "_cutlass_ir") == null) return Error.MissingCutlassIrLibrary;
}

pub fn discoveryInvocation(config: PythonBridgeConfig) Error!mlir_harness.Invocation {
    try config.validate();
    var inv = mlir_harness.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("discover");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--json");
    return inv;
}

pub fn metadataInvocation(config: PythonBridgeConfig) Error!mlir_harness.Invocation {
    try config.validate();
    var inv = mlir_harness.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("metadata");
    try inv.append("--module");
    try inv.append(config.package_module);
    return inv;
}

pub fn verifyInvocation(config: PythonBridgeConfig, input_mlir: []const u8, pipeline: []const u8) Error!mlir_harness.Invocation {
    try config.validate();
    if (input_mlir.len == 0 or pipeline.len == 0) return Error.InvalidBridgeConfig;
    var inv = mlir_harness.Invocation.init();
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

pub fn lowerInvocation(config: PythonBridgeConfig, input_mlir: []const u8, output_mlir: []const u8, pipeline: []const u8) Error!mlir_harness.Invocation {
    try config.validate();
    if (input_mlir.len == 0 or output_mlir.len == 0 or pipeline.len == 0) return Error.InvalidBridgeConfig;
    var inv = mlir_harness.Invocation.init();
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
    const opts: runtime_plan.CompileOptions = .{};
    try opts.writePipeline(.cutlass_dsl, out);
}

pub fn writeLirCutlassPipeline(out: anytype) Error!void {
    const opts: runtime_plan.CompileOptions = .{};
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

fn writeJsonField(out: anytype, key: []const u8, value: []const u8, comma: bool) Error!void {
    try out.append("  ");
    try out.appendQuotedString(key);
    try out.append(": ");
    try out.appendQuotedString(value);
    if (comma) try out.append(",");
    try out.append("\n");
}

test "cutlass_bridge validates cutlass shared library candidates" {
    try validateCutlassIrSharedLibrary("/pkg/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-310-x86_64-linux-gnu.so");
    try std.testing.expectError(Error.InvalidSharedLibrary, validateCutlassIrSharedLibrary("/tmp/libmlir.so"));
    try std.testing.expectError(Error.MissingCutlassIrLibrary, validateCutlassIrSharedLibrary(""));
}

test "cutlass_bridge builds discovery and verify invocations" {
    const cfg: PythonBridgeConfig = .{ .python_exe = "python", .bridge_script = "tools/cutlass_mlir_bridge.py" };
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
