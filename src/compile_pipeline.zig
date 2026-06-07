const std = @import("std");
const mlir = @import("mlir.zig");
const runtime = @import("runtime.zig");
const cutlass = @import("cutlass.zig");
const mlir_harness = @import("mlir.zig");
const execution = @import("execution.zig");

pub const Error = runtime.Error || cutlass.Error || mlir.Error || error{
    InvalidCompileRequest,
    InvalidArtifactKind,
    InvalidArtifactSet,
    InvalidPipelineKind,
    InvalidOutputPath,
    MissingArtifactPath,
    TooManyArtifacts,
};

pub const ArtifactKind = enum {
    lowered_mlir,
    cubin,
    ptx,
    object,
    json,
    diagnostics,

    pub fn extension(self: ArtifactKind) []const u8 {
        return switch (self) {
            .lowered_mlir => ".lowered.mlir",
            .cubin => ".cubin",
            .ptx => ".ptx",
            .object => ".o",
            .json => ".json",
            .diagnostics => ".diag.txt",
        };
    }
};

pub const PipelineKind = enum {
    parse_only,
    canonicalize,
    cute_to_nvvm,
    lir_to_cute_to_nvvm,

    pub fn defaultName(self: PipelineKind) []const u8 {
        return switch (self) {
            .parse_only => "parse-only",
            .canonicalize => "builtin.module(canonicalize)",
            .cute_to_nvvm => "cute-to-nvvm",
            .lir_to_cute_to_nvvm => "lir-to-cute-to-nvvm",
        };
    }
};

pub const CompileRequest = struct {
    input_mlir: []const u8,
    work_dir: []const u8,
    function_name: []const u8,
    arch: []const u8 = "sm_90",
    pipeline_kind: PipelineKind = .cute_to_nvvm,
    flavor: runtime.CompileFlavor = .cutlass_dsl,
    keep_ptx: bool = true,
    keep_cubin: bool = true,
    keep_object: bool = false,
    enable_verifier: bool = true,
    dump_lowered_mlir: bool = true,

    pub fn validate(self: CompileRequest) Error!void {
        if (self.input_mlir.len == 0 or self.work_dir.len == 0 or self.arch.len == 0)
            return Error.InvalidCompileRequest;
        try mlir.validateSymbol(self.function_name);
        if (!self.keep_ptx and !self.keep_cubin and !self.keep_object and !self.dump_lowered_mlir)
            return Error.InvalidCompileRequest;
    }

    pub fn compileOptions(self: CompileRequest) runtime.CompileOptions {
        return .{
            .arch = self.arch,
            .dump_dir = self.work_dir,
            .function_name = self.function_name,
            .keep_ptx = self.keep_ptx,
            .keep_cubin = self.keep_cubin,
            .preserve_line_info = true,
        };
    }

    pub fn writePipeline(self: CompileRequest, out: anytype) Error!void {
        try self.validate();
        switch (self.pipeline_kind) {
            .parse_only => try out.append("parse-only"),
            .canonicalize => try out.append("builtin.module(canonicalize)"),
            .cute_to_nvvm => try self.compileOptions().writePipeline(.cutlass_dsl, out),
            .lir_to_cute_to_nvvm => try self.compileOptions().writePipeline(.cute_experimental_lir, out),
        }
    }

    pub fn artifactPath(
        self: CompileRequest,
        comptime kind: ArtifactKind,
        out: anytype,
    ) Error!void {
        try self.validate();
        try out.append(self.work_dir);
        if (!std.mem.endsWith(u8, self.work_dir, "/")) try out.append("/");
        try out.append(self.function_name);
        switch (kind) {
            // CUTLASS CompileOptions pass dump paths are base paths: the pass
            // writes binary CUBIN directly to dump-cubin-path, e.g.
            // /tmp/out/my_kernel, not necessarily /tmp/out/my_kernel.cubin.
            .cubin => {},
            .ptx => try out.append(kind.extension()),
            else => try out.append(kind.extension()),
        }
    }

    pub fn fullArtifactPath(
        self: CompileRequest,
        comptime kind: ArtifactKind,
        out: anytype,
    ) Error!void {
        try self.validate();
        try out.append(self.work_dir);
        if (!std.mem.endsWith(u8, self.work_dir, "/")) try out.append("/");
        try out.append(self.function_name);
        switch (kind) {
            .cubin, .ptx => {
                try out.append(".");
                try out.append(self.arch);
                try out.append(kind.extension());
            },
            else => try out.append(kind.extension()),
        }
    }

    pub fn expectedArtifacts(self: CompileRequest) Error!ArtifactSet {
        var set: ArtifactSet = .{};
        if (self.dump_lowered_mlir) {
            var p: mlir.TextBuffer(512) = .{};
            try self.artifactPath(.lowered_mlir, &p);
            try set.append(try ArtifactRecord.init(.lowered_mlir, p.slice()));
        }
        if (self.keep_cubin) {
            var p: mlir.TextBuffer(512) = .{};
            try self.artifactPath(.cubin, &p);
            try set.append(try ArtifactRecord.init(.cubin, p.slice()));
        }
        if (self.keep_ptx) {
            var p: mlir.TextBuffer(512) = .{};
            try self.artifactPath(.ptx, &p);
            try set.append(try ArtifactRecord.init(.ptx, p.slice()));
        }
        if (self.keep_object) {
            var p: mlir.TextBuffer(512) = .{};
            try self.artifactPath(.object, &p);
            try set.append(try ArtifactRecord.init(.object, p.slice()));
        }
        return set;
    }
};

pub const ArtifactRecord = struct {
    kind: ArtifactKind,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,
    exists: bool = false,
    size: usize = 0,

    pub fn init(kind: ArtifactKind, path_text: []const u8) Error!ArtifactRecord {
        if (path_text.len == 0 or path_text.len > 512) return Error.MissingArtifactPath;
        var rec: ArtifactRecord = .{ .kind = kind };
        @memcpy(rec.path_buf[0..path_text.len], path_text);
        rec.path_len = path_text.len;
        return rec;
    }

    pub fn path(self: *const ArtifactRecord) []const u8 {
        return self.path_buf[0..self.path_len];
    }

    pub fn validate(self: *const ArtifactRecord) Error!void {
        if (self.path_len == 0) return Error.MissingArtifactPath;
    }

    pub fn writeJson(self: *const ArtifactRecord, out: anytype) Error!void {
        try self.validate();
        try out.append("{\"kind\":");
        try out.appendQuotedString(@tagName(self.kind));
        try out.append(",\"path\":");
        try out.appendQuotedString(self.path());
        try out.append(",\"exists\":");
        try out.append(if (self.exists) "true" else "false");
        try out.append(",\"size\":");
        try out.appendUnsigned(self.size);
        try out.append("}");
    }
};

pub const ArtifactSet = struct {
    records: [16]ArtifactRecord = undefined,
    len: usize = 0,

    pub fn append(self: *ArtifactSet, rec: ArtifactRecord) Error!void {
        try rec.validate();
        if (self.len >= self.records.len) return Error.TooManyArtifacts;
        self.records[self.len] = rec;
        self.len += 1;
    }

    pub fn findIndex(self: *const ArtifactSet, kind: ArtifactKind) ?usize {
        for (self.records[0..self.len], 0..) |r, i| if (r.kind == kind) return i;
        return null;
    }

    pub fn find(self: *const ArtifactSet, kind: ArtifactKind) ?*const ArtifactRecord {
        const idx = self.findIndex(kind) orelse return null;
        return &self.records[idx];
    }

    pub fn hasPathFor(self: *const ArtifactSet, kind: ArtifactKind) bool {
        return self.find(kind) != null;
    }

    pub fn toExecutionArtifacts(
        self: *const ArtifactSet,
        input_mlir: []const u8,
    ) Error!execution.ArtifactSet {
        const cubin = self.find(.cubin) orelse return Error.MissingArtifactPath;
        return .{
            .mlir_path = input_mlir,
            .cubin_path = cubin.path(),
            .ptx_path = if (self.find(.ptx)) |ptx| ptx.path() else null,
            .manifest_path = if (self.find(.json)) |json| json.path() else null,
        };
    }

    pub fn writeJson(self: *const ArtifactSet, out: anytype) Error!void {
        try out.append("[");
        for (self.records[0..self.len], 0..) |*r, i| {
            if (i != 0) try out.append(",");
            try r.writeJson(out);
        }
        try out.append("]");
    }
};

pub const CompileOutcome = struct {
    request: CompileRequest,
    artifacts: ArtifactSet,
    bridge_command: mlir.Invocation,

    pub fn writeJson(self: CompileOutcome, out: anytype) Error!void {
        try out.append("{\"input_mlir\":");
        try out.appendQuotedString(self.request.input_mlir);
        try out.append(",\"work_dir\":");
        try out.appendQuotedString(self.request.work_dir);
        try out.append(",\"function_name\":");
        try out.appendQuotedString(self.request.function_name);
        try out.append(",\"pipeline_kind\":");
        try out.appendQuotedString(@tagName(self.request.pipeline_kind));
        try out.append(",\"artifacts\":");
        try self.artifacts.writeJson(out);
        try out.append(",\"bridge_command\":");
        var cmd: mlir.TextBuffer(4096) = .{};
        try self.bridge_command.writeShell(&cmd);
        try out.appendQuotedString(cmd.slice());
        try out.append("}");
    }
};

pub fn bridgeCompileInvocation(
    config: cutlass.PythonBridgeConfig,
    request: CompileRequest,
) Error!mlir.Invocation {
    try config.validate();
    try request.validate();
    var pipeline: mlir.TextBuffer(4096) = .{};
    try request.writePipeline(&pipeline);
    var inv = mlir.Invocation.init();
    try inv.append(config.python_exe);
    try inv.append(config.bridge_script);
    try inv.append("compile-artifact");
    try inv.append("--module");
    try inv.append(config.package_module);
    try inv.append("--input");
    try inv.append(request.input_mlir);
    try inv.append("--work-dir");
    try inv.append(request.work_dir);
    try inv.append("--function");
    try inv.append(request.function_name);
    try inv.append("--pipeline");
    try inv.append(pipeline.slice());
    if (request.enable_verifier) try inv.append("--enable-verifier");
    if (request.keep_cubin) try inv.append("--expect-cubin");
    if (request.keep_ptx) try inv.append("--expect-ptx");
    if (request.keep_object) try inv.append("--expect-object");
    return inv;
}

pub fn planBridgeCompilation(
    config: cutlass.PythonBridgeConfig,
    request: CompileRequest,
) Error!CompileOutcome {
    return .{
        .request = request,
        .artifacts = try request.expectedArtifacts(),
        .bridge_command = try bridgeCompileInvocation(config, request),
    };
}

pub fn bridgeCompileCommandText(
    config: cutlass.PythonBridgeConfig,
    request: CompileRequest,
    out: anytype,
) Error!void {
    const inv = try bridgeCompileInvocation(config, request);
    try inv.writeShell(out);
}

pub fn writeExpectedArtifactManifest(request: CompileRequest, out: anytype) Error!void {
    const artifacts = try request.expectedArtifacts();
    try artifacts.writeJson(out);
}

pub fn defaultKernelCompileRequest(
    input_mlir: []const u8,
    work_dir: []const u8,
    function_name: []const u8,
    arch: []const u8,
) CompileRequest {
    return .{
        .input_mlir = input_mlir,
        .work_dir = work_dir,
        .function_name = function_name,
        .arch = arch,
        .pipeline_kind = .cute_to_nvvm,
        .keep_cubin = true,
        .keep_ptx = false,
        .keep_object = false,
        .enable_verifier = true,
        .dump_lowered_mlir = true,
    };
}

pub fn writeCompileRunbook(out: anytype, outcome: CompileOutcome) Error!void {
    try out.append("1. Emit CUTLASS-parseable MLIR: ");
    try out.append(outcome.request.input_mlir);
    try out.append("\n2. Run bridge compile command:\n   ");
    var cmd: mlir.TextBuffer(4096) = .{};
    try outcome.bridge_command.writeShell(&cmd);
    try out.append(cmd.slice());
    try out.append("\n3. Expected artifacts: ");
    try outcome.artifacts.writeJson(out);
    try out.append("\n4. Pass cubin path to execution.ExecutableKernel when present.\n");
}

test "compile_pipeline: cute-to-nvvm pipeline command and artifacts are wired" {
    const req: CompileRequest = .{
        .input_mlir = "gemm.mlir",
        .work_dir = "zig-cache/not-cute",
        .function_name = "gemm",
        .keep_ptx = true,
        .keep_cubin = true,
    };
    var pipe: mlir.TextBuffer(4096) = .{};
    try req.writePipeline(&pipe);
    try std.testing.expect(std.mem.indexOf(u8, pipe.slice(), "cute-to-nvvm") != null);
    const outcome = try planBridgeCompilation(.{}, req);
    try std.testing.expect(outcome.artifacts.hasPathFor(.cubin));
    try std.testing.expect(outcome.artifacts.hasPathFor(.ptx));
    var json: mlir.TextBuffer(8192) = .{};
    try outcome.writeJson(&json);
    try std.testing.expect(std.mem.indexOf(u8, json.slice(), "compile-artifact") != null);
    try std.testing.expect(std.mem.indexOf(u8, json.slice(), "zig-cache/not-cute/gemm") != null);
}

test "compile_pipeline: LIR pipeline and execution artifact conversion" {
    const req: CompileRequest = .{
        .input_mlir = "kernel.mlir",
        .work_dir = "out",
        .function_name = "kernel",
        .pipeline_kind = .lir_to_cute_to_nvvm,
        .keep_ptx = false,
    };
    var pipe: mlir.TextBuffer(4096) = .{};
    try req.writePipeline(&pipe);
    try std.testing.expect(std.mem.indexOf(u8, pipe.slice(), "lir-to-cute") != null);
    const artifacts = try req.expectedArtifacts();
    const exec_artifacts = try artifacts.toExecutionArtifacts(req.input_mlir);
    try std.testing.expectEqualStrings("out/kernel", exec_artifacts.cubin_path);
}

pub const CompilationError = error{ CompilationFailed, InvalidCompileOption };
pub const OptLevel = enum { O0, O1, O2, O3 };
pub const GPUArch = enum { sm70, sm75, sm80, sm89, sm90, sm100, sm103, sm120 };
pub const CompileOption = struct { key: []const u8, value: []const u8 = "" };
pub const BooleanCompileOption = struct { key: []const u8, enabled: bool = true };
pub const StringCompileOption = struct { key: []const u8, value: []const u8 };
pub const BooleanBasedFileDumpOption = BooleanCompileOption;
pub const EmptyCompileOption = struct { key: []const u8 };
pub const PtxasOptions = struct { options: []const []const u8 = &.{} };
pub const EnableAssertions = BooleanCompileOption;
pub const GenerateLineInfo = BooleanCompileOption;
pub const KeepCUBIN = BooleanCompileOption;
pub const KeepPTX = BooleanCompileOption;
pub const LinkLibraries = struct { paths: []const []const u8 = &.{} };
pub const EnableTVMFFI = BooleanCompileOption;
pub const DumpDir = StringCompileOption;
pub const CompileCallable = *const fn () void;
pub const PostCompileHookContext = struct { artifact_path: []const u8 = "", cubin_hash: u64 = 0 };
pub const Compiler = struct { opt: OptLevel = .O2, arch: GPUArch = .sm90 };

pub fn makeCompilePlan(
    function_name: []const u8,
    pipeline: []const u8,
) runtime.CompilePlan {
    return .{ .function_name = function_name, .pipeline = pipeline };
}
pub fn option(key: []const u8, value: []const u8) CompileOption {
    return .{ .key = key, .value = value };
}
pub fn boolOption(key: []const u8, enabled: bool) BooleanCompileOption {
    return .{ .key = key, .enabled = enabled };
}
pub fn gpuArchName(arch: GPUArch) []const u8 {
    return switch (arch) {
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

test "compiler_api: source named options and arch spellings" {
    const c: Compiler = .{ .arch = .sm100 };
    try std.testing.expectEqualStrings("sm_100", gpuArchName(c.arch));
    const o = boolOption("keep-ptx", true);
    try std.testing.expect(o.enabled);
}
