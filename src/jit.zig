const std = @import("std");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");

pub const Error = mlir.Error || runtime.Error || error{ TooManySpecializations, InvalidJitArgument, InvalidCompilerState };

pub const JitArgumentKind = enum { constexpr, runtime, tensor, pointer, stream, scalar, dataclass };

pub const JitArgument = struct {
    name: []const u8,
    kind: JitArgumentKind,
    mlir_type: mlir.Type,

    pub fn init(
        name: []const u8,
        kind: JitArgumentKind,
        mlir_type: mlir.Type,
    ) Error!JitArgument {
        try mlir.validateSymbol(name);
        return .{ .name = name, .kind = kind, .mlir_type = mlir_type };
    }
};

pub const JitSignature = struct {
    args: [64]JitArgument = undefined,
    len: usize = 0,

    pub fn append(self: *JitSignature, arg: JitArgument) Error!void {
        if (self.len >= self.args.len) return Error.TooManySpecializations;
        self.args[self.len] = arg;
        self.len += 1;
    }

    pub fn runtimeArgCount(self: *const JitSignature) usize {
        var n: usize = 0;
        for (self.args[0..self.len]) |a| {
            if (a.kind != .constexpr) n += 1;
        }
        return n;
    }

    pub fn writeCacheKey(self: *const JitSignature, out: anytype) Error!void {
        for (self.args[0..self.len]) |a| {
            try out.append(a.name);
            try out.append(":");
            try out.append(@tagName(a.kind));
            try out.append(":");
            try out.append(a.mlir_type.text);
            try out.append(";");
        }
    }
};

pub const CompilerOptions = struct {
    pipeline: mlir.Pipeline = mlir.Pipeline.default(.cute_to_nvvm, .{}),
    verify_each: bool = true,
    keep_intermediates: bool = false,
};

pub const KernelArtifact = struct {
    mlir_path: []const u8,
    cubin_path: []const u8,
    entry: []const u8,
    options: CompilerOptions = .{},

    pub fn writeCompileCommand(self: KernelArtifact, out: anytype) Error!void {
        try self.options.pipeline.writeCommand(out, self.mlir_path, self.cubin_path);
    }
};

pub const JitExecutor = struct {
    signature: JitSignature,
    artifact: KernelArtifact,

    pub fn init(signature: JitSignature, artifact: KernelArtifact) JitExecutor {
        return .{ .signature = signature, .artifact = artifact };
    }

    pub fn prepareLaunch(
        self: JitExecutor,
        module: runtime.BinaryModule,
        config: runtime.LaunchConfig,
    ) Error!runtime.LaunchRecord {
        const fnc = try runtime.KernelFunction.init(module, self.artifact.entry);
        return runtime.recordLaunch(fnc, config, self.signature.runtimeArgCount());
    }
};

pub const JitArgAdapterRegistry = struct {
    entries: [32]Entry = undefined,
    len: usize = 0,

    pub const Entry = struct { type_name: []const u8, adapter_name: []const u8 };

    pub fn register(
        self: *JitArgAdapterRegistry,
        type_name: []const u8,
        adapter_name: []const u8,
    ) Error!void {
        if (self.len >= self.entries.len) return Error.TooManySpecializations;
        self.entries[self.len] = .{
            .type_name = type_name,
            .adapter_name = adapter_name,
        };
        self.len += 1;
    }

    pub fn find(self: *const JitArgAdapterRegistry, type_name: []const u8) ?[]const u8 {
        for (self.entries[0..self.len]) |e| if (std.mem.eql(u8, e.type_name, type_name)) return e.adapter_name;
        return null;
    }
};

pub fn isArgAnnotationConstexpr(kind: JitArgumentKind) bool {
    return kind == .constexpr;
}

pub fn isArgumentConstexpr(arg: JitArgument) bool {
    return arg.kind == .constexpr;
}

test "jit: signature cache key and launch preparation" {
    var sig: JitSignature = .{};
    try sig.append(try JitArgument.init("M", .constexpr, mlir.Type.i(32)));
    try sig.append(try JitArgument.init("A", .pointer, mlir.Type.raw("!cute.ptr")));
    var key: mlir.TextBuffer(256) = .{};
    try sig.writeCacheKey(&key);
    try std.testing.expect(std.mem.indexOf(u8, key.slice(), "M:constexpr") != null);

    const art: KernelArtifact = .{
        .mlir_path = "kernel.mlir",
        .cubin_path = "kernel.cubin",
        .entry = "kernel",
    };
    const exec = JitExecutor.init(sig, art);
    const rec = try exec.prepareLaunch(
        try runtime.BinaryModule.init("kernel.cubin", .cubin),
        try runtime.LaunchConfig.init(
            try runtime.Dim3.init(1, 1, 1),
            try runtime.Dim3.init(128, 1, 1),
            0,
            .{},
        ),
    );
    try std.testing.expectEqual(@as(usize, 1), rec.argument_count);
}
