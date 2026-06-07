const std = @import("std");
const mlir = @import("mlir.zig");
const runtime = @import("runtime.zig");
const tensor = @import("tensor.zig");

pub const Error = mlir.Error || runtime.Error || tensor.Error || error{ AssertionFailed, InvalidBenchmarkConfig, TooManyTuningConfigs };

pub const AssertionMode = enum { compile_time, runtime };

pub const Assertion = struct {
    mode: AssertionMode,
    message: []const u8 = "assertion failed",

    pub fn verify(self: Assertion, condition: bool) Error!void {
        _ = self;
        if (!condition) return Error.AssertionFailed;
    }

    pub fn emit(self: Assertion, builder: anytype, condition: mlir.Operand) Error!void {
        try builder.operationNoResult(.{
            .name = switch (self.mode) {
                .compile_time => "cute.testing.compile_time_assert",
                .runtime => "cute.testing.runtime_assert",
            },
            .operands = &.{condition},
            .attrs = &.{.{ .key = "message", .value = self.message }},
            .operand_types = &.{mlir.Type.i(1)},
            .result_types = &.{},
        });
    }
};

pub const RuntimeAssertion = struct {
    assertion: Assertion,
    passed: bool = true,

    pub fn verify(self: RuntimeAssertion) Error!void {
        try self.assertion.verify(self.passed);
    }
};

pub const JitArguments = struct {
    names: [64][]const u8 = undefined,
    len: usize = 0,

    pub fn addToScope(self: *JitArguments, name: []const u8) Error!void {
        if (self.len >= self.names.len) return Error.TooManyTuningConfigs;
        self.names[self.len] = name;
        self.len += 1;
    }
};

pub const BenchmarkConfig = struct {
    warmup_iterations: usize = 5,
    iterations: usize = 100,
    workspace_count: usize = 1,
    streamSync: bool = true,

    pub fn validate(self: BenchmarkConfig) Error!void {
        if (self.iterations == 0 or self.workspace_count == 0)
            return Error.InvalidBenchmarkConfig;
    }
};

pub const BenchmarkResult = struct {
    mean_us: f64,
    min_us: f64,
    max_us: f64,
    iterations: usize,

    pub fn throughputBytesPerSec(self: BenchmarkResult, bytes: usize) f64 {
        if (self.mean_us == 0) return 0;
        return (@as(f64, @floatFromInt(bytes)) * 1_000_000.0) / self.mean_us;
    }
};

pub const AutotuneCandidate = struct {
    name: []const u8,
    score: f64,
};

pub const AutotuneResult = struct {
    best: AutotuneCandidate,
    tried: usize,
};

pub fn chooseBest(candidates: []const AutotuneCandidate) Error!AutotuneResult {
    if (candidates.len == 0) return Error.InvalidBenchmarkConfig;
    var best = candidates[0];
    for (candidates[1..]) |c| {
        if (c.score < best.score) best = c;
    }
    return .{ .best = best, .tried = candidates.len };
}

pub const TensorInitKind = enum { zeros, ones, random_uniform, random_normal, identity, custom };

pub const TensorInitConfig = struct {
    kind: TensorInitKind = .random_uniform,
    seed: u64 = 0,
    low: f64 = -1.0,
    high: f64 = 1.0,

    pub fn validate(self: TensorInitConfig) Error!void {
        if (self.kind == .random_uniform and !(self.low <= self.high))
            return Error.InvalidBenchmarkConfig;
    }
};

pub fn shouldUseNormalInit(cfg: TensorInitConfig) bool {
    return cfg.kind == .random_normal;
}

pub fn convertTensor(
    builder: anytype,
    source: tensor.TensorSsa,
    result_dtype: @TypeOf(source.dtype),
) Error!tensor.TensorSsa {
    var src_ty: mlir.TextBuffer(128) = .{};
    try source.vectorType(&src_ty);
    var dst_ty: mlir.TextBuffer(128) = .{};
    try dst_ty.append("vector<");
    try dst_ty.appendUnsigned(@intCast(try source.shape_value.product()));
    try dst_ty.append("x");
    try dst_ty.append(result_dtype.mlir_type);
    try dst_ty.append(">");
    const v = try builder.genericOp(
        "cute.testing.convert",
        &.{.{ .value = source.value }},
        &.{},
        &.{mlir.Type.raw(src_ty.slice())},
        &.{mlir.Type.raw(dst_ty.slice())},
    );
    return tensor.TensorSsa.init(v, source.shape_value, result_dtype);
}

test "testing: assertion and autotune helpers" {
    try (Assertion{ .mode = .compile_time }).verify(true);
    try std.testing.expectError(
        Error.AssertionFailed,
        (Assertion{ .mode = .runtime }).verify(false),
    );
    const res = try chooseBest(&.{
        .{ .name = "a", .score = 2.0 },
        .{ .name = "b", .score = 1.0 },
    });
    try std.testing.expectEqualStrings("b", res.best.name);
}

test "testing: runtime assertion emits MLIR op" {
    var b: mlir.Builder(512) = .{};
    try (Assertion{ .mode = .runtime, .message = "ok" }).emit(&b, .arg(0));
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "runtime_assert") != null);
}
