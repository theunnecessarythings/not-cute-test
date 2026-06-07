const std = @import("std");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");
const typing = @import("typing.zig");

pub const Error = mlir.Error || runtime.Error || error{ InvalidExportName, TooManyArguments, BufferTooSmall };

pub const max_args = 32;

pub const AotConfig = struct {
    libdir: []const u8 = "lib",
    include_dir: []const u8 = "include",
    libs: []const []const u8 = &.{"cutlass_cute_runtime"},
    rpaths: []const []const u8 = &.{},

    pub fn writeLdFlags(self: AotConfig, out: anytype) Error!void {
        try out.append("-L");
        try out.append(self.libdir);
        for (self.libs) |lib| {
            try out.append(" -l");
            try out.append(lib);
        }
        for (self.rpaths) |rp| {
            try out.append(" -Wl,-rpath,");
            try out.append(rp);
        }
    }
};

pub const ArgumentKind = enum { pointer, tensor, scalar, stream, opaque_ref };

pub const CArgument = struct {
    name: []const u8,
    kind: ArgumentKind,
    c_type: []const u8,
    dynamic: bool = false,

    pub fn init(name: []const u8, kind: ArgumentKind, c_type: []const u8) Error!CArgument {
        try mlir.validateSymbol(name);
        if (c_type.len == 0) return Error.InvalidExportName;
        return .{ .name = name, .kind = kind, .c_type = c_type };
    }
};

pub const CHeaderArguments = struct {
    args: [max_args]CArgument = undefined,
    len: usize = 0,

    pub fn append(self: *CHeaderArguments, arg: CArgument) Error!void {
        if (self.len >= max_args) return Error.TooManyArguments;
        self.args[self.len] = arg;
        self.len += 1;
    }

    pub fn slice(self: *const CHeaderArguments) []const CArgument {
        return self.args[0..self.len];
    }
};

pub const WrapperConfig = struct {
    function_name: []const u8,
    kernel_name: []const u8,
    namespace: ?[]const u8 = null,
    emit_cuda_check: bool = true,
    extern_c: bool = true,

    pub fn init(function_name: []const u8, kernel_name: []const u8) Error!WrapperConfig {
        try mlir.validateSymbol(function_name);
        try mlir.validateSymbol(kernel_name);
        return .{ .function_name = function_name, .kernel_name = kernel_name };
    }
};

pub fn cTypeForNumeric(dtype: typing.Numeric) []const u8 {
    if (dtype.name.len == 0) return "void";
    return switch (dtype.kind) {
        .boolean => "bool",
        .signed_int => switch (dtype.width) {
            4, 8 => "int8_t",
            16 => "int16_t",
            32 => "int32_t",
            64 => "int64_t",
            else => "void*",
        },
        .unsigned_int => switch (dtype.width) {
            8 => "uint8_t",
            16 => "uint16_t",
            32 => "uint32_t",
            64 => "uint64_t",
            else => "void*",
        },
        .float => switch (dtype.width) {
            16 => "cutlass_half_t",
            32 => "float",
            64 => "double",
            else => "void*",
        },
        .bfloat => "cutlass_bfloat16_t",
        .tfloat => "cutlass_tf32_t",
        else => "uint8_t",
    };
}

pub fn writeHeaderPreamble(out: anytype) Error!void {
    try out.append("#pragma once\n");
    try out.append("#include <stdint.h>\n");
    try out.append("#include <stdbool.h>\n");
    try out.append("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
}

pub fn writeHeaderEpilogue(out: anytype) Error!void {
    try out.append("\n#ifdef __cplusplus\n}\n#endif\n");
}

pub fn writeFunctionDeclaration(out: anytype, cfg: WrapperConfig, args: []const CArgument) Error!void {
    if (cfg.extern_c) try out.append("extern ");
    try out.append("int ");
    try out.append(cfg.function_name);
    try out.append("(");
    for (args, 0..) |arg, i| {
        if (i != 0) try out.append(", ");
        try out.append(arg.c_type);
        try out.append(" ");
        try out.append(arg.name);
    }
    try out.append(");\n");
}

pub fn writeCudaCheck(out: anytype) Error!void {
    try out.append("static inline int not_cute_cuda_check(int code) { return code; }\n");
}

pub fn writeWrapper(out: anytype, cfg: WrapperConfig, args: []const CArgument) Error!void {
    try out.append("int ");
    try out.append(cfg.function_name);
    try out.append("(");
    for (args, 0..) |arg, i| {
        if (i != 0) try out.append(", ");
        try out.append(arg.c_type);
        try out.append(" ");
        try out.append(arg.name);
    }
    try out.append(") {\n");
    try out.append("  // zero-dependency generated wrapper stub; actual CUDA launch is linked externally.\n");
    if (cfg.emit_cuda_check) {
        try out.append("  return not_cute_cuda_check(0);\n");
    } else {
        try out.append("  return 0;\n");
    }
    try out.append("}\n");
}

pub fn writeCompleteHeader(out: anytype, cfg: WrapperConfig, args: []const CArgument) Error!void {
    try writeHeaderPreamble(out);
    if (cfg.emit_cuda_check) try writeCudaCheck(out);
    try writeFunctionDeclaration(out, cfg, args);
    try writeHeaderEpilogue(out);
}

pub const ExternalBinaryModule = struct {
    binary: runtime.BinaryModule,
    exported_functions: [16][]const u8 = undefined,
    len: usize = 0,

    pub fn init(binary: runtime.BinaryModule) ExternalBinaryModule {
        return .{ .binary = binary };
    }

    pub fn addFunction(self: *ExternalBinaryModule, name: []const u8) Error!void {
        try mlir.validateSymbol(name);
        if (self.len >= self.exported_functions.len) return Error.TooManyArguments;
        self.exported_functions[self.len] = name;
        self.len += 1;
    }
};

test "export: generates a usable C header surface" {
    var args: CHeaderArguments = .{};
    try args.append(try CArgument.init("A", .pointer, "void*"));
    try args.append(try CArgument.init("stream", .stream, "void*"));
    var out: mlir.TextBuffer(2048) = .{};
    try writeCompleteHeader(&out, try WrapperConfig.init("launch_kernel", "kernel"), args.slice());
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "extern int launch_kernel") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "extern \"C\"") != null);
}

test "export: AOT ldflags are deterministic" {
    var out: mlir.TextBuffer(256) = .{};
    try (AotConfig{}).writeLdFlags(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "-lcutlass_cute_runtime") != null);
}
