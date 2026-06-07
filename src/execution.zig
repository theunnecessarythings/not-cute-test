const std = @import("std");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");
const runtime_plan = @import("runtime_plan.zig");
const cuda = @import("cuda_driver.zig");

fn trace(message: []const u8) void {
    _ = std.os.linux.write(2, message.ptr, message.len);
}
const export_ = @import("export.zig");
const jit = @import("jit.zig");

pub const Error = cuda.Error || runtime_plan.Error || export_.Error || jit.Error || error{
    InvalidExecutionPlan,
    MissingArtifact,
    MissingKernelSymbol,
    InvalidArgumentLayout,
};

pub const ExecutionMode = enum {
    dry_run,
    cuda_driver,
};

pub const ArtifactSet = struct {
    mlir_path: []const u8,
    cubin_path: []const u8,
    ptx_path: ?[]const u8 = null,
    manifest_path: ?[]const u8 = null,

    pub fn validate(self: ArtifactSet) Error!void {
        if (self.mlir_path.len == 0 or self.cubin_path.len == 0) return Error.MissingArtifact;
    }
};

pub const KernelBinding = struct {
    prefix: []const u8,
    public_name: []const u8,
    entry_symbol: []const u8,

    pub fn init(prefix: []const u8, function_name: []const u8) Error!KernelBinding {
        _ = try runtime_plan.RuntimeSymbols.init(prefix, function_name);
        return .{ .prefix = prefix, .public_name = function_name, .entry_symbol = function_name };
    }

    pub fn writeCInterfaceSymbol(self: KernelBinding, out: anytype) Error!void {
        const symbols = try runtime_plan.RuntimeSymbols.init(self.prefix, self.public_name);
        try symbols.writeCInterface(out);
    }
};

pub const ExecutableKernel = struct {
    compile_plan: *const runtime_plan.CompilePlan,
    launch_plan: *const runtime_plan.LaunchPlan,
    artifacts: ArtifactSet,
    binding: KernelBinding,

    pub fn validate(self: ExecutableKernel) Error!void {
        try self.artifacts.validate();
        try self.compile_plan.options.validate();
        if (self.binding.entry_symbol.len == 0) return Error.MissingKernelSymbol;
        _ = try self.launch_plan.prepareRecord();
    }

    pub fn writeManifest(self: ExecutableKernel, out: anytype) Error!void {
        try self.validate();
        try out.append("{\n");
        try out.append("  \"mlir\": ");
        try out.appendQuotedString(self.artifacts.mlir_path);
        try out.append(",\n  \"cubin\": ");
        try out.appendQuotedString(self.artifacts.cubin_path);
        try out.append(",\n  \"entry_symbol\": ");
        try out.appendQuotedString(self.binding.entry_symbol);
        try out.append(",\n  \"c_interface_symbol\": ");
        var iface: mlir.TextBuffer(256) = .{};
        try self.binding.writeCInterfaceSymbol(&iface);
        try out.appendQuotedString(iface.slice());
        try out.append(",\n  \"argument_slots\": ");
        try out.appendUnsigned(self.launch_plan.args.runtimeSlotCount());
        try out.append(",\n  \"compile_command\": ");
        var cmd: mlir.TextBuffer(4096) = .{};
        try self.compile_plan.writeCuteOptCommand(&cmd);
        try out.appendQuotedString(cmd.slice());
        try out.append("\n}\n");
    }

    pub fn dryRun(self: ExecutableKernel) Error!cuda.DryRunReport {
        try self.validate();
        return cuda.dryRun(.{
            .driver_library = self.compile_plan.tools.cuda_driver_library,
            .module_path = self.artifacts.cubin_path,
            .kernel_symbol = self.binding.entry_symbol,
            .launch = self.launch_plan.config,
        }, self.launch_plan.args.runtimeSlotCount());
    }
};

pub const ExecutionResult = struct {
    mode: ExecutionMode,
    launched: bool = false,
    argument_slots: usize = 0,
    message: []const u8 = "",
};

pub fn runDry(executable: ExecutableKernel) Error!ExecutionResult {
    const report = try executable.dryRun();
    return .{ .mode = .dry_run, .launched = false, .argument_slots = report.argument_slots, .message = "validated without launching CUDA" };
}

pub fn launchWithCudaDriver(allocator: std.mem.Allocator, executable: ExecutableKernel, args: *cuda.LaunchArguments) Error!ExecutionResult {
    try executable.validate();
    trace("CUDA: opening driver and creating context...\n");
    var session = try cuda.ManagedSession.open(executable.compile_plan.tools.cuda_driver_library, 0);
    defer session.close();
    trace("CUDA: loading module...\n");
    const module = try session.loadModule(allocator, executable.artifacts.cubin_path);
    defer cuda.unloadModule(session.driver.symbols, module) catch {};
    trace("CUDA: resolving kernel symbol...\n");
    const function = try session.loadFunction(allocator, module, executable.binding.entry_symbol);
    trace("CUDA: launching kernel...\n");
    try session.launch(function, executable.launch_plan.config, args);
    trace("CUDA: synchronizing stream...\n");
    try cuda.synchronizeStream(session.driver.symbols, session.stream);
    trace("CUDA: launch complete.\n");
    return .{ .mode = .cuda_driver, .launched = true, .argument_slots = args.len, .message = "launched via CUDA driver" };
}

pub fn launchCopyWithCudaDriver(allocator: std.mem.Allocator, executable: ExecutableKernel) Error!ExecutionResult {
    try executable.validate();
    trace("CUDA: opening driver and creating context...\n");
    var session = try cuda.ManagedSession.open(executable.compile_plan.tools.cuda_driver_library, 0);
    defer session.close();

    const src = try cuda.allocateDevice(session.driver.symbols, @sizeOf(f32));
    defer cuda.freeDevice(session.driver.symbols, src) catch {};
    const dst = try cuda.allocateDevice(session.driver.symbols, @sizeOf(f32));
    defer cuda.freeDevice(session.driver.symbols, dst) catch {};

    const src_value: f32 = 1.0;
    try cuda.memcpyHtoD(session.driver.symbols, src, std.mem.asBytes(&src_value));

    trace("CUDA: loading module...\n");
    const module = try session.loadModule(allocator, executable.artifacts.cubin_path);
    defer cuda.unloadModule(session.driver.symbols, module) catch {};
    trace("CUDA: resolving kernel symbol...\n");
    const function = try session.loadFunction(allocator, module, executable.binding.entry_symbol);

    var src_ptr = src.ptr;
    var dst_ptr = dst.ptr;
    var coordinate: i32 = 0;
    var args: cuda.LaunchArguments = .{};
    try args.append(try cuda.KernelArgument.init("src", @ptrCast(&src_ptr)));
    try args.append(try cuda.KernelArgument.init("dst", @ptrCast(&dst_ptr)));
    try args.append(try cuda.KernelArgument.init("coordinate", @ptrCast(&coordinate)));

    trace("CUDA: launching kernel...\n");
    try session.launch(function, executable.launch_plan.config, &args);
    trace("CUDA: synchronizing stream...\n");
    try cuda.synchronizeStream(session.driver.symbols, session.stream);
    trace("CUDA: launch complete.\n");
    return .{ .mode = .cuda_driver, .launched = true, .argument_slots = args.len, .message = "launched via CUDA driver" };
}

pub fn writeExecutionCWrapper(out: anytype, executable: ExecutableKernel, args: []const export_.CArgument) Error!void {
    try executable.validate();
    try out.append("// Generated by not-cute execution wiring.\n");
    try cuda.writeCDriverDeclarations(out);
    try out.append("\n");
    try runtime_plan.writeCInterfaceWrapperSource(out, executable.launch_plan.symbols, args);
}

pub fn makeExecutableKernel(plan: *const runtime_plan.CompilePlan, launch: *const runtime_plan.LaunchPlan, mlir_path: []const u8, cubin_path: []const u8) Error!ExecutableKernel {
    const binding = try KernelBinding.init(launch.symbols.prefix, launch.symbols.function_name);
    return .{
        .compile_plan = plan,
        .launch_plan = launch,
        .artifacts = .{ .mlir_path = mlir_path, .cubin_path = cubin_path },
        .binding = binding,
    };
}

pub fn writeBuildRunbook(out: anytype, executable: ExecutableKernel) Error!void {
    try executable.validate();
    try out.append("1. Emit MLIR to ");
    try out.append(executable.artifacts.mlir_path);
    try out.append("\n2. Compile with: ");
    var cmd: mlir.TextBuffer(4096) = .{};
    try executable.compile_plan.writeCuteOptCommand(&cmd);
    try out.append(cmd.slice());
    try out.append("\n3. Load CUDA driver library: ");
    try out.append(executable.compile_plan.tools.cuda_driver_library);
    try out.append("\n4. Load module: ");
    try out.append(executable.artifacts.cubin_path);
    try out.append("\n5. Resolve kernel: ");
    try out.append(executable.binding.entry_symbol);
    try out.append("\n6. Pack ");
    try out.appendUnsigned(executable.launch_plan.args.runtimeSlotCount());
    try out.append(" argument slot(s) and call cuLaunchKernel.\n");
}

test "execution: executable kernel manifest includes compile and launch wiring" {
    var args: runtime_plan.ArgPack = .{};
    try args.append(try runtime_plan.PackedArg.scalarBytes("alpha", .scalar_u64, "12345678"));
    const cfg = try runtime.LaunchConfig.init(try runtime.Dim3.init(1, 1, 1), try runtime.Dim3.init(128, 1, 1), 0, runtime.Stream.default());
    const symbols = try runtime_plan.RuntimeSymbols.init("notcute", "kernel");
    const launch: runtime_plan.LaunchPlan = .{
        .symbols = symbols,
        .module = try runtime.BinaryModule.init("kernel.cubin", .cubin),
        .config = cfg,
        .args = args,
    };
    const compile: runtime_plan.CompilePlan = .{ .options = .{ .function_name = "kernel" }, .input_mlir = "kernel.mlir", .output_cubin = "kernel.cubin" };
    const exe = try makeExecutableKernel(&compile, &launch, "kernel.mlir", "kernel.cubin");
    const res = try runDry(exe);
    try std.testing.expectEqual(@as(usize, 1), res.argument_slots);
    var out: mlir.TextBuffer(4096) = .{};
    try exe.writeManifest(&out);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute-opt --pass-pipeline") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "mlir_ciface_kernel") != null);
}

test "execution: runbook documents CUDA driver launch sequence" {
    const cfg = try runtime.LaunchConfig.init(try runtime.Dim3.init(2, 1, 1), try runtime.Dim3.init(64, 1, 1), 0, runtime.Stream.default());
    const symbols = try runtime_plan.RuntimeSymbols.init("demo", "copy_kernel");
    const launch: runtime_plan.LaunchPlan = .{
        .symbols = symbols,
        .module = try runtime.BinaryModule.init("copy.cubin", .cubin),
        .config = cfg,
        .args = .{},
    };
    const compile: runtime_plan.CompilePlan = .{ .options = .{ .function_name = "copy_kernel" }, .input_mlir = "copy.mlir", .output_cubin = "copy.cubin" };
    const exe = try makeExecutableKernel(&compile, &launch, "copy.mlir", "copy.cubin");
    var out: mlir.TextBuffer(2048) = .{};
    try writeBuildRunbook(&out, exe);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cuLaunchKernel") != null);
}
