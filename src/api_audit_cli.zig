const std = @import("std");
const api = @import("api_surface.zig");
const arch_exact = @import("arch_exact.zig");

pub fn main(_: std.process.Init) !void {
    std.debug.print(
        \\
        \\not-cute API/architecture audit
        \\source_public_records={d}
        \\source_classes={d}
        \\source_functions={d}
        \\zig_name_matches={d}
        \\implemented_cute_records={d}
        \\cute_records={d}
        \\arch_nvgpu_records={d}
        \\arch_copy_records={d}
        \\arch_mma_records={d}
        \\arch_records_with_rules={d}
        \\arch_records_with_mlir_factory={d}
        \\
    , .{
        api.source_record_count,
        api.source_class_count,
        api.source_function_count,
        api.countImplementedNameMatches(),
        api.countImplementedModulePrefix("cutlass.cute"),
        api.countModulePrefix("cutlass.cute"),
        arch_exact.source_arch_record_count,
        arch_exact.countKind(.copy),
        arch_exact.countKind(.mma),
        arch_exact.countWithRules(),
        arch_exact.countWithMlirTypeFactory(),
    });
}
