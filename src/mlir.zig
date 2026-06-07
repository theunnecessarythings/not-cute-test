const atom = @import("atom.zig");
const copy_mma = @import("copy_mma.zig");
const layout = @import("layout.zig");
const build_options = @import("build_options");
const runtime = @import("runtime.zig");
const std = @import("std");
const tensor = @import("tensor.zig");
const typing = @import("typing.zig");

pub const Error = layout.Error || error{
    EmptyCase,
    GoldenMismatch,
    InvalidMlirAttribute,
    InvalidMlirIdentifier,
    InvalidMlirOperation,
    InvalidMlirString,
    InvalidMlirType,
    InvalidToolConfig,
    MissingExpectedDiagnostic,
    MissingTerminator,
    NegativeTestUnexpectedSuccess,
    RegionUnderflow,
    TooManyArguments,
    TooManyResults,
    ToolFailed,
    ToolNotConfigured,
    UnbalancedRegion,
    UnterminatedString,
};

pub const max_results = 16;

pub const Value = struct {
    id: usize = 0,
    /// Non-null names are emitted literally.  They should include their sigil,
    /// for example "%arg0", "^bb0", or "@callee".
    name: ?[]const u8 = null,
    /// Multi-result MLIR values are printed as `%n#k` at use sites.
    result_index: usize = 0,
    multi_result: bool = false,

    pub fn numbered(id: usize) Value {
        return .{ .id = id };
    }

    pub fn named(name: []const u8) Value {
        return .{ .name = name };
    }

    pub fn arg(comptime index: usize) Value {
        return .{ .name = comptimeArgName(index) };
    }

    pub fn writeTo(self: Value, out: anytype) Error!void {
        if (self.name) |name| {
            try out.append(name);
            return;
        }
        try out.append("%");
        try out.appendUnsigned(self.id);
        if (self.multi_result) {
            try out.append("#");
            try out.appendUnsigned(self.result_index);
        }
    }
};

fn comptimeArgName(comptime index: usize) []const u8 {
    return comptime std.fmt.comptimePrint("%arg{}", .{index});
}

pub const RawValue = struct {
    text: []const u8,
};

/// Reference to an IR value owned outside the current builder.
pub const ExternalValue = RawValue;

pub const Operand = union(enum) {
    value: Value,
    raw: RawValue,

    pub fn arg(comptime index: usize) Operand {
        return .{ .raw = .{ .text = comptimeArgName(index) } };
    }

    pub fn named(text: []const u8) Operand {
        return .{ .raw = .{ .text = text } };
    }

    pub fn external(name: []const u8) Operand {
        return named(name);
    }

    pub fn writeTo(self: Operand, out: anytype) Error!void {
        switch (self) {
            .value => |v| try v.writeTo(out),
            .raw => |r| try out.append(r.text),
        }
    }
};

pub const Type = struct {
    text: []const u8,

    pub fn i(comptime bits: u16) Type {
        return .{ .text = comptime std.fmt.comptimePrint("i{}", .{bits}) };
    }

    pub fn si(comptime bits: u16) Type {
        return .{ .text = comptime std.fmt.comptimePrint("si{}", .{bits}) };
    }

    pub fn ui(comptime bits: u16) Type {
        return .{ .text = comptime std.fmt.comptimePrint("ui{}", .{bits}) };
    }

    pub fn f(comptime bits: u16) Type {
        return .{ .text = comptime std.fmt.comptimePrint("f{}", .{bits}) };
    }

    pub fn bf16() Type {
        return .{ .text = "bf16" };
    }

    pub fn tf32() Type {
        return .{ .text = "tf32" };
    }

    pub fn index() Type {
        return .{ .text = "index" };
    }

    pub fn none() Type {
        return .{ .text = "none" };
    }

    pub fn llvmVoid() Type {
        return .{ .text = "!llvm.void" };
    }

    pub fn raw(text: []const u8) Type {
        return .{ .text = text };
    }

    /// Construct a dialect or application-defined IR type.
    pub fn custom(representation: []const u8) Type {
        return .{ .text = representation };
    }

    pub fn ptr(comptime address_space: ?u32) Type {
        return if (address_space) |space|
            .{ .text = comptimePtrType(space) }
        else
            .{ .text = "!llvm.ptr" };
    }

    pub fn cutePtr(comptime elem: []const u8, comptime space: []const u8) Type {
        return .{
            .text = comptime std.fmt.comptimePrint("!cute.ptr<{s}, {s}>", .{ elem, space }),
        };
    }

    pub fn cuteLayout(comptime shape: []const u8, comptime stride: []const u8) Type {
        return .{
            .text = comptime std.fmt.comptimePrint("!cute.layout<{s}, {s}>", .{ shape, stride }),
        };
    }

    pub fn vector(comptime shape_and_elem: []const u8) Type {
        return .{
            .text = comptime std.fmt.comptimePrint("vector<{s}>", .{shape_and_elem}),
        };
    }

    pub fn memref(comptime shape_and_elem: []const u8) Type {
        return .{
            .text = comptime std.fmt.comptimePrint("memref<{s}>", .{shape_and_elem}),
        };
    }
};

fn comptimePtrType(comptime address_space: u32) []const u8 {
    return comptime std.fmt.comptimePrint("!llvm.ptr<{}>", .{address_space});
}

pub const Attribute = struct {
    key: []const u8,
    value: []const u8,

    pub fn raw(key: []const u8, value: []const u8) Attribute {
        return .{ .key = key, .value = value };
    }

    pub fn bool_(key: []const u8, value: bool) Attribute {
        return .{ .key = key, .value = if (value) "true" else "false" };
    }

    pub fn i32_(comptime key: []const u8, comptime value: i32) Attribute {
        return .{
            .key = key,
            .value = comptime std.fmt.comptimePrint("{} : i32", .{value}),
        };
    }

    pub fn str(comptime key: []const u8, comptime value: []const u8) Attribute {
        return .{ .key = key, .value = "\"" ++ value ++ "\"" };
    }

    pub fn writeTo(self: Attribute, out: anytype) Error!void {
        try validateAttributeKey(self.key);
        try out.append(self.key);
        try out.append(" = ");
        try out.append(self.value);
    }
};

pub const ValueRange = struct {
    values: [max_results]Value = undefined,
    len: usize = 0,

    pub fn first(self: ValueRange) Value {
        std.debug.assert(self.len > 0);
        return self.values[0];
    }

    pub fn at(self: ValueRange, index: usize) Value {
        std.debug.assert(index < self.len);
        return self.values[index];
    }
};

pub fn TextBuffer(comptime capacity: usize) type {
    return struct {
        const Self = @This();
        bytes: [capacity]u8 = undefined,
        len: usize = 0,

        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        pub fn slice(self: *const Self) []const u8 {
            return self.bytes[0..self.len];
        }

        pub fn contents(self: *const Self) []const u8 {
            return self.slice();
        }

        pub fn append(self: *Self, text: []const u8) Error!void {
            if (self.len + text.len > capacity) return Error.OutOfCapacity;
            @memcpy(self.bytes[self.len..][0..text.len], text);
            self.len += text.len;
        }

        pub fn appendByte(self: *Self, byte: u8) Error!void {
            if (self.len >= capacity) return Error.OutOfCapacity;
            self.bytes[self.len] = byte;
            self.len += 1;
        }

        pub fn appendUnsigned(self: *Self, value: usize) Error!void {
            var tmp: [40]u8 = undefined;
            const printed = std.fmt.bufPrint(&tmp, "{}", .{value}) catch unreachable;
            try self.append(printed);
        }

        pub fn appendSigned(self: *Self, value: i128) Error!void {
            var tmp: [64]u8 = undefined;
            const printed = std.fmt.bufPrint(&tmp, "{}", .{value}) catch unreachable;
            try self.append(printed);
        }

        pub fn appendFloat(self: *Self, value: f64) Error!void {
            var tmp: [96]u8 = undefined;
            const printed = std.fmt.bufPrint(&tmp, "{}", .{value}) catch unreachable;
            try self.append(printed);
        }

        pub fn appendQuotedString(self: *Self, text: []const u8) Error!void {
            try self.appendByte('"');
            for (text) |c| switch (c) {
                '"', '\\' => {
                    try self.appendByte('\\');
                    try self.appendByte(c);
                },
                '\n' => try self.append("\\n"),
                '\r' => try self.append("\\r"),
                '\t' => try self.append("\\t"),
                else => try self.appendByte(c),
            };
            try self.appendByte('"');
        }
    };
}

/// Fixed-capacity storage for an IR artifact or encoded IR fragment.
pub fn Storage(comptime capacity: usize) type {
    return TextBuffer(capacity);
}

pub const OperationSpec = struct {
    name: []const u8,
    operands: []const Operand = &.{},
    attrs: []const Attribute = &.{},
    operand_types: []const Type = &.{},
    result_types: []const Type = &.{},
    /// Emit MLIR generic quoted operation form: `"dialect.op"(...) ...`.
    quoted: bool = false,
};

pub const Operation = OperationSpec;

pub fn Builder(comptime capacity: usize) type {
    return struct {
        const Self = @This();
        text: TextBuffer(capacity) = .{},
        next_value: usize = 0,
        indent: usize = 0,
        open_regions: usize = 0,

        pub fn reset(self: *Self) void {
            self.text.clear();
            self.next_value = 0;
            self.indent = 0;
            self.open_regions = 0;
        }

        pub fn slice(self: *const Self) []const u8 {
            return self.text.slice();
        }

        pub fn contents(self: *const Self) []const u8 {
            return self.slice();
        }

        pub fn append(self: *Self, text: []const u8) Error!void {
            try self.text.append(text);
        }

        pub fn appendUnsigned(self: *Self, value: usize) Error!void {
            try self.text.appendUnsigned(value);
        }

        pub fn appendSigned(self: *Self, value: i128) Error!void {
            try self.text.appendSigned(value);
        }

        pub fn newline(self: *Self) Error!void {
            try self.text.appendByte('\n');
        }

        fn writeIndent(self: *Self) Error!void {
            for (0..self.indent) |_| try self.text.append("  ");
        }

        pub fn comment(self: *Self, text: []const u8) Error!void {
            try self.writeIndent();
            try self.append("// ");
            try self.append(text);
            try self.newline();
        }

        pub fn rawLine(self: *Self, text: []const u8) Error!void {
            try self.writeIndent();
            try self.append(text);
            try self.newline();
        }

        /// Insert an operation not yet modeled by the typed builder surface.
        pub fn instruction(self: *Self, representation: []const u8) Error!void {
            try self.rawLine(representation);
        }

        pub fn beginModule(self: *Self) Error!void {
            try self.beginModuleWithAttrs(&.{});
        }

        pub fn beginModuleWithAttrs(self: *Self, attrs: []const Attribute) Error!void {
            try self.writeIndent();
            try self.append("module");
            if (attrs.len != 0) {
                try self.append(" attributes ");
                try self.writeAttrDict(attrs);
            }
            try self.append(" {");
            try self.newline();
            self.indent += 1;
            self.open_regions += 1;
        }

        pub fn endModule(self: *Self) Error!void {
            try self.endRegion();
        }

        pub fn beginGpuModule(
            self: *Self,
            name: []const u8,
            attrs: []const Attribute,
        ) Error!void {
            try validateSymbol(name);
            try self.writeIndent();
            try self.append("gpu.module @");
            try self.append(name);
            if (attrs.len != 0) {
                try self.append(" attributes ");
                try self.writeAttrDict(attrs);
            }
            try self.append(" {");
            try self.newline();
            self.indent += 1;
            self.open_regions += 1;
        }

        pub fn beginRawRegion(self: *Self, header: []const u8) Error!void {
            try self.writeIndent();
            try self.append(header);
            try self.append(" {");
            try self.newline();
            self.indent += 1;
            self.open_regions += 1;
        }

        pub fn endRegion(self: *Self) Error!void {
            if (self.open_regions == 0 or self.indent == 0)
                return Error.RegionUnderflow;
            self.open_regions -= 1;
            self.indent -= 1;
            try self.writeIndent();
            try self.append("}");
            try self.newline();
        }

        pub fn finish(self: *Self) Error![]const u8 {
            if (self.open_regions != 0 or self.indent != 0)
                return Error.UnbalancedRegion;
            try validateBalancedText(self.slice());
            return self.slice();
        }

        pub fn finalize(self: *Self) Error![]const u8 {
            return self.finish();
        }

        pub fn beginFunc(
            self: *Self,
            name: []const u8,
            args: []const Type,
            return_type: ?Type,
        ) Error!void {
            try self.beginFuncWithAttrs(
                name,
                args,
                if (return_type) |r| &.{r} else &.{},
                &.{},
            );
        }

        pub fn beginFuncWithAttrs(
            self: *Self,
            name: []const u8,
            args: []const Type,
            return_types: []const Type,
            attrs: []const Attribute,
        ) Error!void {
            try validateSymbol(name);
            try self.writeIndent();
            try self.append("func.func @");
            try self.append(name);
            try self.append("(");
            for (args, 0..) |arg, i| {
                if (i != 0) try self.append(", ");
                try self.append("%arg");
                try self.appendUnsigned(i);
                try self.append(": ");
                try validateTypeText(arg.text);
                try self.append(arg.text);
            }
            try self.append(")");
            try self.writeReturnSignature(return_types);
            if (attrs.len != 0) {
                try self.append(" attributes ");
                try self.writeAttrDict(attrs);
            }
            try self.append(" {");
            try self.newline();
            self.indent += 1;
            self.open_regions += 1;
        }

        pub fn beginLlvmFunc(
            self: *Self,
            name: []const u8,
            args: []const Type,
            return_types: []const Type,
            attrs: []const Attribute,
        ) Error!void {
            try validateSymbol(name);
            try self.writeIndent();
            try self.append("llvm.func @");
            try self.append(name);
            try self.append("(");
            for (args, 0..) |arg, i| {
                if (i != 0) try self.append(", ");
                try self.append("%arg");
                try self.appendUnsigned(i);
                try self.append(": ");
                try self.append(arg.text);
            }
            try self.append(")");
            try self.writeReturnSignature(return_types);
            if (attrs.len != 0) {
                try self.append(" attributes ");
                try self.writeAttrDict(attrs);
            }
            try self.append(" {");
            try self.newline();
            self.indent += 1;
            self.open_regions += 1;
        }

        pub fn endFunc(self: *Self) Error!void {
            try self.endRegion();
        }

        pub fn ret(
            self: *Self,
            operands: []const Operand,
            types: []const Type,
        ) Error!void {
            if (operands.len != types.len) return Error.RankMismatch;
            try self.writeIndent();
            try self.append("return");
            if (operands.len != 0) {
                try self.append(" ");
                try self.writeOperands(operands);
                try self.append(" : ");
                try self.writeTypeList(types, false);
            }
            try self.newline();
        }

        pub fn funcReturn(
            self: *Self,
            operands: []const Operand,
            types: []const Type,
        ) Error!void {
            try self.ret(operands, types);
        }

        pub fn gpuReturn(self: *Self) Error!void {
            try self.rawLine("gpu.return");
        }

        pub fn llvmReturn(
            self: *Self,
            operands: []const Operand,
            types: []const Type,
        ) Error!void {
            if (operands.len != types.len) return Error.RankMismatch;
            try self.writeIndent();
            try self.append("llvm.return");
            if (operands.len != 0) {
                try self.append(" ");
                try self.writeOperands(operands);
                try self.append(" : ");
                try self.writeTypeList(types, false);
            }
            try self.newline();
        }

        pub fn constantI(self: *Self, value: i128, ty: Type) Error!Value {
            return arith.constantInt(self, value, ty);
        }

        pub fn constantIndex(self: *Self, value: i128) Error!Value {
            return arith.constantInt(self, value, Type.index());
        }

        pub fn constantF(self: *Self, value: f64, ty: Type) Error!Value {
            const result = self.freshValue();
            try self.writeResultPrefixFor(&.{ty}, result.id);
            try self.append("arith.constant ");
            try self.text.appendFloat(value);
            try self.append(" : ");
            try self.append(ty.text);
            try self.newline();
            return result;
        }

        pub fn genericOp(
            self: *Self,
            op_name: []const u8,
            operands: []const Operand,
            attrs: []const Attribute,
            operand_types: []const Type,
            result_types: []const Type,
        ) Error!Value {
            const range = try self.operation(.{
                .name = op_name,
                .operands = operands,
                .attrs = attrs,
                .operand_types = operand_types,
                .result_types = result_types,
            });
            if (range.len != 1) return Error.TooManyResults;
            return range.first();
        }

        pub fn operation(self: *Self, spec: OperationSpec) Error!ValueRange {
            if (spec.result_types.len > max_results) return Error.TooManyResults;
            if (spec.operands.len != spec.operand_types.len) return Error.RankMismatch;
            try validateOperationName(spec.name);
            for (spec.operand_types) |ty| try validateTypeText(ty.text);
            for (spec.result_types) |ty| try validateTypeText(ty.text);

            const range = try self.freshRange(spec.result_types.len);
            try self.writeResultPrefixFor(
                spec.result_types,
                if (spec.result_types.len == 0) 0 else range.values[0].id,
            );
            if (spec.quoted) try self.append("\"");
            try self.append(spec.name);
            if (spec.quoted) try self.append("\"");
            if (spec.operands.len != 0) {
                if (spec.quoted) try self.append("(") else try self.append(" ");
                try self.writeOperands(spec.operands);
                if (spec.quoted) try self.append(")");
            } else if (spec.quoted) {
                try self.append("()");
            }
            if (spec.attrs.len != 0) {
                try self.append(" ");
                try self.writeAttrDict(spec.attrs);
            }
            try self.append(" : ");
            try self.writeFunctionType(spec.operand_types, spec.result_types);
            try self.newline();
            return range;
        }

        pub fn operationNoResult(self: *Self, spec: OperationSpec) Error!void {
            if (spec.result_types.len != 0) return Error.TooManyResults;
            _ = try self.operation(spec);
        }

        pub fn cuteMakeLayout(
            self: *Self,
            shape_type: []const u8,
            stride_type: []const u8,
        ) Error!Value {
            return cute.makeLayout(self, shape_type, stride_type);
        }

        pub fn call(
            self: *Self,
            callee: []const u8,
            operands: []const Operand,
            operand_types: []const Type,
            result_types: []const Type,
        ) Error!ValueRange {
            try validateSymbol(callee);
            if (operands.len != operand_types.len) return Error.RankMismatch;
            for (operand_types) |ty| try validateTypeText(ty.text);
            for (result_types) |ty| try validateTypeText(ty.text);
            const range = try self.freshRange(result_types.len);
            try self.writeResultPrefixFor(
                result_types,
                if (result_types.len == 0) 0 else range.values[0].id,
            );
            try self.append("func.call @");
            try self.append(callee);
            try self.append("(");
            try self.writeOperands(operands);
            try self.append(") : ");
            try self.writeFunctionType(operand_types, result_types);
            try self.newline();
            return range;
        }

        pub fn freshValue(self: *Self) Value {
            const result: Value = .{ .id = self.next_value };
            self.next_value += 1;
            return result;
        }

        pub fn freshRange(self: *Self, len: usize) Error!ValueRange {
            if (len > max_results) return Error.TooManyResults;
            var result: ValueRange = .{};
            result.len = len;
            if (len == 0) return result;
            const base = self.next_value;
            self.next_value += 1;
            for (0..len) |i| {
                result.values[i] = .{
                    .id = base,
                    .result_index = i,
                    .multi_result = len > 1,
                };
            }
            return result;
        }

        pub fn writeResultPrefixFor(
            self: *Self,
            result_types: []const Type,
            result_id: usize,
        ) Error!void {
            try self.writeIndent();
            if (result_types.len == 0) return;
            try self.append("%");
            try self.appendUnsigned(result_id);
            if (result_types.len > 1) {
                try self.append(":");
                try self.appendUnsigned(result_types.len);
            }
            try self.append(" = ");
        }

        pub fn writeOperands(self: *Self, operands: []const Operand) Error!void {
            for (operands, 0..) |op, i| {
                if (i != 0) try self.append(", ");
                try op.writeTo(self);
            }
        }

        fn writeTypeList(
            self: *Self,
            types: []const Type,
            parens_for_multi: bool,
        ) Error!void {
            if (parens_for_multi and types.len != 1) try self.append("(");
            for (types, 0..) |ty, i| {
                if (i != 0) try self.append(", ");
                try self.append(ty.text);
            }
            if (parens_for_multi and types.len != 1) try self.append(")");
        }

        fn writeReturnSignature(self: *Self, return_types: []const Type) Error!void {
            if (return_types.len == 0) return;
            try self.append(" -> ");
            try self.writeTypeList(return_types, true);
        }

        pub fn writeFunctionType(
            self: *Self,
            operand_types: []const Type,
            result_types: []const Type,
        ) Error!void {
            try self.append("(");
            for (operand_types, 0..) |ty, i| {
                if (i != 0) try self.append(", ");
                try self.append(ty.text);
            }
            try self.append(") -> ");
            if (result_types.len == 0) {
                try self.append("()");
            } else if (result_types.len == 1) {
                try self.append(result_types[0].text);
            } else {
                try self.writeTypeList(result_types, true);
            }
        }

        fn writeAttrDict(self: *Self, attrs: []const Attribute) Error!void {
            try self.append("{");
            for (attrs, 0..) |attr, i| {
                if (i != 0) try self.append(", ");
                try attr.writeTo(self);
            }
            try self.append("}");
        }
    };
}

/// Capacity-bounded builder for an MLIR module.
pub fn ModuleBuilder(comptime capacity: usize) type {
    return Builder(capacity);
}

const IrValue = Value;
const IrOperand = Operand;
const IrType = Type;
const IrAttribute = Attribute;
const IrOperation = Operation;
const IrValueRange = ValueRange;

/// Canonical representation-neutral entry point for IR construction.
pub const IR = struct {
    pub const Value = IrValue;
    pub const Operand = IrOperand;
    pub const Type = IrType;
    pub const Attribute = IrAttribute;
    pub const Operation = IrOperation;
    pub const ValueRange = IrValueRange;

    pub fn Storage(comptime capacity: usize) type {
        return TextBuffer(capacity);
    }

    pub fn Module(comptime capacity: usize) type {
        return Builder(capacity);
    }
};

pub const arith = struct {
    pub fn constantInt(builder: anytype, value: i128, ty: Type) Error!Value {
        const result = builder.freshValue();
        try builder.writeResultPrefixFor(&.{ty}, result.id);
        try builder.append("arith.constant ");
        try builder.appendSigned(value);
        try builder.append(" : ");
        try builder.append(ty.text);
        try builder.newline();
        return result;
    }

    pub fn addi(builder: anytype, lhs: Operand, rhs: Operand, ty: Type) Error!Value {
        return binary(builder, "arith.addi", lhs, rhs, ty);
    }

    pub fn subi(builder: anytype, lhs: Operand, rhs: Operand, ty: Type) Error!Value {
        return binary(builder, "arith.subi", lhs, rhs, ty);
    }

    pub fn muli(builder: anytype, lhs: Operand, rhs: Operand, ty: Type) Error!Value {
        return binary(builder, "arith.muli", lhs, rhs, ty);
    }

    pub fn addf(builder: anytype, lhs: Operand, rhs: Operand, ty: Type) Error!Value {
        return binary(builder, "arith.addf", lhs, rhs, ty);
    }

    pub fn cmpi(
        builder: anytype,
        predicate: []const u8,
        lhs: Operand,
        rhs: Operand,
        ty: Type,
    ) Error!Value {
        try validateSymbol(predicate);
        const result = builder.freshValue();
        try builder.writeResultPrefixFor(&.{Type.i(1)}, result.id);
        try builder.append("arith.cmpi ");
        try builder.append(predicate);
        try builder.append(", ");
        try lhs.writeTo(builder);
        try builder.append(", ");
        try rhs.writeTo(builder);
        try builder.append(" : ");
        try builder.append(ty.text);
        try builder.newline();
        return result;
    }

    pub fn select(
        builder: anytype,
        cond: Operand,
        if_value: Operand,
        else_value: Operand,
        ty: Type,
    ) Error!Value {
        const result = builder.freshValue();
        try builder.writeResultPrefixFor(&.{ty}, result.id);
        try builder.append("arith.select ");
        try cond.writeTo(builder);
        try builder.append(", ");
        try if_value.writeTo(builder);
        try builder.append(", ");
        try else_value.writeTo(builder);
        try builder.append(" : ");
        try builder.append(ty.text);
        try builder.newline();
        return result;
    }

    fn binary(
        builder: anytype,
        name: []const u8,
        lhs: Operand,
        rhs: Operand,
        ty: Type,
    ) Error!Value {
        return builder.genericOp(name, &.{ lhs, rhs }, &.{}, &.{ ty, ty }, &.{ty});
    }
};

pub const builtin = struct {
    pub fn unrealizedConversionCast(
        builder: anytype,
        operands: []const Operand,
        operand_types: []const Type,
        result_types: []const Type,
    ) Error!ValueRange {
        return builder.operation(.{
            .name = "builtin.unrealized_conversion_cast",
            .operands = operands,
            .operand_types = operand_types,
            .result_types = result_types,
        });
    }
};

pub const vector = struct {
    pub fn fromElements(
        builder: anytype,
        operands: []const Operand,
        scalar_type: Type,
        result_type: Type,
    ) Error!Value {
        var operand_types: [64]Type = undefined;
        if (operands.len > operand_types.len) return Error.OutOfCapacity;
        for (0..operands.len) |i| operand_types[i] = scalar_type;
        return builder.genericOp(
            "vector.from_elements",
            operands,
            &.{},
            operand_types[0..operands.len],
            &.{result_type},
        );
    }

    pub fn broadcast(
        builder: anytype,
        value: Operand,
        src_type: Type,
        result_type: Type,
    ) Error!Value {
        return builder.genericOp(
            "vector.broadcast",
            &.{value},
            &.{},
            &.{src_type},
            &.{result_type},
        );
    }
};

pub const cute = struct {
    pub fn makeLayout(
        builder: anytype,
        shape_type: []const u8,
        stride_type: []const u8,
    ) Error!Value {
        const result_type = Type.raw("!cute.layout");
        return builder.genericOp(
            "cute.make_layout",
            &.{},
            &.{
                .{ .key = "shape", .value = shape_type },
                .{ .key = "stride", .value = stride_type },
            },
            &.{},
            &.{result_type},
        );
    }

    pub fn makeIntTuple(
        builder: anytype,
        operands: []const Operand,
        operand_types: []const Type,
        result_type: Type,
    ) Error!Value {
        return builder.genericOp(
            "cute.make_int_tuple",
            operands,
            &.{},
            operand_types,
            &.{result_type},
        );
    }

    pub fn tupleAdd(
        builder: anytype,
        lhs: Operand,
        rhs: Operand,
        ty: Type,
    ) Error!Value {
        return builder.genericOp(
            "cute.tuple_add",
            &.{ lhs, rhs },
            &.{},
            &.{ ty, ty },
            &.{ty},
        );
    }

    pub fn slice(
        builder: anytype,
        layout_value: Operand,
        coord_value: Operand,
        layout_type: Type,
        coord_type: Type,
        result_type: Type,
    ) Error!Value {
        return builder.genericOp(
            "cute.slice",
            &.{ layout_value, coord_value },
            &.{},
            &.{ layout_type, coord_type },
            &.{result_type},
        );
    }

    pub fn tileToShape(
        builder: anytype,
        layout_value: Operand,
        shape_value: Operand,
        layout_type: Type,
        shape_type: Type,
        result_type: Type,
    ) Error!Value {
        return builder.genericOp(
            "cute.tile_to_shape",
            &.{ layout_value, shape_value },
            &.{},
            &.{ layout_type, shape_type },
            &.{result_type},
        );
    }
};

pub const gpu = struct {
    pub fn threadId(builder: anytype, dimension: []const u8) Error!Value {
        const result = builder.freshValue();
        try builder.writeResultPrefixFor(&.{Type.index()}, result.id);
        try builder.append("gpu.thread_id ");
        try builder.append(dimension);
        try builder.append(" : index");
        try builder.newline();
        return result;
    }
};

pub const llvm = struct {
    pub fn inlineAsm(
        builder: anytype,
        asm_string: []const u8,
        constraints: []const u8,
        operands: []const Operand,
        operand_types: []const Type,
        result_types: []const Type,
        side_effects: bool,
    ) Error!ValueRange {
        if (operands.len != operand_types.len) return Error.RankMismatch;
        const range = try builder.freshRange(result_types.len);
        try builder.writeResultPrefixFor(
            result_types,
            if (result_types.len == 0) 0 else range.values[0].id,
        );
        try builder.append("llvm.inline_asm ");
        try builder.text.appendQuotedString(asm_string);
        try builder.append(", ");
        try builder.text.appendQuotedString(constraints);
        if (side_effects) try builder.append(" side_effects");
        if (operands.len != 0) {
            try builder.append(" ");
            try builder.writeOperands(operands);
        }
        try builder.append(" : ");
        try builder.writeFunctionType(operand_types, result_types);
        try builder.newline();
        return range;
    }
};

pub const CompileOptions = struct {
    arch: []const u8 = "sm_90",
    opt_level: u8 = 3,
    cubin_format: []const u8 = "bin",
    enable_cuda_dialect: bool = false,
    cuda_dialect_external_module: bool = false,

    pub fn writeCuteToNvvmOptions(self: CompileOptions, out: anytype) Error!void {
        try out.append("cubin-format=");
        try out.append(self.cubin_format);
        try out.append(" arch=");
        try out.append(self.arch);
        try out.append(" opt-level=");
        try out.appendUnsigned(self.opt_level);
        if (self.enable_cuda_dialect) try out.append(" enable-cuda-dialect=true");
        if (self.cuda_dialect_external_module) try out.append(" cuda-dialect-external-module=true");
    }
};

pub const PipelineKind = enum {
    cute_to_nvvm,
    lir_to_cute_to_nvvm,
    canonicalize_only,
};

pub const Pipeline = struct {
    /// opt executable, e.g. "cute-opt" or "mlir-opt".
    opt: []const u8 = "cute-opt",
    passes: []const []const u8 = &.{},
    raw_pipeline: ?[]const u8 = null,

    pub fn default(kind: PipelineKind, options: CompileOptions) Pipeline {
        return .{
            .opt = "cute-opt",
            .passes = &.{},
            .raw_pipeline = switch (kind) {
                .cute_to_nvvm => defaultCuteToNvvmPipeline(options),
                .lir_to_cute_to_nvvm => defaultLirToCuteToNvvmPipeline(options),
                .canonicalize_only => "builtin.module(canonicalize,cse)",
            },
        };
    }

    pub fn writePipeline(self: Pipeline, out: anytype) Error!void {
        if (self.raw_pipeline) |p| {
            try out.append(p);
            return;
        }
        if (self.passes.len == 0) {
            try out.append("builtin.module()");
            return;
        }
        try out.append("builtin.module(");
        for (self.passes, 0..) |pass, i| {
            if (i != 0) try out.append(",");
            try out.append(pass);
        }
        try out.append(")");
    }

    pub fn writeCommand(
        self: Pipeline,
        out: anytype,
        input_path: []const u8,
        output_path: ?[]const u8,
    ) Error!void {
        try out.append(self.opt);
        if (self.raw_pipeline) |pipeline_text| {
            try out.append(" --pass-pipeline=");
            try out.appendQuotedString(pipeline_text);
        } else {
            for (self.passes) |pass| {
                try out.append(" --");
                try out.append(pass);
            }
        }
        try out.append(" ");
        try out.append(input_path);
        if (output_path) |path| {
            try out.append(" -o ");
            try out.append(path);
        }
    }
};

fn defaultCuteToNvvmPipeline(options: CompileOptions) []const u8 {
    if (options.enable_cuda_dialect or options.cuda_dialect_external_module) {
        return "builtin.module(cute-to-nvvm{cubin-format=bin enable-cuda-dialect=true cuda-dialect-external-module=true})";
    }
    return "builtin.module(cute-to-nvvm{cubin-format=bin})";
}

fn defaultLirToCuteToNvvmPipeline(options: CompileOptions) []const u8 {
    if (options.enable_cuda_dialect or options.cuda_dialect_external_module) {
        return "builtin.module(gpu.module(lir-to-cute{enable-cuda-dialect enable-lir-func-finalization=false}),lir-func-finalization{enable-cuda-dialect=true},cute-to-nvvm{cubin-format=bin enable-cuda-dialect=true})";
    }
    return "builtin.module(gpu.module(lir-to-cute{enable-lir-func-finalization=false}),lir-func-finalization,cute-to-nvvm{cubin-format=bin})";
}

pub fn validateSymbol(name: []const u8) Error!void {
    if (name.len == 0) return Error.InvalidMlirIdentifier;
    const first = name[0];
    if (!isIdentHead(first) and first != '$' and first != '.')
        return Error.InvalidMlirIdentifier;
    for (name[1..]) |c| {
        const ok = isIdentTail(c) or c == '$' or c == '.';
        if (!ok) return Error.InvalidMlirIdentifier;
    }
}

pub fn validateOperationName(name: []const u8) Error!void {
    if (name.len == 0) return Error.InvalidMlirOperation;
    var saw_dot = false;
    for (name) |c| {
        const ok = isIdentTail(c) or c == '.' or c == '_' or c == '-';
        if (!ok) return Error.InvalidMlirOperation;
        if (c == '.') saw_dot = true;
    }
    if (!saw_dot) return Error.InvalidMlirOperation;
}

fn validateAttributeKey(key: []const u8) Error!void {
    if (key.len == 0) return Error.InvalidMlirAttribute;
    for (key) |c| {
        const ok = isIdentTail(c) or c == '.' or c == '_' or c == '-';
        if (!ok) return Error.InvalidMlirAttribute;
    }
}

pub fn validateTypeText(text: []const u8) Error!void {
    if (text.len == 0) return Error.InvalidMlirType;
    var angle_depth: isize = 0;
    var in_string = false;
    var escape = false;
    for (text) |c| {
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '<' => angle_depth += 1,
            '>' => {
                angle_depth -= 1;
                if (angle_depth < 0) return Error.InvalidMlirType;
            },
            '\n', '\r' => return Error.InvalidMlirType,
            else => {},
        }
    }
    if (in_string or angle_depth != 0) return Error.InvalidMlirType;
}

pub fn validateBalancedText(text: []const u8) Error!void {
    var braces: isize = 0;
    var parens: isize = 0;
    var brackets: isize = 0;
    var angles: isize = 0;
    var in_string = false;
    var escape = false;
    for (text) |c| {
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '{' => braces += 1,
            '}' => {
                braces -= 1;
                if (braces < 0) return Error.UnbalancedRegion;
            },
            '(' => parens += 1,
            ')' => {
                parens -= 1;
                if (parens < 0) return Error.UnbalancedRegion;
            },
            '[' => brackets += 1,
            ']' => {
                brackets -= 1;
                if (brackets < 0) return Error.UnbalancedRegion;
            },
            '<' => angles += 1,
            '>' => {
                if (angles > 0) angles -= 1;
            },
            else => {},
        }
    }
    if (in_string) return Error.UnterminatedString;
    if (braces != 0 or parens != 0 or brackets != 0 or angles != 0)
        return Error.UnbalancedRegion;
}

fn isIdentHead(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}

fn isIdentTail(c: u8) bool {
    return isIdentHead(c) or (c >= '0' and c <= '9');
}

test "mlir_text: emits a module with arithmetic" {
    var b: Builder(4096) = .{};
    try b.beginModule();
    try b.beginFunc("add_one", &.{Type.i(32)}, Type.i(32));
    const one = try b.constantI(1, Type.i(32));
    const sum = try b.genericOp(
        "arith.addi",
        &.{ .{ .raw = .{ .text = "%arg0" } }, .{ .value = one } },
        &.{},
        &.{ Type.i(32), Type.i(32) },
        &.{Type.i(32)},
    );
    try b.ret(&.{.{ .value = sum }}, &.{Type.i(32)});
    try b.endFunc();
    try b.endModule();
    _ = try b.finish();

    const expected =
        \\module {
        \\  func.func @add_one(%arg0: i32) -> i32 {
        \\    %0 = arith.constant 1 : i32
        \\    %1 = arith.addi %arg0, %0 : (i32, i32) -> i32
        \\    return %1 : i32
        \\  }
        \\}
        \\
    ;
    try std.testing.expectEqualStrings(expected, b.slice());
}

test "mlir_text: multi-result generic op uses MLIR result group syntax" {
    var b: Builder(2048) = .{};
    try b.beginModule();
    try b.beginFunc("split", &.{Type.i(32)}, null);
    const values = try b.operation(.{
        .name = "test.split",
        .operands = &.{Operand.arg(0)},
        .operand_types = &.{Type.i(32)},
        .result_types = &.{ Type.i(16), Type.i(16) },
        .quoted = true,
    });
    try b.ret(
        &.{ .{ .value = values.at(0) }, .{ .value = values.at(1) } },
        &.{ Type.i(16), Type.i(16) },
    );
    try b.endFunc();
    try b.endModule();
    _ = try b.finish();

    const expected =
        \\module {
        \\  func.func @split(%arg0: i32) {
        \\    %0:2 = "test.split"(%arg0) : (i32) -> (i16, i16)
        \\    return %0#0, %0#1 : i16, i16
        \\  }
        \\}
        \\
    ;
    try std.testing.expectEqualStrings(expected, b.slice());
}

test "mlir_text: every result-producing path enforces max_results" {
    var b: Builder(2048) = .{};
    const too_many = [_]Type{Type.i(1)} ** (max_results + 1);
    try std.testing.expectError(Error.TooManyResults, b.freshRange(too_many.len));
    try std.testing.expectError(
        Error.TooManyResults,
        b.call("callee", &.{}, &.{}, &too_many),
    );
    try std.testing.expectEqual(@as(usize, 0), b.next_value);
}

test "mlir_text: GPU module, cute op, and cuda pipeline" {
    var b: Builder(4096) = .{};
    try b.beginModuleWithAttrs(&.{Attribute.str("cute.source", "zig-port")});
    try b.beginGpuModule(
        "kernels",
        &.{Attribute.raw("cc_attr", "#core.compute_capability<arch = sm_90>")},
    );
    try b.beginFuncWithAttrs(
        "kernel",
        &.{ Type.ptr(3), Type.i(32) },
        &.{},
        &.{Attribute.raw("gpu.kernel", "unit")},
    );
    const tid = try gpu.threadId(&b, "x");
    const one = try arith.constantInt(&b, 1, Type.index());
    _ = try arith.addi(&b, .{ .value = tid }, .{ .value = one }, Type.index());
    try b.gpuReturn();
    try b.endFunc();
    try b.endRegion();
    try b.endModule();
    _ = try b.finish();

    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "gpu.module @kernels") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "gpu.thread_id x : index") != null);

    var out: TextBuffer(512) = .{};
    const p = Pipeline.default(
        .cute_to_nvvm,
        .{ .enable_cuda_dialect = true, .cuda_dialect_external_module = true },
    );
    try p.writeCommand(&out, "in.mlir", "out.mlir");
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "--pass-pipeline=") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute-to-nvvm") != null);
}

test "mlir_text: pass pipeline command preserves legacy pass-list behavior" {
    var out: TextBuffer(512) = .{};
    const p: Pipeline = .{
        .opt = "cute-opt",
        .passes = &.{ "canonicalize", "cse", "convert-vector-to-llvm" },
    };
    try p.writeCommand(&out, "in.mlir", "out.mlir");
    try std.testing.expectEqualStrings(
        "cute-opt --canonicalize --cse --convert-vector-to-llvm in.mlir -o out.mlir",
        out.slice(),
    );
}

test "mlir_text: validation catches malformed names and unbalanced text" {
    try std.testing.expectError(
        Error.InvalidMlirOperation,
        validateOperationName("arith addi"),
    );
    try std.testing.expectError(Error.InvalidMlirIdentifier, validateSymbol("9bad"));
    try std.testing.expectError(
        Error.InvalidMlirType,
        validateTypeText("vector<4xi32"),
    );
    try std.testing.expectError(
        Error.UnbalancedRegion,
        validateBalancedText("module {"),
    );
}

/// Source-grounded MLIR dialect operation inventory.
///
/// This is intentionally a registry of textual operation spelling, not generated
/// bindings.  The uploaded CuteDSL tree imports generated Python MLIR bindings
/// from `cutlass._mlir`; those bindings are not part of the source archive and
/// would violate the requested zero-dependency direction.  The Zig port therefore
/// records the dialect/op names used by the Python code and emits them as text.
pub const Dialect = enum {
    arith,
    builtin,
    cf,
    cuda,
    cute,
    cute_nvgpu,
    func,
    gpu,
    lir,
    llvm,
    math,
    nvgpu,
    nvvm,
    scf,
    vector,
};

pub const KnownOp = struct {
    dialect: Dialect,
    name: []const u8,

    pub fn fullName(self: KnownOp, out: anytype) !void {
        try out.append(@tagName(self.dialect));
        try out.append(".");
        try out.append(self.name);
    }
};

pub const arith_ops = [_][]const u8{
    "constant", "addi",       "addf",      "subi",   "subf",   "muli",     "mulf",       "divui",        "divsi",
    "divf",     "floordivsi", "ceildivsi", "remui",  "remsi",  "remf",     "andi",       "ori",          "xori",
    "shli",     "shrui",      "shrsi",     "cmpi",   "cmpf",   "select",   "index_cast", "index_castui", "extui",
    "extsi",    "trunci",     "extf",      "truncf", "sitofp", "uitofp",   "fptosi",     "fptoui",       "bitcast",
    "negf",     "minsi",      "minui",     "maxsi",  "maxui",  "minimumf", "maximumf",
};

pub const math_ops = [_][]const u8{
    "absf", "absi",  "acos", "asin",  "atan",  "atan2", "ceil",  "copysign", "cos",  "ctpop",
    "erf",  "exp",   "exp2", "floor", "fpowi", "gcd",   "ipowi", "log",      "log2", "log10",
    "powf", "rsqrt", "sin",  "sqrt",  "tan",   "tanh",
};

pub const builtin_ops = [_][]const u8{
    "module", "unrealized_conversion_cast",
};

pub const func_ops = [_][]const u8{
    "func", "return", "call",
};

pub const gpu_ops = [_][]const u8{
    "module", "container_module", "binary", "global",    "launch",   "launch_func", "printf",
    "return", "sync",             "wait",   "thread_id", "block_id", "grid_dim",    "block_dim",
};

pub const vector_ops = [_][]const u8{
    "broadcast",     "bitcast",        "constant_mask", "extract",       "extractelement",       "extract_strided_slice",
    "from_elements", "gather",         "insert",        "insertelement", "insert_strided_slice", "multi_reduction",
    "reduction",     "scatter",        "shape_cast",    "shuffle",       "splat",                "to_elements",
    "transfer_read", "transfer_write",
};

pub const llvm_ops = [_][]const u8{
    "addrspacecast", "alloca",  "and",  "bitcast",       "br",       "call",         "cond_br",    "extractelement",
    "extractvalue",  "fptrunc", "func", "getelementptr", "global",   "icmp",         "inline_asm", "insertvalue",
    "inttoptr",      "load",    "mul",  "or",            "ptrtoint", "return",       "sitofp",     "store",
    "trunc",         "urem",    "xor",  "addressof",     "constant", "global_dtors", "undef",      "zero",
    "poison",
};

pub const scf_ops = [_][]const u8{
    "for", "if", "while", "condition", "execute_region", "yield",
};

pub const cf_ops = [_][]const u8{
    "assert", "br", "cond_br",
};

pub const nvvm_ops = [_][]const u8{
    "barrier",                  "barrier0",               "barrier_arrive",        "bar_warp_sync",         "cp_async_bulk_commit_group",
    "cp_async_bulk_wait_group", "cp_async_commit_group",  "cp_async_wait_group",   "cluster_arrive",        "cluster_arrive_relaxed",
    "cluster_wait",             "elect_sync",             "fence_acq_rel_cta",     "fence_acq_rel_cluster", "fence_acq_rel_gpu",
    "fence_acq_rel_sys",        "fence_proxy",            "fma_packed_f32x2",      "match_sync",            "mapa",
    "mbarrier_init_shared",     "mbarrier_txn",           "prefetch",              "read.ptx.sreg.clock",   "read.ptx.sreg.clock64",
    "read.ptx.sreg.ctaid.x",    "read.ptx.sreg.ctaid.y",  "read.ptx.sreg.ctaid.z", "read.ptx.sreg.laneid",  "read.ptx.sreg.nctaid.x",
    "read.ptx.sreg.nctaid.y",   "read.ptx.sreg.nctaid.z", "read.ptx.sreg.ntid.x",  "read.ptx.sreg.ntid.y",  "read.ptx.sreg.ntid.z",
    "read.ptx.sreg.smid",       "read.ptx.sreg.tid.x",    "read.ptx.sreg.tid.y",   "read.ptx.sreg.tid.z",   "redux_sync",
    "setmaxregister",           "shfl.sync",              "store",                 "load",                  "tcgen05_commit",
    "tcgen05_wait",             "vote_ballot_sync",       "vote_sync",
};

pub const cuda_ops = [_][]const u8{
    "cast",                   "kernel",                           "launch_cfg_create",      "launch_cfg_programmatic_stream_serialization_allowed",
    "launch_cfg_cluster_dim", "launch_cfg_preferred_cluster_dim", "launch_cfg_cooperative", "launch_ex",
    "return",                 "return_if_error",
};

pub const cute_ops = [_][]const u8{
    "assume",           "blocked_product",       "complement",          "copy",               "cosize",             "deref_arith_tuple_iter",
    "elem_less",        "equal",                 "filter",              "filter_zeros",       "flat_product",       "gemm",
    "get_iter",         "get_layout",            "get_leaves",          "get_shape",          "inttoptr",           "is_static",
    "logical_product",  "make_arith_tuple_iter", "make_atom",           "make_coord",         "make_fragment_like", "make_identity_tensor",
    "make_int_tuple",   "make_layout",           "make_layout_like",    "make_shape",         "make_stride",        "make_tensor",
    "make_tile",        "make_view",             "memref_alloca",       "memref_load",        "memref_load_vec",    "memref_store",
    "memref_store_vec", "mma_make_fragment",     "pack_coord",          "pack_int_tuple",     "pack_shape",         "pack_stride",
    "pack_tile",        "prefetch",              "prepend_to_rank",     "print_view",         "raked_product",      "slice",
    "static",           "tile_to_shape",         "tiled_mma_partition", "tiled_product",      "tuple_add",          "tuple_div",
    "tuple_mod",        "tuple_mul",             "tuple_product",       "tuple_product_each", "tuple_sub",          "zipped_product",
};

pub const cute_nvgpu_ops = [_][]const u8{
    "arch_alloc_smem",                     "arch_get_dyn_smem",                 "arch_get_dyn_smem_size",                  "arch_make_warp_uniform",
    "arch_sm100_alloc_tmem",               "arch_sm100_dealloc_tmem",           "arch_sm100_relinquish_tmem_alloc_permit", "arch_sm100_retrieve_tmem_ptr",
    "atom_get_copy_s2t_smem_desc_view",    "atom_get_value",                    "atom_make_exec_tma",                      "atom_make_non_exec_im2col_tma_load",
    "atom_make_non_exec_im2col_tma_store", "atom_make_non_exec_tiled_tma_load", "atom_make_non_exec_tiled_tma_reduce",     "atom_make_non_exec_tiled_tma_store",
    "atom_make_s2t_copy",                  "atom_make_tmem_copy",               "atom_set_value",                          "atom_tma_partition",
    "copy_tma_desc",                       "get_default_tma_format",            "make_tmem_layout_sfa",                    "make_tmem_layout_sfb",
    "make_umma_smem_desc",                 "prefetch_tma_desc",                 "tile_to_mma_shape",                       "update_tma_desc",
};

pub const lir_ops = [_][]const u8{
    "allocate_buffer",            "copy",                  "create_circular_buffer_pipeline", "create_circular_buffer_pipeline_state",
    "create_pipeline",            "create_pipeline_state", "create_pipeline_with_mask",       "dot",
    "dot_block_scaled",           "func",                  "get_mbarrier",                    "get_pipeline_consume_stage",
    "get_pipeline_produce_stage", "mbarrier_expect_tx",    "partition",                       "pipeline_advance_iterator",
    "producer_acquire",           "producer_commit",       "producer_try_acquire",            "consumer_release",
    "consumer_tail",              "consumer_try_wait",     "consumer_wait",                   "return",
    "simt_auto_vec_copy",         "tma_load",              "tma_load_multicast",              "tma_store",
};

fn table(dialect: Dialect) []const []const u8 {
    return switch (dialect) {
        .arith => &arith_ops,
        .builtin => &builtin_ops,
        .cf => &cf_ops,
        .cuda => &cuda_ops,
        .cute => &cute_ops,
        .cute_nvgpu => &cute_nvgpu_ops,
        .func => &func_ops,
        .gpu => &gpu_ops,
        .lir => &lir_ops,
        .llvm => &llvm_ops,
        .math => &math_ops,
        .nvgpu => &[_][]const u8{},
        .nvvm => &nvvm_ops,
        .scf => &scf_ops,
        .vector => &vector_ops,
    };
}

pub fn isKnown(dialect: Dialect, name: []const u8) bool {
    for (table(dialect)) |candidate| {
        if (std.mem.eql(u8, candidate, name)) return true;
    }
    return false;
}

pub fn isKnownFullName(full_name: []const u8) bool {
    const dot = std.mem.indexOfScalar(u8, full_name, '.') orelse return false;
    const prefix = full_name[0..dot];
    const suffix = full_name[dot + 1 ..];
    inline for (@typeInfo(Dialect).@"enum".fields) |field| {
        if (std.mem.eql(u8, prefix, field.name)) {
            return isKnown(@enumFromInt(field.value), suffix);
        }
    }
    return false;
}

test "mlir_ops: source-grounded registry recognizes major CuteDSL ops" {
    try std.testing.expect(isKnown(.cute, "make_layout"));
    try std.testing.expect(isKnown(.cute, "tile_to_shape"));
    try std.testing.expect(isKnown(.cute_nvgpu, "atom_make_exec_tma"));
    try std.testing.expect(isKnown(.arith, "cmpi"));
    try std.testing.expect(isKnown(.nvvm, "shfl.sync"));
    try std.testing.expect(isKnown(.lir, "tma_load"));
    try std.testing.expect(!isKnown(.cute, "not_a_real_cute_op"));
}

pub const layout_case_mlir =
    \\module {
    \\  func.func @layout_case() {
    \\    %0 = cute.make_shape() : () -> !cute.shape<"(2,3)">
    \\    %1 = cute.make_stride() : () -> !cute.stride<"(3,1)">
    \\    %2 = cute.make_layout(%0, %1) : !cute.layout<"(2,3):(3,1)">
    \\    return
    \\  }
    \\}
    \\
;

pub const tensor_case_mlir =
    \\module {
    \\  func.func @tensor_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: vector<4xf32>) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">) -> vector<4xf32>
    \\    %1 = arith.addf %0, %arg1 : vector<4xf32>
    \\    cute.memref.store_vec(%1, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, align<16>, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const copy_case_mlir =
    \\module {
    \\  func.func @copy_case(%arg0: !cute.memref<f32, gmem, align<16>, "(1):(1)">, %arg1: !cute.memref<f32, gmem, align<16>, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%atom, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, align<16>, "(1):(1)">, !cute.memref<f32, gmem, align<16>, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const mma_case_mlir =
    \\module {
    \\  func.func @mma_case(%arg0: !cute.memref<f32, generic, "(1):(1)">, %arg1: !cute.memref<f32, generic, "(1):(1)">, %arg2: !cute.memref<f32, generic, "(1):(1)">, %arg3: !cute.memref<f32, generic, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%atom, %arg3, %arg0, %arg1, %arg2) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const ToolKind = enum {
    cute_opt,
    mlir_opt,
    filecheck,
    custom,
};

pub const MlirCaseKind = enum {
    layout,
    tensor,
    copy,
    mma,
    negative,
};

pub const ToolConfig = struct {
    cute_opt: []const u8 = build_options.cute_opt_path,
    mlir_opt: []const u8 = build_options.mlir_opt_path,
    filecheck: []const u8 = build_options.filecheck_path,
    enable_external_tools: bool = build_options.enable_mlir_tools,
    assume_tools_present: bool = build_options.assume_mlir_tools_present,
    max_output_bytes: usize = 1 << 20,

    pub fn pathFor(
        self: ToolConfig,
        kind: ToolKind,
        custom_path: ?[]const u8,
    ) Error![]const u8 {
        return switch (kind) {
            .cute_opt => self.cute_opt,
            .mlir_opt => self.mlir_opt,
            .filecheck => self.filecheck,
            .custom => custom_path orelse Error.InvalidToolConfig,
        };
    }

    pub fn shouldRunExternal(self: ToolConfig) bool {
        return self.enable_external_tools or self.assume_tools_present;
    }
};

pub const Invocation = struct {
    argv: [32][]const u8 = undefined,
    argc: usize = 0,

    pub fn init() Invocation {
        return .{};
    }

    pub fn append(self: *Invocation, arg: []const u8) Error!void {
        if (self.argc >= self.argv.len) return Error.TooManyArguments;
        self.argv[self.argc] = arg;
        self.argc += 1;
    }

    pub fn args(self: *const Invocation) []const []const u8 {
        return self.argv[0..self.argc];
    }

    pub fn writeShell(self: *const Invocation, out: anytype) !void {
        for (self.args(), 0..) |arg, i| {
            if (i != 0) try out.append(" ");
            try appendShellQuoted(out, arg);
        }
    }
};

pub const GoldenCase = struct {
    name: []const u8,
    kind: MlirCaseKind,
    mlir_text: []const u8,
    expect_failure: bool = false,
    expected_diagnostic: ?[]const u8 = null,
};

pub fn expectGolden(actual: []const u8, expected: []const u8) Error!void {
    if (!std.mem.eql(u8, actual, expected)) return Error.GoldenMismatch;
}

pub fn expectContains(haystack: []const u8, needle: []const u8) Error!void {
    if (std.mem.indexOf(u8, haystack, needle) == null)
        return Error.MissingExpectedDiagnostic;
}

pub fn validateGeneratedMlir(text: []const u8) Error!void {
    if (text.len == 0) return Error.EmptyCase;
    try validateBalancedText(text);
    try expectContains(text, "module");
}

pub fn cuteOptVerifyInvocation(
    config: ToolConfig,
    input_path: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.cute_opt, null));
    try inv.append("--verify-diagnostics");
    try inv.append(input_path);
    return inv;
}

pub fn cuteOptPipelineInvocation(
    config: ToolConfig,
    input_path: []const u8,
    output_path: []const u8,
    pipeline: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.cute_opt, null));
    try inv.append(pipeline);
    try inv.append(input_path);
    try inv.append("-o");
    try inv.append(output_path);
    return inv;
}

pub fn mlirOptPipelineInvocation(
    config: ToolConfig,
    input_path: []const u8,
    output_path: []const u8,
    pass_pipeline: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.mlir_opt, null));
    try inv.append(pass_pipeline);
    try inv.append(input_path);
    try inv.append("-o");
    try inv.append(output_path);
    return inv;
}

pub fn fileCheckInvocation(
    config: ToolConfig,
    input_path: []const u8,
    check_file: []const u8,
) Error!Invocation {
    var inv = Invocation.init();
    try inv.append(try config.pathFor(.filecheck, null));
    try inv.append(check_file);
    try inv.append("--input-file");
    try inv.append(input_path);
    return inv;
}

pub fn emitLayoutCase(builder: anytype) Error!void {
    try builder.append(layout_case_mlir);
}

pub fn emitTensorCase(builder: anytype) Error!void {
    try builder.append(tensor_case_mlir);
}

pub fn emitCopyCase(builder: anytype) Error!void {
    try builder.append(copy_case_mlir);
}

pub fn emitMmaCase(builder: anytype) Error!void {
    try builder.append(mma_case_mlir);
}

pub fn emitNegativeCase(builder: anytype) Error!void {
    // Deliberately unbalanced and malformed.  This case is for external verifier
    // negative tests and must not be passed through Builder.finish().
    try builder.rawLine("module {");
    try builder.rawLine("  func.func @negative_case(%arg0: i32) {");
    try builder.rawLine("    %0 = arith.addi %arg0, %arg0 : (i32) -> i32");
    try builder.rawLine("    // expected-error {{malformed return}}");
}

fn tensorValue(meta: tensor.TensorMeta, value: Value) tensor.TensorValue {
    return tensor.TensorValue.init(meta, value, "");
}

fn makeGenericCopyAtom(
    dtype: typing.Numeric,
    src_space: typing.AddressSpace,
    dst_space: typing.AddressSpace,
) Error!atom.CopyAtom {
    const thr = layout.makeCompactLayout(.{4});
    const tv = layout.makeCompactLayout(.{ 4, 1 });
    var tr: atom.Trait = .{ .name = "copy", .thr_id = thr };
    tr = tr.withCopyLayouts(tv, tv);
    return atom.makeCopyAtom(
        atom.OpDescriptor.copyTyped("copy", "generic", "unit", dtype, src_space, dst_space, dtype.width, &.{}),
        tr,
    );
}

fn makeGenericMmaAtom() Error!atom.MmaAtom {
    const thr = layout.makeCompactLayout(.{32});
    const tv = layout.makeCompactLayout(.{ 32, 1 });
    var tr: atom.Trait = .{
        .name = "mma",
        .thr_id = thr,
        .shape_mnk = layout.Tree.fromComptime(.{ 16, 8, 8 }),
    };
    tr = tr.withMmaLayouts(tv, tv, tv);
    return atom.makeMmaAtom(
        atom.OpDescriptor.mmaTyped("mma", "generic", "unit", layout.Tree.fromComptime(.{
            16,
            8,
            8,
        }), typing.Float16, typing.Float16, typing.Float32, &.{.accumulate}),
        tr,
    );
}

fn appendShellQuoted(out: anytype, arg: []const u8) !void {
    if (arg.len == 0) {
        try out.append("''");
        return;
    }
    var needs_quote = false;
    for (arg) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '/' or c == '.' or c == '=' or c == ':' or c == ',')) {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        try out.append(arg);
        return;
    }
    try out.append("'");
    for (arg) |c| {
        if (c == '\'') try out.append("'\\''") else try out.appendByte(c);
    }
    try out.append("'");
}

test "mlir_harness: deterministic layout golden case" {
    var b: Builder(4096) = .{};
    try emitLayoutCase(&b);
    _ = try b.finish();
    const expected = @embedFile("testdata/golden/layout_case.mlir");
    try std.testing.expectEqualStrings(expected, b.slice());
    try validateGeneratedMlir(b.slice());
}

test "mlir_harness: deterministic tensor golden case" {
    var b: Builder(8192) = .{};
    try emitTensorCase(&b);
    _ = try b.finish();
    try expectContains(b.slice(), "cute.memref.load_vec");
    try expectContains(b.slice(), "cute.memref.store_vec");
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.memref_load_vec") == null);
}

test "mlir_harness: deterministic copy and mma golden cases" {
    var b: Builder(16384) = .{};
    try emitCopyCase(&b);
    _ = try b.finish();
    try expectContains(b.slice(), "cute.make_atom()");
    try expectContains(b.slice(), "cute.copy_atom_call(");
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "!cute.tensor") == null);

    b.reset();
    try emitMmaCase(&b);
    _ = try b.finish();
    try expectContains(b.slice(), "cute.make_atom()");
    try expectContains(b.slice(), "cute.mma_atom_call(");
}

test "mlir_harness: negative golden case intentionally fails local structural validation" {
    var b: Builder(2048) = .{};
    try emitNegativeCase(&b);
    const expected = @embedFile("testdata/golden/negative_case.mlir");
    try std.testing.expectEqualStrings(expected, b.slice());
    try std.testing.expectError(
        Error.UnbalancedRegion,
        validateBalancedText(b.slice()),
    );
}

test "mlir_harness: tool invocation builders are deterministic" {
    const config: ToolConfig = .{
        .cute_opt = "/opt/cute/bin/cute-opt",
        .mlir_opt = "/opt/llvm/bin/mlir-opt",
        .filecheck = "/opt/llvm/bin/FileCheck",
    };
    const verify = try cuteOptVerifyInvocation(config, "case.mlir");
    try std.testing.expectEqualStrings("/opt/cute/bin/cute-opt", verify.args()[0]);
    try std.testing.expectEqualStrings("--verify-diagnostics", verify.args()[1]);

    const pipe = try mlirOptPipelineInvocation(
        config,
        "in.mlir",
        "out.mlir",
        "--pass-pipeline=builtin.module(canonicalize,cse)",
    );
    var shell: TextBuffer(512) = .{};
    try pipe.writeShell(&shell);
    try std.testing.expect(std.mem.indexOf(u8, shell.slice(), "mlir-opt") != null);
    try std.testing.expect(std.mem.indexOf(u8, shell.slice(), "--pass-pipeline") != null);
}
