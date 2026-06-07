const std = @import("std");
const layout = @import("layout.zig");
const ops = @import("mlir_ops.zig");

pub const mlir_ops = ops;
pub const max_results = 16;

pub const Error = layout.Error || error{
    InvalidMlirIdentifier,
    InvalidMlirType,
    InvalidMlirAttribute,
    InvalidMlirString,
    InvalidMlirOperation,
    TooManyResults,
    UnterminatedString,
    UnbalancedRegion,
    RegionUnderflow,
    MissingTerminator,
};

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

pub const Operand = union(enum) {
    value: Value,
    raw: RawValue,

    pub fn arg(comptime index: usize) Operand {
        return .{ .raw = .{ .text = comptimeArgName(index) } };
    }

    pub fn named(text: []const u8) Operand {
        return .{ .raw = .{ .text = text } };
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

pub const OperationSpec = struct {
    name: []const u8,
    operands: []const Operand = &.{},
    attrs: []const Attribute = &.{},
    operand_types: []const Type = &.{},
    result_types: []const Type = &.{},
    /// Emit MLIR generic quoted operation form: `"dialect.op"(...) ...`.
    quoted: bool = false,
};

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
