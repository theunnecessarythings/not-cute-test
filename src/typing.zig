const std = @import("std");
const layout = @import("layout.zig");
const mlir = @import("mlir_text.zig");

pub const Error = layout.Error || mlir.Error || error{ InvalidNumericKind, InvalidDivisibility };

pub const AddressSpace = enum(u32) {
    generic = 0,
    gmem = 1,
    smem = 3,
    tmem = 5,

    pub fn mlirName(self: AddressSpace) []const u8 {
        return switch (self) {
            .generic => "generic",
            .gmem => "gmem",
            .smem => "smem",
            .tmem => "tmem",
        };
    }
};

pub const NumericKind = enum {
    boolean,
    signed_int,
    unsigned_int,
    float,
    bfloat,
    tfloat,
    fp8_e5m2,
    fp8_e4m3fn,
    fp8_e4m3b11fnuz,
    fp8_e4m3,
    fp8_e8m0fnu,
    fp4_e2m1fn,
    fp6_e2m3fn,
    fp6_e3m2fn,
};

pub const Numeric = struct {
    name: []const u8,
    width: u16,
    kind: NumericKind,
    mlir_type: []const u8,

    pub fn bytes(self: Numeric) usize {
        return @max(@as(usize, 1), (@as(usize, self.width) + 7) / 8);
    }

    pub fn isInteger(self: Numeric) bool {
        return self.kind == .signed_int or self.kind == .unsigned_int or self.kind == .boolean;
    }

    pub fn isFloat(self: Numeric) bool {
        return !self.isInteger();
    }
};

pub const Boolean: Numeric = .{
    .name = "Boolean",
    .width = 1,
    .kind = .boolean,
    .mlir_type = "i1",
};
pub const Int4: Numeric = .{
    .name = "Int4",
    .width = 4,
    .kind = .signed_int,
    .mlir_type = "i4",
};
pub const Int8: Numeric = .{
    .name = "Int8",
    .width = 8,
    .kind = .signed_int,
    .mlir_type = "i8",
};
pub const Int16: Numeric = .{
    .name = "Int16",
    .width = 16,
    .kind = .signed_int,
    .mlir_type = "i16",
};
pub const Int32: Numeric = .{
    .name = "Int32",
    .width = 32,
    .kind = .signed_int,
    .mlir_type = "i32",
};
pub const Int64: Numeric = .{
    .name = "Int64",
    .width = 64,
    .kind = .signed_int,
    .mlir_type = "i64",
};
pub const Int128: Numeric = .{
    .name = "Int128",
    .width = 128,
    .kind = .signed_int,
    .mlir_type = "i128",
};
pub const Uint8: Numeric = .{
    .name = "Uint8",
    .width = 8,
    .kind = .unsigned_int,
    .mlir_type = "i8",
};
pub const Uint16: Numeric = .{
    .name = "Uint16",
    .width = 16,
    .kind = .unsigned_int,
    .mlir_type = "i16",
};
pub const Uint32: Numeric = .{
    .name = "Uint32",
    .width = 32,
    .kind = .unsigned_int,
    .mlir_type = "i32",
};
pub const Uint64: Numeric = .{
    .name = "Uint64",
    .width = 64,
    .kind = .unsigned_int,
    .mlir_type = "i64",
};
pub const Uint128: Numeric = .{
    .name = "Uint128",
    .width = 128,
    .kind = .unsigned_int,
    .mlir_type = "i128",
};
pub const Float64: Numeric = .{
    .name = "Float64",
    .width = 64,
    .kind = .float,
    .mlir_type = "f64",
};
pub const Float32: Numeric = .{
    .name = "Float32",
    .width = 32,
    .kind = .float,
    .mlir_type = "f32",
};
pub const TFloat32: Numeric = .{
    .name = "TFloat32",
    .width = 32,
    .kind = .tfloat,
    .mlir_type = "tf32",
};
pub const Float16: Numeric = .{
    .name = "Float16",
    .width = 16,
    .kind = .float,
    .mlir_type = "f16",
};
pub const BFloat16: Numeric = .{
    .name = "BFloat16",
    .width = 16,
    .kind = .bfloat,
    .mlir_type = "bf16",
};
pub const Float8E5M2: Numeric = .{
    .name = "Float8E5M2",
    .width = 8,
    .kind = .fp8_e5m2,
    .mlir_type = "f8E5M2",
};
pub const Float8E4M3FN: Numeric = .{
    .name = "Float8E4M3FN",
    .width = 8,
    .kind = .fp8_e4m3fn,
    .mlir_type = "f8E4M3FN",
};
pub const Float8E4M3B11FNUZ: Numeric = .{
    .name = "Float8E4M3B11FNUZ",
    .width = 8,
    .kind = .fp8_e4m3b11fnuz,
    .mlir_type = "f8E4M3B11FNUZ",
};
pub const Float8E4M3: Numeric = .{
    .name = "Float8E4M3",
    .width = 8,
    .kind = .fp8_e4m3,
    .mlir_type = "f8E4M3",
};
pub const Float8E8M0FNU: Numeric = .{
    .name = "Float8E8M0FNU",
    .width = 8,
    .kind = .fp8_e8m0fnu,
    .mlir_type = "f8E8M0FNU",
};
pub const Float4E2M1FN: Numeric = .{
    .name = "Float4E2M1FN",
    .width = 4,
    .kind = .fp4_e2m1fn,
    .mlir_type = "f4E2M1FN",
};
pub const Float6E2M3FN: Numeric = .{
    .name = "Float6E2M3FN",
    .width = 6,
    .kind = .fp6_e2m3fn,
    .mlir_type = "f6E2M3FN",
};
pub const Float6E3M2FN: Numeric = .{
    .name = "Float6E3M2FN",
    .width = 6,
    .kind = .fp6_e3m2fn,
    .mlir_type = "f6E3M2FN",
};

pub const SymInt = struct {
    width: u16 = 32,
    divisibility: u64 = 1,
    symbol: ?[]const u8 = null,

    pub fn init(width: u16, divisibility: u64, symbol: ?[]const u8) Error!SymInt {
        if (width != 32 and width != 64) return Error.InvalidNumericKind;
        if (divisibility == 0) return Error.InvalidDivisibility;
        return .{ .width = width, .divisibility = divisibility, .symbol = symbol };
    }

    pub fn modulo(self: SymInt, other: SymInt) SymInt {
        return .{
            .width = @max(self.width, other.width),
            .divisibility = gcd(self.divisibility, other.divisibility),
            .symbol = null,
        };
    }

    pub fn mul(self: SymInt, other: SymInt) SymInt {
        return .{
            .width = @max(self.width, other.width),
            .divisibility = self.divisibility * other.divisibility,
            .symbol = null,
        };
    }

    pub fn writeMlir(self: SymInt, out: anytype) Error!void {
        try out.append("!cute.int_tuple<\"?");
        if (self.width == 32) {
            try out.append("{div=");
        } else {
            try out.append("{i");
            try out.appendUnsigned(self.width);
            try out.append(" div=");
        }
        try out.appendUnsigned(@intCast(self.divisibility));
        try out.append("}\">");
    }
};

pub fn symInt32(divisibility: u64, symbol: ?[]const u8) Error!SymInt {
    return SymInt.init(32, divisibility, symbol);
}

pub fn symInt64(divisibility: u64, symbol: ?[]const u8) Error!SymInt {
    return SymInt.init(64, divisibility, symbol);
}

pub const TypedTensor = struct {
    dtype: Numeric,
    shape: layout.Tree,
    stride: layout.Tree,
    memspace: AddressSpace = .generic,
    assumed_align: ?usize = null,

    pub fn init(
        dtype: Numeric,
        shape: layout.Tree,
        stride: layout.Tree,
        memspace: AddressSpace,
        assumed_align: ?usize,
    ) Error!TypedTensor {
        if (!shape.sameProfile(&stride)) return Error.ProfileMismatch;
        try shape.assertPositive();
        return .{
            .dtype = dtype,
            .shape = shape,
            .stride = stride,
            .memspace = memspace,
            .assumed_align = assumed_align orelse dtype.bytes(),
        };
    }

    pub fn elementType(self: TypedTensor) Numeric {
        return self.dtype;
    }

    pub fn writeMlirType(self: TypedTensor, out: anytype) Error!void {
        try out.append("!cute.memref<ptr<");
        try out.append(if (self.dtype.width == 1) "i8" else self.dtype.mlir_type);
        try out.append(", ");
        try out.append(self.memspace.mlirName());
        try out.append(", align=");
        try out.appendUnsigned(self.assumed_align orelse self.dtype.bytes());
        try out.append(">, layout<shape=");
        try writeTreeLiteral(&self.shape, out);
        try out.append(", stride=");
        try writeTreeLiteral(&self.stride, out);
        try out.append(">>");
    }
};

fn writeTreeLiteral(tree: *const layout.Tree, out: anytype) Error!void {
    try writeTreeLiteralSub(tree, tree.root, out);
}

fn writeTreeLiteralSub(tree: *const layout.Tree, id: u16, out: anytype) Error!void {
    switch (tree.nodes.at(id)) {
        .leaf => |v| try out.appendSigned(v),
        .tuple => |span| {
            try out.append("(");
            for (0..span.len) |i| {
                if (i != 0) try out.append(",");
                try writeTreeLiteralSub(tree, tree.children.at(span.start + i), out);
            }
            try out.append(")");
        },
    }
}

fn gcd(a_in: u64, b_in: u64) u64 {
    var a = a_in;
    var b = b_in;
    while (b != 0) {
        const r = a % b;
        a = b;
        b = r;
    }
    return a;
}

test "typing: numeric descriptors preserve CuteDSL names" {
    try std.testing.expectEqual(@as(u16, 32), Float32.width);
    try std.testing.expectEqualStrings("f32", Float32.mlir_type);
    try std.testing.expect(Int32.isInteger());
    try std.testing.expect(BFloat16.isFloat());
}

test "typing: symbolic int and typed tensor MLIR strings" {
    var out: mlir.TextBuffer(512) = .{};
    const s = try symInt64(8, "N");
    try s.writeMlir(&out);
    try std.testing.expectEqualStrings(
        "!cute.int_tuple<\"?{i64 div=8}\">",
        out.slice(),
    );

    out.clear();
    const tt = try TypedTensor.init(
        Float32,
        layout.Tree.fromComptime(.{ 16, 8 }),
        layout.Tree.fromComptime(.{ 8, 1 }),
        .gmem,
        null,
    );
    try tt.writeMlirType(&out);
    try std.testing.expectEqualStrings(
        "!cute.memref<ptr<f32, gmem, align=4>, layout<shape=(16,8), stride=(8,1)>>",
        out.slice(),
    );
}
