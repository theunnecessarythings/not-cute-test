const std = @import("std");
const layout = @import("layout.zig");
const core = @import("core_static.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Error = core.Error || error{ DivisionByZero, InvalidMode, InvalidSwizzle };
pub const max_modes = layout.max_leaves;
pub const ModePath = layout.BoundedList(usize, max_modes);

/// Exact static counterpart of CuteDSL core.py Ratio.
pub const Ratio = struct {
    numerator: Scalar,
    denominator: Scalar,

    pub fn init(numerator: Scalar, denominator: Scalar) Error!Ratio {
        if (denominator == 0) return Error.DivisionByZero;
        return .{ .numerator = numerator, .denominator = denominator };
    }

    pub fn reduced(self: Ratio) Ratio {
        if (self.numerator == 0) return .{ .numerator = 0, .denominator = 1 };
        var a = absScalar(self.numerator);
        var b = absScalar(self.denominator);
        while (b != 0) {
            const r = a % b;
            a = b;
            b = r;
        }
        const g: Scalar = @intCast(a);
        var n = @divTrunc(self.numerator, g);
        var d = @divTrunc(self.denominator, g);
        if (d < 0) {
            n = -n;
            d = -d;
        }
        return .{ .numerator = n, .denominator = d };
    }

    pub fn isIntegral(self: Ratio) bool {
        return @mod(self.numerator, self.denominator) == 0;
    }

    pub fn mul(self: Ratio, other: Ratio) Error!Ratio {
        const n = std.math.mul(Scalar, self.numerator, other.numerator) catch return Error.Overflow;
        const d = std.math.mul(Scalar, self.denominator, other.denominator) catch return Error.Overflow;
        return (try init(n, d)).reduced();
    }

    pub fn mulInt(self: Ratio, value: Scalar) Error!Ratio {
        const n = std.math.mul(Scalar, self.numerator, value) catch return Error.Overflow;
        return (try init(n, self.denominator)).reduced();
    }

    pub fn toIntFloor(self: Ratio) Scalar {
        return @divFloor(self.numerator, self.denominator);
    }

    pub fn eql(self: Ratio, other: Ratio) bool {
        const a = self.reduced();
        const b = other.reduced();
        return a.numerator == b.numerator and a.denominator == b.denominator;
    }
};

pub const Scale = union(enum) {
    int: Scalar,
    ratio: Ratio,

    pub fn asRatio(self: Scale) Ratio {
        return switch (self) {
            .int => |v| Ratio.init(v, 1) catch unreachable,
            .ratio => |r| r,
        };
    }

    pub fn mul(self: Scale, other: Scale) Error!Scale {
        const r = try self.asRatio().mul(other.asRatio());
        if (r.isIntegral()) return .{ .int = r.toIntFloor() };
        return .{ .ratio = r };
    }

    pub fn eql(self: Scale, other: Scale) bool {
        return self.asRatio().eql(other.asRatio());
    }
};

/// Static counterpart of CuteDSL core.py ScaledBasis.
pub const ScaledBasis = struct {
    value: Scale,
    mode: ModePath,

    pub fn initInt(value: Scalar, modes: []const usize) Error!ScaledBasis {
        return init(.{ .int = value }, modes);
    }

    pub fn initRatio(value: Ratio, modes: []const usize) Error!ScaledBasis {
        return init(.{ .ratio = value.reduced() }, modes);
    }

    pub fn init(value: Scale, modes: []const usize) Error!ScaledBasis {
        var path: ModePath = .{};
        for (modes) |m| try path.append(m);
        return .{ .value = value, .mode = path };
    }

    pub fn E(mode: usize) Error!ScaledBasis {
        return initInt(1, &.{mode});
    }

    pub fn EPath(modes: []const usize) Error!ScaleOrBasis {
        if (modes.len == 0) return .{ .int = 1 };
        return .{ .basis = try initInt(1, modes) };
    }

    pub fn scale(self: ScaledBasis, factor: Scale) Error!ScaledBasis {
        return .{ .value = try factor.mul(self.value), .mode = self.mode };
    }

    pub fn basisValue(self: ScaledBasis) Scale {
        return self.value;
    }

    pub fn eql(self: ScaledBasis, other: ScaledBasis) bool {
        if (!self.value.eql(other.value)) return false;
        if (self.mode.len != other.mode.len) return false;
        for (self.mode.slice(), 0..) |m, i| if (m != other.mode.at(i)) return false;
        return true;
    }
};

pub const ScaleOrBasis = union(enum) {
    int: Scalar,
    basis: ScaledBasis,
};

/// Apply ScaledBasis.mode to a static tree, mirroring core.py basis_get.
pub fn basisGet(basis: ScaledBasis, tree: *const layout.Tree) Error!layout.Tree {
    var current = tree.*;
    for (basis.mode.slice()) |mode| current = try current.topMode(mode);
    return current;
}

/// Static swizzle descriptor and evaluator corresponding to core.py Swizzle.
pub const Swizzle = struct {
    num_bits: u6,
    num_base: u6,
    num_shift: i8,

    pub fn init(num_bits: u6, num_base: u6, num_shift: i8) Error!Swizzle {
        if (num_bits == 0) return Error.InvalidSwizzle;
        const abs_shift: u8 = @intCast(if (num_shift < 0) -num_shift else num_shift);
        if (@as(u16, num_base) + @as(u16, num_bits) + @as(u16, abs_shift) >= 128) return Error.InvalidSwizzle;
        return .{ .num_bits = num_bits, .num_base = num_base, .num_shift = num_shift };
    }

    pub fn eql(self: Swizzle, other: Swizzle) bool {
        return self.num_bits == other.num_bits and self.num_base == other.num_base and self.num_shift == other.num_shift;
    }

    pub fn apply(self: Swizzle, offset: Unsigned) Unsigned {
        const bit_mask: Unsigned = (@as(Unsigned, 1) << self.num_bits) - 1;
        if (self.num_shift >= 0) {
            const shift: u7 = @intCast(self.num_shift);
            const y_mask = bit_mask << (@as(u7, self.num_base) + shift);
            return offset ^ ((offset & y_mask) >> shift);
        } else {
            const shift: u7 = @intCast(-self.num_shift);
            const y_mask = bit_mask << self.num_base;
            return offset ^ ((offset & y_mask) << shift);
        }
    }
};

fn absScalar(value: Scalar) Unsigned {
    if (value < 0) return @intCast(-value);
    return @intCast(value);
}

test "basis: ratio reduction and multiplication" {
    const r = (try Ratio.init(6, -8)).reduced();
    try std.testing.expectEqual(@as(Scalar, -3), r.numerator);
    try std.testing.expectEqual(@as(Scalar, 4), r.denominator);
    const m = try r.mulInt(8);
    try std.testing.expect(m.eql(try Ratio.init(-6, 1)));
}

test "basis: scaled basis equality scaling and basis_get" {
    const e = try ScaledBasis.E(1);
    const s = try e.scale(.{ .int = 4 });
    const expected = try ScaledBasis.initInt(4, &.{1});
    try std.testing.expect(s.eql(expected));

    const t = layout.Tree.fromComptime(.{ 10, .{ 20, 30 } });
    const sub = try basisGet(try ScaledBasis.initInt(1, &.{ 1, 0 }), &t);
    try std.testing.expectEqualSlices(Scalar, &.{20}, (try sub.flattenLeaves()).slice());
}

test "basis: swizzle descriptor equality and xor mapping" {
    const sw = try Swizzle.init(2, 2, 4);
    try std.testing.expect(sw.eql(try Swizzle.init(2, 2, 4)));
    const offset: Unsigned = @as(Unsigned, 0b11) << 6;
    const mapped = sw.apply(offset);
    try std.testing.expectEqual(offset ^ (@as(Unsigned, 0b11) << 2), mapped);
}
