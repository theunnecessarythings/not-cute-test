const std = @import("std");
const layout = @import("layout.zig");

pub const Scalar = layout.Scalar;
pub const Error = layout.Error || error{
    InvalidMap,
    NotComposable,
    ImagesOverlap,
    NotComplementable,
};

pub const max_rank = layout.max_leaves;

pub fn SmallVec(comptime T: type) type {
    return layout.BoundedList(T, max_rank);
}

pub const FinMorphism = struct {
    /// Domain cardinality of the pointed finite set, excluding base point.
    domain: usize,
    /// Codomain cardinality of the pointed finite set, excluding base point.
    codomain: usize,
    /// 1-based codomain indices, with 0 denoting the base point.
    map: SmallVec(usize) = .{},

    pub fn init(domain: usize, codomain: usize, map: []const usize) Error!FinMorphism {
        if (map.len != domain) return Error.InvalidMap;
        var out: FinMorphism = .{ .domain = domain, .codomain = codomain };
        var seen = [_]bool{false} ** max_rank;
        if (codomain > max_rank) return Error.OutOfCapacity;
        for (map) |value| {
            if (value > codomain) return Error.InvalidMap;
            if (value != 0) {
                if (seen[value - 1]) return Error.InvalidMap;
                seen[value - 1] = true;
            }
            try out.map.append(value);
        }
        return out;
    }

    pub fn compose(self: FinMorphism, beta: FinMorphism) Error!FinMorphism {
        if (self.codomain != beta.domain) return Error.NotComposable;
        var result: SmallVec(usize) = .{};
        for (self.map.slice()) |value| {
            try result.append(if (value == 0) 0 else beta.map.at(value - 1));
        }
        return init(self.domain, beta.codomain, result.slice());
    }

    pub fn sum(self: FinMorphism, beta: FinMorphism) Error!FinMorphism {
        var result: SmallVec(usize) = .{};
        try result.appendSlice(self.map.slice());
        for (beta.map.slice()) |value| try result.append(if (value == 0) 0 else value + self.codomain);
        return init(
            self.domain + beta.domain,
            self.codomain + beta.codomain,
            result.slice(),
        );
    }

    pub fn imagesAreDisjoint(self: FinMorphism, beta: FinMorphism) Error!bool {
        if (self.codomain != beta.codomain) return Error.NotComposable;
        var seen = [_]bool{false} ** max_rank;
        if (self.codomain > max_rank) return Error.OutOfCapacity;
        for (self.map.slice()) |value| {
            if (value != 0) seen[value - 1] = true;
        }
        for (beta.map.slice()) |value| {
            if (value != 0 and seen[value - 1]) return false;
        }
        return true;
    }

    pub fn wedge(self: FinMorphism, beta: FinMorphism) Error!FinMorphism {
        if (!try self.imagesAreDisjoint(beta)) return Error.ImagesOverlap;
        var result: SmallVec(usize) = .{};
        try result.appendSlice(self.map.slice());
        try result.appendSlice(beta.map.slice());
        return init(self.domain + beta.domain, self.codomain, result.slice());
    }
};

pub const TupleMorphism = struct {
    domain: SmallVec(Scalar) = .{},
    codomain: SmallVec(Scalar) = .{},
    map: FinMorphism,

    pub fn init(
        domain: []const Scalar,
        codomain: []const Scalar,
        map: []const usize,
    ) Error!TupleMorphism {
        if (domain.len > max_rank or codomain.len > max_rank)
            return Error.OutOfCapacity;
        var d: SmallVec(Scalar) = .{};
        var c: SmallVec(Scalar) = .{};
        for (domain) |x| {
            if (x <= 0) return Error.InvalidShape;
            try d.append(x);
        }
        for (codomain) |x| {
            if (x <= 0) return Error.InvalidShape;
            try c.append(x);
        }
        const fin = try FinMorphism.init(domain.len, codomain.len, map);
        for (fin.map.slice(), 0..) |value, i| {
            if (value != 0 and domain[i] != codomain[value - 1])
                return Error.InvalidMap;
        }
        return .{ .domain = d, .codomain = c, .map = fin };
    }

    pub fn compose(self: TupleMorphism, g: TupleMorphism) Error!TupleMorphism {
        if (!sameScalarSlice(self.codomain.slice(), g.domain.slice()))
            return Error.NotComposable;
        const composed = try self.map.compose(g.map);
        return init(self.domain.slice(), g.codomain.slice(), composed.map.slice());
    }

    pub fn sum(self: TupleMorphism, g: TupleMorphism) Error!TupleMorphism {
        var d: SmallVec(Scalar) = .{};
        var c: SmallVec(Scalar) = .{};
        try d.appendSlice(self.domain.slice());
        try d.appendSlice(g.domain.slice());
        try c.appendSlice(self.codomain.slice());
        try c.appendSlice(g.codomain.slice());
        const summed = try self.map.sum(g.map);
        return init(d.slice(), c.slice(), summed.map.slice());
    }

    pub fn restrict(self: TupleMorphism, subtuple: []const usize) Error!TupleMorphism {
        var d: SmallVec(Scalar) = .{};
        var m: SmallVec(usize) = .{};
        var prev: usize = 0;
        for (subtuple) |one_based| {
            if (one_based == 0 or one_based > self.domain.len or one_based <= prev)
                return Error.InvalidSelection;
            prev = one_based;
            try d.append(self.domain.at(one_based - 1));
            try m.append(self.map.map.at(one_based - 1));
        }
        return init(d.slice(), self.codomain.slice(), m.slice());
    }

    pub fn factorize(self: TupleMorphism, subtuple: []const usize) Error!TupleMorphism {
        var c: SmallVec(Scalar) = .{};
        var present = [_]bool{false} ** max_rank;
        var prev: usize = 0;
        for (subtuple) |one_based| {
            if (one_based == 0 or one_based > self.codomain.len or one_based <= prev)
                return Error.InvalidSelection;
            prev = one_based;
            present[one_based - 1] = true;
            try c.append(self.codomain.at(one_based - 1));
        }
        var m: SmallVec(usize) = .{};
        for (self.map.map.slice()) |value| {
            if (value == 0) {
                try m.append(0);
            } else {
                if (!present[value - 1]) return Error.InvalidSelection;
                var missing_before: usize = 0;
                for (0..(value - 1)) |i| {
                    if (!present[i]) missing_before += 1;
                }
                try m.append(value - missing_before);
            }
        }
        return init(self.domain.slice(), c.slice(), m.slice());
    }

    pub fn squeeze(self: TupleMorphism) Error!TupleMorphism {
        var domain_sub: SmallVec(usize) = .{};
        for (self.domain.slice(), 0..) |value, i| {
            if (value != 1) try domain_sub.append(i + 1);
        }
        const restricted = try self.restrict(domain_sub.slice());
        var codomain_sub: SmallVec(usize) = .{};
        for (restricted.codomain.slice(), 0..) |value, i| {
            if (value != 1) try codomain_sub.append(i + 1);
        }
        return restricted.factorize(codomain_sub.slice());
    }

    pub fn concat(self: TupleMorphism, g: TupleMorphism) Error!TupleMorphism {
        if (!sameScalarSlice(self.codomain.slice(), g.codomain.slice()))
            return Error.NotComposable;
        const wedged = try self.map.wedge(g.map);
        var d: SmallVec(Scalar) = .{};
        try d.appendSlice(self.domain.slice());
        try d.appendSlice(g.domain.slice());
        return init(d.slice(), self.codomain.slice(), wedged.map.slice());
    }

    pub fn complement(self: TupleMorphism) Error!TupleMorphism {
        for (self.map.map.slice()) |value| {
            if (value == 0) return Error.NotComplementable;
        }
        var image = [_]bool{false} ** max_rank;
        for (self.map.map.slice()) |value| {
            image[value - 1] = true;
        }
        var d: SmallVec(Scalar) = .{};
        var m: SmallVec(usize) = .{};
        for (self.codomain.slice(), 0..) |extent, i| {
            if (!image[i]) {
                try d.append(extent);
                try m.append(i + 1);
            }
        }
        return init(d.slice(), self.codomain.slice(), m.slice());
    }

    pub fn isIsomorphism(self: TupleMorphism) bool {
        if (self.domain.len != self.codomain.len) return false;
        var seen = [_]bool{false} ** max_rank;
        for (self.map.map.slice()) |value| {
            if (value == 0 or value > self.domain.len) return false;
            if (seen[value - 1]) return false;
            seen[value - 1] = true;
        }
        return true;
    }
};

fn sameScalarSlice(a: []const Scalar, b: []const Scalar) bool {
    if (a.len != b.len) return false;
    for (a, 0..) |x, i| {
        if (x != b[i]) return false;
    }
    return true;
}

test "morphism: pointed finite set composition" {
    const f = try FinMorphism.init(3, 4, &.{ 2, 0, 4 });
    const g = try FinMorphism.init(4, 2, &.{ 1, 2, 0, 0 });
    const h = try f.compose(g);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0, 0 }, h.map.slice());
}

test "morphism: tuple complement is disjoint and concatenates to iso" {
    const f = try TupleMorphism.init(&.{ 2, 5 }, &.{ 2, 3, 5, 7 }, &.{ 1, 3 });
    const c = try f.complement();
    try std.testing.expectEqualSlices(Scalar, &.{ 3, 7 }, c.domain.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, c.map.map.slice());
    const all = try f.concat(c);
    try std.testing.expect(all.isIsomorphism());
}
