const std = @import("std");
const layout = @import("layout.zig");
const atom = @import("atom.zig");

pub const Error = atom.Error;
pub const Op = atom.OpDescriptor;
pub const MmaOp = atom.OpDescriptor;
pub const CopyOp = atom.OpDescriptor;
pub const Atom = atom.Atom;
pub const MmaAtom = atom.MmaAtom;
pub const CopyAtom = atom.CopyAtom;
pub const TiledMma = atom.TiledMma;
pub const TiledCopy = atom.TiledCopy;

pub fn makeAtom(desc: atom.OpDescriptor, tr: atom.Trait) Error!atom.Atom {
    return .{ .desc = desc, .trait = tr };
}
pub fn makeMmaAtom(desc: atom.OpDescriptor, tr: atom.Trait) Error!atom.MmaAtom {
    return atom.makeMmaAtom(desc, tr);
}
pub fn makeCopyAtom(desc: atom.OpDescriptor, tr: atom.Trait) Error!atom.CopyAtom {
    return atom.makeCopyAtom(desc, tr);
}
pub fn makeTiledMma(
    mma: atom.MmaAtom,
    atom_layout_mnk: layout.Layout,
    permutation_mnk: ?layout.Tree,
) Error!atom.TiledMma {
    return atom.makeTiledMma(mma, atom_layout_mnk, permutation_mnk);
}
pub fn makeTiledCopy(
    copy: atom.CopyAtom,
    layout_tv_tiled: layout.Layout,
    tiler_mn: layout.Tree,
) Error!atom.TiledCopy {
    return atom.makeTiledCopy(copy, layout_tv_tiled, tiler_mn);
}
pub fn makeTiledCopyTv(
    copy: atom.CopyAtom,
    thr_layout: layout.Layout,
    val_layout: layout.Layout,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyTv(copy, thr_layout, val_layout);
}
pub fn makeCotiledCopy(
    copy: atom.CopyAtom,
    tiled: atom.TiledCopy,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyS(copy, tiled);
}
pub fn makeTiledCopyA(
    copy: atom.CopyAtom,
    tiled_mma: atom.TiledMma,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyA(copy, tiled_mma);
}
pub fn makeTiledCopyB(
    copy: atom.CopyAtom,
    tiled_mma: atom.TiledMma,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyB(copy, tiled_mma);
}
pub fn makeTiledCopyC(
    copy: atom.CopyAtom,
    tiled_mma: atom.TiledMma,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyC(copy, tiled_mma);
}
pub fn makeTiledCopyS(
    copy: atom.CopyAtom,
    tiled_copy: atom.TiledCopy,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyS(copy, tiled_copy);
}
pub fn makeTiledCopyD(
    copy: atom.CopyAtom,
    tiled_copy: atom.TiledCopy,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyD(copy, tiled_copy);
}
pub fn makeTiledCopyCAtom(
    copy: atom.CopyAtom,
    tiled_mma: atom.TiledMma,
) Error!atom.TiledCopy {
    return atom.makeTiledCopyC(copy, tiled_mma);
}
pub fn copyAtomCall(
    builder: anytype,
    copy: atom.CopyAtom,
    src: []const atom.TensorOperand,
    dst: []const atom.TensorOperand,
    pred: ?atom.TensorOperand,
) Error!void {
    return atom.copyAtomCall(builder, copy, src, dst, pred);
}
pub fn mmaAtomCall(
    builder: anytype,
    mma: atom.MmaAtom,
    d: atom.TensorOperand,
    a: []const atom.TensorOperand,
    b: []const atom.TensorOperand,
    c: atom.TensorOperand,
) Error!void {
    return atom.mmaAtomCall(builder, mma, d, a, b, c);
}

test "atom_api: snake-case wrappers call real atom constructors" {
    const desc = atom.OpDescriptor.copy(
        "copy",
        "generic",
        @import("typing.zig").Float32,
    );
    const tv = try layout.Layout.makeCompact(layout.Tree.fromComptime(.{1}));
    const tr: atom.Trait = .{
        .name = "copy_trait",
        .thr_id = tv,
        .layout_src_tv = tv,
        .layout_dst_tv = tv,
    };
    const c = try makeCopyAtom(desc, tr);
    try std.testing.expectEqual(atom.OpKind.copy, c.atom.kind());
}
