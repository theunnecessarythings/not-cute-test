// upstream parity: cute_layout_algebra.ipynb layout construction
module {
  func.func @cute_layout_algebra() {
    %0 = cute.make_shape() : () -> !cute.shape<"(2,3)">
    %1 = cute.make_stride() : () -> !cute.stride<"(3,1)">
    %2 = cute.make_layout(%0, %1) : !cute.layout<"(2,3):(3,1)">
    return
  }
}
