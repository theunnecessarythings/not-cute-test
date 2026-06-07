module {
  func.func @identity_tensor_case() {
    %0 = cute.make_shape() : () -> !cute.shape<"(2,3)">
    %1 = cute.make_identity_tensor(%0) : !cute.coord_tensor<"(0,0)", "(2,3):(1@0,1@1)">
    return
  }
}
