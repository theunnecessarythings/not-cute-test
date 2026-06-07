module {
  func.func @memref_load_case(%arg0: !cute.memref<f32, gmem, align<16>, "(2,3):(3,1)">, %arg1: !cute.coord<"(2,3)">) -> f32 {
    %0 = cute.memref.load(%arg0, %arg1) : (!cute.memref<f32, gmem, align<16>, "(2,3):(3,1)">, !cute.coord<"(2,3)">) -> f32
    return %0 : f32
  }
}
