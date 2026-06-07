module {
  func.func @tensor_scalar_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: !cute.coord<"(2)">, %arg2: f32) -> f32 {
    %0 = cute.memref.load(%arg0, %arg1) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">, !cute.coord<"(2)">) -> f32
    cute.memref.store(%arg0, %arg1, %arg2) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">, !cute.coord<"(2)">, f32) -> ()
    return %0 : f32
  }
}

