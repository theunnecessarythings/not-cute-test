module {
  func.func @routed_tensor_vector(%arg0: !cute.memref<f32, gmem, "(4):(1)">) {
    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, "(4):(1)">) -> vector<4xf32>
    cute.memref.store_vec(%0, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, "(4):(1)">) -> ()
    return
  }
}
