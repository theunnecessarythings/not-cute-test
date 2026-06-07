// upstream parity: vector TensorSSA/elementwise-add flow
!memref_vec = !cute.memref<f32, gmem, align<16>, "(4):(1)">
module {
  gpu.module @notcute_upstream {
    gpu.func @elementwise_add(%a: !memref_vec, %b: !memref_vec, %c: !memref_vec) kernel {
      %av = cute.memref.load_vec(%a) : (!memref_vec) -> vector<4xf32>
      %bv = cute.memref.load_vec(%b) : (!memref_vec) -> vector<4xf32>
      %cv = arith.addf %av, %bv : vector<4xf32>
      cute.memref.store_vec(%cv, %c) : (vector<4xf32>, !memref_vec) -> ()
      gpu.return
    }
  }
}
