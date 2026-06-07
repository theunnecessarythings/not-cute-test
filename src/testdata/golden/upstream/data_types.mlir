// upstream parity: data_types.ipynb numeric operations
module {
  gpu.module @notcute_upstream {
    gpu.func @data_types() kernel {
      %a = arith.constant 10 : i32
      %b = arith.constant 3 : i32
      %c = arith.addi %a, %b : i32
      %d = arith.muli %a, %b : i32
      gpu.return
    }
  }
}
