// upstream parity: print.ipynb static/dynamic value flow
module {
  gpu.module @notcute_upstream {
    gpu.func @print_values() kernel {
      %a = arith.constant 8 : i32
      %b = arith.constant 2 : i32
      %c = arith.addi %a, %b : i32
      gpu.return
    }
  }
}
