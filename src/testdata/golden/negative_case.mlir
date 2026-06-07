module {
  func.func @negative_case(%arg0: i32) {
    %0 = arith.addi %arg0, %arg0 : (i32) -> i32
    // expected-error {{malformed return}}
