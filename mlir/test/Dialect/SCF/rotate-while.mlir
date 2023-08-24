// RUN: mlir-opt --scf-rotate-while --split-input-file %s | FileCheck %s

// CHECK-LABEL: @basic
func.func @basic() {
  // CHECK-NEXT: %[[GUARD:.+]] = "test.cond"()
  // CHECK-NEXT: scf.if %[[GUARD]] {
  // CHECK-NEXT:   scf.while : () -> () {
  // CHECK-NEXT:     "test.body"()
  // CHECK-NEXT:     %[[COND:.+]] = "test.cond"()
  // CHECK-NEXT:     scf.condition(%[[COND]])
  // CHECK-NEXT:   } do {
  // CHECK-NEXT:     scf.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  scf.while : () -> () {
    %cond = "test.cond"() : () -> i1
    scf.condition(%cond)
  } do {
    "test.body"() : () -> ()
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: @output_before_result
func.func @output_before_result() -> f32 {
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 0.0
  // CHECK-NEXT: %[[GUARD:.+]]:2 = "test.before"(%[[CST]])
  // CHECK-NEXT: %[[IF:.+]] = scf.if %[[GUARD]]#0 -> {{.+}} {
  // CHECK-NEXT:   %[[WHILE:.+]] = scf.while (%[[B_ARG:.+]] = %[[GUARD]]#1) : {{.+}} {
  // CHECK-NEXT:     "test.after"(%[[B_ARG]])
  // CHECK-NEXT:     %[[BEFORE:.+]]:2 = "test.before"(%[[B_ARG]])
  // CHECK-NEXT:     scf.condition(%[[BEFORE]]#0) %[[BEFORE]]#1
  // CHECK-NEXT:   } do {
  // CHECK-NEXT:   ^bb0(%[[A_ARG:.+]]: {{.+}}):
  // CHECK-NEXT:     scf.yield %[[A_ARG]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   scf.yield %[[WHILE]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %[[GUARD]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[IF]]
  %cst = arith.constant 0.0 : f32
  %while = scf.while (%b_arg = %cst) : (f32) -> (f32) {
    %before:2 = "test.before"(%b_arg) : (f32) -> (i1, f32)
    scf.condition(%before#0) %before#1 : f32
  } do {
  ^bb0(%a_arg: f32):
    "test.after"(%a_arg) : (f32) -> ()
    scf.yield %a_arg : f32
  }
  return %while : f32
}

// -----

// CHECK-LABEL: @output_after_result
func.func @output_after_result() -> f32 {
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 0.0
  // CHECK-NEXT: %[[GUARD:.+]] = "test.before"(%[[CST]])
  // CHECK-NEXT: %[[IF:.+]] = scf.if %[[GUARD]] -> {{.+}} {
  // CHECK-NEXT:   %[[WHILE:.+]] = scf.while (%[[B_ARG:.+]] = %[[CST]]) : {{.+}} {
  // CHECK-NEXT:     %[[AFTER:.+]] = "test.after"(%[[B_ARG]])
  // CHECK-NEXT:     %[[BEFORE:.+]] = "test.before"(%[[AFTER]])
  // CHECK-NEXT:     scf.condition(%[[BEFORE]]) %[[AFTER]]
  // CHECK-NEXT:   } do {
  // CHECK-NEXT:   ^bb0(%[[A_ARG:.+]]: {{.+}}):
  // CHECK-NEXT:     scf.yield %[[A_ARG]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   scf.yield %[[WHILE]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %[[CST]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[IF]]
  %cst = arith.constant 0.0 : f32
  %while = scf.while (%b_arg = %cst) : (f32) -> (f32) {
    %before = "test.before"(%b_arg) : (f32) -> i1
    scf.condition(%before) %b_arg : f32
  } do {
  ^bb0(%a_arg: f32):
    %after = "test.after"(%a_arg) : (f32) -> f32
    scf.yield %after : f32
  }
  return %while : f32
}

// -----

!arg0 = f16
!arg1 = f32
!iter = f64
!res0 = i16
!res1 = i32
func.func private @before(!arg0, !iter) -> (i1, !res0, !res1)
func.func private @after(!arg1, !res0, !res1) -> !iter
// CHECK-LABEL: @complex
func.func @complex(%arg0: !arg0, %arg1: !arg1) -> !res0 {
  // CHECK-SAME: (%[[ARG0:.+]]: {{.+}}, %[[ARG1:.+]]: {{.+}})
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 0.0
  // CHECK-NEXT: %[[GUARD:.+]]:3 = call @before(%[[ARG0]], %[[CST]])
  // CHECK-NEXT: %[[IF:.+]] = scf.if %[[GUARD]]#0 -> {{.+}} {
  // CHECK-NEXT:   %[[WHILE:.+]]:2 = scf.while (%[[INIT0:.+]] = %[[GUARD]]#1, %[[INIT1:.+]] = %[[GUARD]]#2) : {{.+}} {
  // CHECK-NEXT:     %[[AFTER:.+]] = func.call @after(%[[ARG1]], %[[INIT0]], %[[INIT1]])
  // CHECK-NEXT:     %[[BEFORE:.+]]:3 = func.call @before(%[[ARG0]], %[[AFTER]])
  // CHECK-NEXT:     scf.condition(%[[BEFORE]]#0) %[[BEFORE]]#1, %[[BEFORE]]#2
  // CHECK-NEXT:   } do {
  // CHECK-NEXT:   ^bb0(%[[FWD0:.+]]: {{.+}}, %[[FWD1:.+]]: {{.+}}):
  // CHECK-NEXT:     scf.yield %[[FWD0]], %[[FWD1]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   scf.yield %[[WHILE]]#0
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %[[GUARD]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[IF]]
  %cst = arith.constant 0.0 : !iter
  %res:2 = scf.while (%iter = %cst) : (!iter) -> (!res0, !res1) {
    %before:3 = func.call @before(%arg0, %iter) : (!arg0, !iter) -> (i1, !res0, !res1)
    scf.condition(%before#0) %before#1, %before#2 : !res0, !res1
  } do {
  ^bb0(%cond_arg0: !res0, %cond_arg1: !res1):
    %after = func.call @after(%arg1, %cond_arg0, %cond_arg1) : (!arg1, !res0, !res1) -> !iter
    scf.yield %after : !iter
  }
  return %res#0 : !res0
}
