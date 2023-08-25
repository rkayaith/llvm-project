#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ROTATEWHILELOOPPASS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

// Rotate the loop, merging the 'before' and 'after' blocks together, with an
// 'if' guarding the initial iteration. For example, starting with:
//   %result = while(%iter_arg = %init) {
//     %cond, %result = before(%iter_arg)
//     condition(%cond) %result
//   } do (%result) {
//     %iter_arg = after(%result)
//     yield %iter_arg
//   }
//
// We :
//   %guard_cond, %guard_result = before(%init)
//   %result = if (%guard_cond) {
//     %result = while(%result = %guard_result) {
//       %iter_arg = after(%result)
//       %cond, %result = before(%iter_arg)
//       condition(%cond) %result
//     } do (%result) {
//       yield %result
//     }
//     yield %result
//   } else {
//     yield %guard_result
//   }
static LogicalResult rotateWhileOp(WhileOp whileOp, PatternRewriter &r) {
  if (llvm::hasSingleElement(*whileOp.getAfterBody()))
    return r.notifyMatchFailure(whileOp, "loop is already rotated");

  // Clone the 'before' block outside the loop, to evaulate the guard
  // condition.
  IRMapping guardMap;
  guardMap.map(whileOp.getBeforeArguments(), whileOp.getOperands());
  for (Operation &op : whileOp.getBeforeBody()->without_terminator())
    r.clone(op, guardMap);
  Value guardCond = guardMap.lookup(whileOp.getConditionOp().getCondition());
  SmallVector<Value> guardResults = llvm::to_vector(
      llvm::map_range(whileOp.getConditionOp().getArgs(),
                      [&](Value v) { return guardMap.lookupOrDefault(v); }));

  auto createRotatedWhileOp =
      [&whileOp, &guardResults](OpBuilder &builder, Location loc) -> WhileOp {
    IRRewriter r(builder);
    auto rotated =
        r.create<WhileOp>(loc, whileOp.getResultTypes(), guardResults);

    {
      IRRewriter::InsertionGuard guard(r);

      // The 'before' block contains the original 'after' + 'before' blocks.
      YieldOp yield = whileOp.getYieldOp();
      r.inlineRegionBefore(whileOp.getAfter(), rotated.getBefore(),
                           rotated.getBefore().end());
      r.mergeBlocks(whileOp.getBeforeBody(), rotated.getBeforeBody(),
                    yield.getResults());
      r.eraseOp(yield);

      // The 'after' block just forwards all arguments.
      SmallVector<Location> argLocs = llvm::to_vector(
          llvm::map_range(rotated.getBeforeArguments(),
                          [](BlockArgument a) { return a.getLoc(); }));
      Block *afterBody =
          r.createBlock(&rotated.getAfter(), rotated.getAfter().begin(),
                        rotated.getBeforeBody()->getArgumentTypes(), argLocs);
      r.create<YieldOp>(loc, afterBody->getArguments());
    }
    return rotated;
  };

  r.replaceOpWithNewOp<IfOp>(
      whileOp, guardCond, /*thenBuilder=*/
      [&](OpBuilder &b, Location loc) {
        WhileOp rotated = createRotatedWhileOp(b, loc);
        b.create<YieldOp>(loc, rotated.getResults());
      },
      /*elseBuilder=*/
      [&](OpBuilder &b, Location loc) {
        b.create<YieldOp>(loc, guardResults);
      });
  return success();
}

namespace {
struct RotateWhileLoopPass
    : public mlir::impl::RotateWhileLoopPassBase<RotateWhileLoopPass> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add(rotateWhileOp);
    // IfOp::getCanonicalizationPatterns(patterns, ctx);
    // WhileOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));
  }
};
} // namespace
