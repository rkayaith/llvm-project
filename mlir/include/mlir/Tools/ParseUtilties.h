//===- ParseUtilities.h - MLIR Tool Parse Utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file containts common utilities for implementing the file-parsing
// behaviour for MLIR tools.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PARSEUTILITIES_H
#define MLIR_TOOLS_PARSEUTILITIES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

namespace mlir {
/// TODO (rk)
inline OwningOpRef<ModuleOp>
parseSourceFileWithImplicitModule(llvm::SourceMgr &sourceMgr,
                                  const ParserConfig &config) {
  OwningOpRef<ModuleOp> mod =
      ModuleOp::create(UnknownLoc::get(config.getContext()));
  Block &block = *mod->getBody();

  LocationAttr loc;
  if (failed(parseSourceFile(sourceMgr, &block, config, &loc)))
    return nullptr;
  mod->getOperation()->setLoc(loc);

  if (llvm::hasSingleElement(block)) {
    if (auto op = dyn_cast<ModuleOp>(block.front())) {
      op->remove();
      mod = op;
    }
  }

  return mod;
}

/// This parses the file specified by the indicated SourceMgr. If parsing was
/// not successful, null is returned and an error message is emitted through the
/// error handler registered in the context.
/// If 'insertImplicitModule' is true a top-level 'builtin.module' op will be
/// inserted that contains the parsed IR, unless one exists already.
inline OwningOpRef<Operation *>
parseSourceFileForTool(llvm::SourceMgr &sourceMgr, const ParserConfig &config,
                       bool insertImplicitModule) {
  if (insertImplicitModule)
    return parseSourceFileWithImplicitModule(sourceMgr, config);
  return parseSourceFile(sourceMgr, config);
}
} // namespace mlir

#endif // MLIR_TOOLS_PARSEUTILITIES_H
