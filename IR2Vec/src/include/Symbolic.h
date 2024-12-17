// Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
// Department of Computer Science and Engineering, IIT Hyderabad
//
// This software is available under the BSD 4-Clause License. Please see LICENSE
// file in the top-level directory for more details.
//
#ifndef __IR2Vec_Symbolic_H__
#define __IR2Vec_Symbolic_H__

#include "utils.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h> // For demangling C++ names

#include <fstream>
#include <map>
#include <array>

class IR2Vec_Symbolic {

private:
  llvm::Module &M;
  IR2Vec::Vector pgmVector;

  IR2Vec::Vector getValue(std::string key);
  void getValue_opchist(std::string key, std::map<std::string, int> &opcHist);

  // IR2Vec::Vector bb2Vec(llvm::BasicBlock &B,
  //                       llvm::SmallVector<llvm::Function *, 15> &funcStack);
  std::array<IR2Vec::Vector, 4> bb2Vec(llvm::BasicBlock &B,
                        llvm::SmallVector<llvm::Function *, 15> &funcStack);

  void bb2Vec_opchist(llvm::BasicBlock &B,
                        llvm::SmallVector<llvm::Function *, 15> &funcStack, std::map<std::string, int> &opcHist);


  // IR2Vec::Vector func2Vec(llvm::Function &F,
  //                         llvm::SmallVector<llvm::Function *, 15> &funcStack);
  std::array<IR2Vec::Vector, 4> func2Vec(llvm::Function &F,
                          llvm::SmallVector<llvm::Function *, 15> &funcStack);
  
  std::array<IR2Vec::Vector, 1> func2Vec_opchist(llvm::Function &F,
                          llvm::SmallVector<llvm::Function *, 15> &funcStack, std::map<std::string, int> &opcHist);
  std::string res;
  llvm::SmallMapVector<const llvm::Function *, IR2Vec::Vector, 16> funcVecMap;

  llvm::SmallMapVector<const llvm::Function *, std::array<IR2Vec::Vector, 4>, 16> funcVecMap_OTA;
  llvm::SmallMapVector<const llvm::Function *, std::array<IR2Vec::Vector, 1>, 16> funcVecMap_opchist;

  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
      instVecMap;

  //For separately dumping vectors for opcode, type and argument
  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
      instVecMap_O;
  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
      instVecMap_T;
  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
      instVecMap_A;
  
  std::map<std::string, IR2Vec::Vector> opcMap;

  std::map<std::string, int> opcHist; //for keeping track of opcode histogram

  std::vector<std::vector<llvm::BasicBlock*>> randomWalk(llvm::Function &F, std::vector<llvm::BasicBlock*> &block_addresses, int k, int n);

  // bool isExternalLibraryCall(llvm::Function *Callee);

public:
  IR2Vec_Symbolic(llvm::Module &M) : M{M} {
    pgmVector = IR2Vec::Vector(DIM, 0);
    res = "";
    IR2Vec::collectDataFromVocab(opcMap);
  }

  void generateSymbolicEncodings(std::ostream *o = nullptr);
  void generateSymbolicEncodingsForFunction(std::ostream *o = nullptr,
                                            std::string name = "");
  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
  getInstVecMap() {
    return instVecMap;
  }

  llvm::SmallMapVector<const llvm::Function *, IR2Vec::Vector, 16>
  getFuncVecMap() {
    return funcVecMap;
  }

  IR2Vec::Vector getProgramVector() { return pgmVector; }
};

#endif
