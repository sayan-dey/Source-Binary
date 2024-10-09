// Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
// Department of Computer Science and Engineering, IIT Hyderabad
//
// This software is available under the BSD 4-Clause License. Please see LICENSE
// file in the top-level directory for more details.
//
#include "CollectIR.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include <fstream>

//header files included by me
#include <vector>
#include <unordered_map>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <random>

using namespace llvm;
using namespace std;

void CollectIR::generateTriplets(std::ostream &out) {

  std::string FileName = M.getSourceFileName();
  // llvm::outs()<<"FileName: "<<FileName<<"\n";

  for (Function &F : M)
  {
    if(F.isDeclaration()) //skipping functions which are not user-defined
    continue;

    std::string funcName = F.getName().str();
    // llvm::outs()<<"Function name: "<<F.getName()<<"\n";

    std::string res="";

    //write randomWalk calling code here

    // std::vector<BasicBlock*> block_addresses;
    // for (BasicBlock &BB : F) {
    //     block_addresses.push_back(&BB);
    // }

    // int k=5, n=2;
    // std::vector<std::vector<BasicBlock*>> bbWalks = randomWalk(F, block_addresses, k, n);


    for (BasicBlock &B : F) //need to traverse blocks of randomwalks here
      traverseBasicBlock(B,res);

    if(res.size()>0)
    res.pop_back();
    out<<FileName<<":"<<funcName<<"\t"<<res<<"\n";
  }
  
  // out << "\t"<<res;
}

void CollectIR::traverseBasicBlock(BasicBlock &B, std::string &res) {
  for (Instruction &I : B) {

    if(res.size()==0)
    res += std::string(I.getOpcodeName()) + "|";
    else
    {
      // res += "\n" + std::string(I.getOpcodeName()) + " ";
      res.pop_back();
      res += "<INST>" + std::string(I.getOpcodeName()) + "|";
    }
  
    auto type = I.getType();
    IR2VEC_DEBUG(I.print(outs()); outs() << "\n";);
    IR2VEC_DEBUG(I.getType()->print(outs()); outs() << " Type\n";);

    std::string stype;

    if (type->isVoidTy()) {
      stype = "voidTy|";
      res += stype;
    } else if (type->isFloatingPointTy()) {
      stype = "floatTy|";
      res += stype;
    } else if (type->isIntegerTy()) {
      stype = "integerTy|";
      res += stype;
    } else if (type->isFunctionTy()) {
      stype = "functionTy|";
      res += stype;
    } else if (type->isStructTy()) {
      stype = "structTy|";
      res += stype;
    } else if (type->isArrayTy()) {
      stype = "arrayTy|";
      res += stype;
    } else if (type->isPointerTy()) {
      stype = "pointerTy|";
      res += stype;
    } else if (type->isVectorTy()) {
      stype = "vectorTy|";
      res += stype;
    } else if (type->isEmptyTy()) {
      stype = "emptyTy|";
      res += stype;
    } else if (type->isLabelTy()) {
      stype = "labelTy|";
      res += stype;
    } else if (type->isTokenTy()) {
      stype = "tokenTy|";
      res += stype;
    } else if (type->isMetadataTy()) {
      stype = "metadataTy|";
      res += stype;
    } else {
      stype = "unknownTy|";
      res += stype;
    }

    IR2VEC_DEBUG(errs() << "Type taken : " << stype << "\n";);

    for (unsigned i = 0; i < I.getNumOperands(); i++) {
      IR2VEC_DEBUG(I.print(outs()); outs() << "\n";);
      IR2VEC_DEBUG(outs() << i << "\n");
      IR2VEC_DEBUG(I.getOperand(i)->print(outs()); outs() << "\n";);

      if (isa<Function>(I.getOperand(i))) {
        res += "function|";
        IR2VEC_DEBUG(outs() << "Function\n");
      } else if (isa<PointerType>(I.getOperand(i)->getType())) {
        res += "pointer|";
        IR2VEC_DEBUG(outs() << "pointer\n");
      } else if (isa<Constant>(I.getOperand(i))) {
        res += "constant|";
        IR2VEC_DEBUG(outs() << "constant\n");
      } else if (isa<BasicBlock>(I.getOperand(i))) {
        res += "label|";
        IR2VEC_DEBUG(outs() << "label\n");
      } else {
        res += "variable|";
        IR2VEC_DEBUG(outs() << "variable2\n");
      }
    }
  }
}


std::vector<std::vector<BasicBlock*>> CollectIR::randomWalk(Function &F, std::vector<BasicBlock*> &block_addresses, int k, int n) {
    std::unordered_map<BasicBlock*, int> func_block_freq;
    std::vector<std::vector<BasicBlock*>> walk_visited_blocks_list;

    std::vector<BasicBlock*> blocks;

    // Putting all block of size>0 in a list
    for (BasicBlock &BB : F) {
        if (BB.size() > 0 && std::find(block_addresses.begin(), block_addresses.end(), &BB) != block_addresses.end()) {
            blocks.push_back(&BB);
        }
    }

    while (true) {
        // Choosing possible starting blocks
        std::vector<BasicBlock*> starting_blocks;
        for (BasicBlock *BB : blocks) {
            if (func_block_freq.find(BB) == func_block_freq.end() || func_block_freq[BB] < n) {
                starting_blocks.push_back(BB);
            }
        }

        if (starting_blocks.empty()) {
            break;
        }

        BasicBlock *current_block = starting_blocks[rand() % starting_blocks.size()];

        std::vector<BasicBlock*> walk_visited_blocks;
        int num_vis = 0;

        while (num_vis < k && std::find(walk_visited_blocks.begin(), walk_visited_blocks.end(), current_block) == walk_visited_blocks.end()) {
            if (func_block_freq.find(current_block) != func_block_freq.end()) {
                func_block_freq[current_block]++;
            } else {
                func_block_freq[current_block] = 1;
            }

            walk_visited_blocks.push_back(current_block);

            if (current_block->size() > 0) {
                num_vis++;
            }

            if (succ_size(current_block) == 1) {
                current_block = *succ_begin(current_block);
                if (std::find(block_addresses.begin(), block_addresses.end(), current_block) == block_addresses.end()) {
                    break;
                }
            } else if (succ_size(current_block) > 1) {
                SmallVector<BasicBlock*, 4> successors(succ_begin(current_block), succ_end(current_block));
                current_block = successors[rand() % successors.size()];
                if (std::find(block_addresses.begin(), block_addresses.end(), current_block) == block_addresses.end()) {
                    break;
                }
            } else {
                break;
            }
        }

        walk_visited_blocks_list.push_back(walk_visited_blocks);
    }

    return walk_visited_blocks_list;
}