// Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
// Department of Computer Science and Engineering, IIT Hyderabad
//
// This software is available under the BSD 4-Clause License. Please see LICENSE
// file in the top-level directory for more details.
//
#include "Symbolic.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Demangle/Demangle.h" //for getting function base name
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"

#include <algorithm> // for transform
#include <ctype.h>
#include <cxxabi.h>
#include <functional> // for plus
#include <iomanip>
#include <queue>
#include <vector>
#include <regex> // For regular expressions
#include <algorithm> // For std::replace
#include <string> // For std::string, std::tolower, etc
#include <array>
#include <boost/algorithm/string/join.hpp>

#include <cxxabi.h> // For demangling C++ names

using namespace llvm;
using namespace IR2Vec;
using namespace std;
using abi::__cxa_demangle;


//added (Sayan) for checking if a calle func is ext lib call or not
bool isExternalLibraryCall(Function *Callee) {
    // Ignore if the function is not declared externally
    if (!Callee->isDeclaration()) return false;

    // Demangle C++ names to make checking easier
    const std::string &Name = Callee->getName().str();
    int Status;
    char *DemangledName = abi::__cxa_demangle(Name.c_str(), nullptr, nullptr, &Status);
    std::string FinalName = (Status == 0) ? std::string(DemangledName) : Name;
    free(DemangledName);

    // Filter out known C++ standard library internal calls
    if (FinalName.find("std::") != std::string::npos ||
        FinalName.find("__cxa") != std::string::npos ||
        FinalName.find("_ZNSa") != std::string::npos ||
        FinalName.find("_ZNSt") != std::string::npos) {
        return false;
    }

    return true;
}


std::string removeEntity(const std::string &text, const std::vector<std::string> &entity_list) {
  std::string result = text;
  for (const std::string &entity : entity_list) {
      std::regex entity_regex(std::regex_replace(entity, std::regex(R"(\%)"), R"(\%)"));
      result = std::regex_replace(result, entity_regex, " ");
  }
  return result;
}

std::string filterStringLiteral(const std::string &strRef) {
  // If "http" is present, skip processing
  if (strRef.find("http") != std::string::npos) {
    return "";
  }

  // Define format specifiers and punctuation
  std::vector<std::string> format_specifiers = {
      "%c", "%s", "%hi", "%h", "%Lf", "%n", "%d", "%i", "%o", "%x", "%p", "%f", "%u", "%e",
      "%E", "%%", "%#lx", "%lu", "%ld", "__", "_"
  };
  
  std::vector<char> punctuations = {
      '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
      ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'
  };

  // Remove format specifiers
  std::string cleaned_str = removeEntity(strRef, format_specifiers);

  // Remove punctuation
  for (char punctuation : punctuations) {
    cleaned_str.erase(std::remove(cleaned_str.begin(), cleaned_str.end(), punctuation), cleaned_str.end());
  }

  // Remove non-alphabetic characters and convert to lowercase
  std::string final_str;
  std::transform(cleaned_str.begin(), cleaned_str.end(), cleaned_str.begin(), ::tolower);
  for (char ch : cleaned_str) {
    if (isalpha(ch)) {
        final_str += ch;
    } else {
        final_str += ' ';
    }
  }

  // Trim and split words
  std::istringstream iss(final_str);
  std::string word;
  std::string filtered_str;
  while (iss >> word) {
      if (!word.empty()) {
          filtered_str += word + " ";
      }
  }
  if (!filtered_str.empty()) {
      filtered_str.pop_back(); // Remove trailing space
  }

  return filtered_str;
}


Vector IR2Vec_Symbolic::getValue(std::string key) {
  Vector vec;
  if (opcMap.find(key) == opcMap.end())
    IR2VEC_DEBUG(errs() << "cannot find key in map : " << key << "\n");
  else
    vec = opcMap[key];
  return vec;
}

void IR2Vec_Symbolic::generateSymbolicEncodings(std::ostream *o) {

  //For getting string literals and library functions....

  vector<string> strRefs, libCalls;

  // Collect string literals from global variables
  for (GlobalVariable &GV : M.globals()) {
      if (GV.hasInitializer()) {
          if (ConstantDataArray *CDA = dyn_cast<ConstantDataArray>(GV.getInitializer())) {
              /*
              if (CDA->isString()) {
                strRefs.push_back(CDA->getAsCString());
                // errs() << "String literal (global): " << CDA->getAsCString() << "\n";
              }
              */

              if (CDA->isString()) {
                std::string strLiteral = CDA->getAsCString().str();
                std::string filteredLiteral = filterStringLiteral(strLiteral);
                if (!filteredLiteral.empty()) {
                    strRefs.push_back(filteredLiteral);
                    // errs() << "Filtered String Literal (global): \"" << filteredLiteral << "\"\n";
                }
              }
          }
      }
  }

  for (Function &F : M) {
    if (F.isDeclaration()) continue; // Skip function declarations


    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (CallBase *CB = dyn_cast<CallBase>(&I)) {
                if (Function *Callee = CB->getCalledFunction()) {
                    // if (Callee->isDeclaration()) //earlier condition
                    if (isExternalLibraryCall(Callee)){
                        // This is a call to a library function
                        libCalls.push_back(Callee->getName().str());
                        // errs() << "Library call: " << Callee->getName() << "\n";
                    }
                }
            }

            // Check for string literals in operands
            for (Use &Op : I.operands()) {
                if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op)) {
                    if (GV->hasInitializer()) {
                        if (ConstantDataArray *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer())) {
                            if (CDA->isString()) {
                                std::string strLiteral = CDA->getAsCString().str();
                                std::string filteredLiteral = filterStringLiteral(strLiteral);
                                if (!filteredLiteral.empty()) {
                                    strRefs.push_back(filteredLiteral);
                                    // errs() << "Filtered String Literal (operand): \"" << filteredLiteral << "\"\n";
                                }
                                // strRefs.push_back(CDA->getAsCString());
                                // errs() << "String literal (operand): " << CDA->getAsCString() << "\n";
                            }
                        }
                    }
                }
            }
        }
    }
  }

  //Logic for putting strRefs and libCalls
  string delim = "^";
  string joinedStrRefs = "", joinedLibCalls = "";
	joinedStrRefs = boost::algorithm::join(strRefs, delim);
  joinedLibCalls = boost::algorithm::join(libCalls, delim);

  int noOfFunc = 0;
  for (auto &f : M) {
    if (!f.isDeclaration()) {
      SmallVector<Function *, 15> funcStack;
      auto tmp = func2Vec(f, funcStack);
      // funcVecMap[&f] = tmp;
      funcVecMap_OTA[&f] = tmp;

      if (level == 'f') {
        res += updatedRes_OTA(tmp, &f, &M);

        if(joinedStrRefs.size()>0)
        {
          res += "StrRefs:";
          res += joinedStrRefs;
        }

        if(joinedLibCalls.size()>0)
        {
          res += "LibCalls:";
          res += joinedLibCalls;
        }

        res += "\n";
        noOfFunc++;
      }

      // else if (level == 'p') {
      std::transform(pgmVector.begin(), pgmVector.end(), tmp[3].begin(),
                     pgmVector.begin(), std::plus<double>());
      //adding entire function vector, like earlier

      // }
    }
  }

  

  IR2VEC_DEBUG(errs() << "Number of functions written = " << noOfFunc << "\n");

  if (level == 'p') {
    if (cls != -1)
      res += std::to_string(cls) + "\t";

    for (auto i : pgmVector) {
      if ((i <= 0.0001 && i > 0) || (i < 0 && i >= -0.0001)) {
        i = 0;
      }
      res += std::to_string(i) + "\t";
    }
    res += "\n";
  }

  if (o)
  {
    *o << res;
  }

  IR2VEC_DEBUG(errs() << "class = " << cls << "\n");
  IR2VEC_DEBUG(errs() << "res = " << res);
}


// for generating symbolic encodings for specific function
void IR2Vec_Symbolic::generateSymbolicEncodingsForFunction(std::ostream *o,
                                                           std::string name) {
  int noOfFunc = 0;
  for (auto &f : M) {
    auto Result = getActualName(&f);
    if (!f.isDeclaration() && Result == name) {
      // Vector tmp;
      SmallVector<Function *, 15> funcStack;
      auto tmp = func2Vec(f, funcStack);
      // funcVecMap[&f] = tmp;
      funcVecMap_OTA[&f] = tmp;
      if (level == 'f') {
        // res += updatedRes(tmp, &f, &M);
        res += updatedRes_OTA(tmp, &f, &M);
        res += "\n";
        noOfFunc++;
      }
    }
  }

  

  if (o)
    *o << res;
}



std::array<Vector, 4> IR2Vec_Symbolic::func2Vec(Function &F,
                                 SmallVector<Function *, 15> &funcStack) {
  // auto It = funcVecMap.find(&F);
  // if (It != funcVecMap.end()) {
  //   return It->second;
  // }

  auto It = funcVecMap_OTA.find(&F);
  if (It != funcVecMap_OTA.end()) {
    return It->second;
  }

  funcStack.push_back(&F);

  Vector funcVector_O(DIM, 0);
  Vector funcVector_T(DIM, 0);
  Vector funcVector_A(DIM, 0);

  Vector funcVector(DIM, 0);
  // ReversePostOrderTraversal<Function *> RPOT(&F);
  MapVector<const BasicBlock *, double> cumulativeScore;

  // llvm::errs()<<F.getName().str()<<"\n";

  /*
  for (auto *b : RPOT) {
    auto bbVector = bb2Vec(*b, funcStack);

    Vector weightedBBVector;
    weightedBBVector = bbVector;

    std::transform(funcVector.begin(), funcVector.end(),
                   weightedBBVector.begin(), funcVector.begin(),
                   std::plus<double>());
  }
  */

  //randomWalk calling code here

  std::vector<BasicBlock*> block_addresses;
  for (BasicBlock &BB : F) {
      block_addresses.push_back(&BB);
  }

  int k=20, n=2;
  std::vector<std::vector<BasicBlock*>> bbWalks = randomWalk(F, block_addresses, k, n);

  for(auto bbWalk: bbWalks)
  {
    for(auto b: bbWalk)
    {
      // auto bbVector = bb2Vec(*b, funcStack);
      auto bbVectors = bb2Vec(*b, funcStack);

      auto bbVector_O = bbVectors[0];
      auto bbVector_T = bbVectors[1];
      auto bbVector_A = bbVectors[2];
      auto bbVector = bbVectors[3]; 

      Vector weightedBBVector_O;
      weightedBBVector_O = bbVector_O;

      Vector weightedBBVector_T;
      weightedBBVector_T = bbVector_T;

      Vector weightedBBVector_A;
      weightedBBVector_A = bbVector_A;

      Vector weightedBBVector;
      weightedBBVector = bbVector;

      // for(int i=0;i<DIM;i++)
      // llvm:errs()<<bbVector[i]<<" ";

      // llvm::errs()<<"\n";

      std::transform(funcVector_O.begin(), funcVector_O.end(),
                   weightedBBVector_O.begin(), funcVector_O.begin(),
                   std::plus<double>());
      std::transform(funcVector_T.begin(), funcVector_T.end(),
                   weightedBBVector_T.begin(), funcVector_T.begin(),
                   std::plus<double>());
      std::transform(funcVector_A.begin(), funcVector_A.end(),
                   weightedBBVector_A.begin(), funcVector_A.begin(),
                   std::plus<double>());

      std::transform(funcVector.begin(), funcVector.end(),
                   weightedBBVector.begin(), funcVector.begin(),
                   std::plus<double>());

    }
    // llvm::errs()<<"\n";
  }

  funcStack.pop_back();

  return {funcVector_O, funcVector_T, funcVector_A, funcVector};
  // return funcVector;
}

std::array<Vector, 4> IR2Vec_Symbolic::bb2Vec(BasicBlock &B,
                               SmallVector<Function *, 15> &funcStack) {


  //For separately dumping vectors for opcode, type and argument
  Vector bbVector_O(DIM,0);
  Vector bbVector_T(DIM,0);
  Vector bbVector_A(DIM,0);

  Vector bbVector(DIM, 0);

  for (auto &I : B) {

    //For separately dumping vectors for opcode, type and argument
    Vector instVector_O(DIM,0);
    Vector instVector_T(DIM,0);
    Vector instVector_A(DIM,0);


    Vector instVector(DIM, 0);
    auto vec = getValue(I.getOpcodeName());
    // if (isa<CallInst>(I)) {
    //   auto ci = dyn_cast<CallInst>(&I);
    //   // ci->dump();
    //   Function *func = ci->getCalledFunction();
    //   if (func) {
    //     // if(!func->isDeclaration())
    //     //     if(func != I.getParent()->getParent())
    //     //         errs() << func->getName() << "\t" <<
    //     //         I.getParent()->getParent()->getName() << "\n";
    //     if (!func->isDeclaration() &&
    //         std::find(funcStack.begin(), funcStack.end(), func) ==
    //             funcStack.end()) {
    //       auto funcVec = func2Vec(*func, funcStack);

    //       std::transform(vec.begin(), vec.end(), funcVec.begin(),
    //       vec.begin(),
    //                      std::plus<double>());
    //     }
    //   } else {
    //     IR2VEC_DEBUG(I.dump());
    //     IR2VEC_DEBUG(errs() << "==========================Function definition
    //     "
    //                          "not found==================\n");
    //   }
    // }
    scaleVector(vec, WO);

    std::transform(instVector_O.begin(), instVector_O.end(), vec.begin(),
                   instVector_O.begin(), std::plus<double>());
    instVecMap_O[&I] = instVector_O;

    std::transform(instVector.begin(), instVector.end(), vec.begin(),
                   instVector.begin(), std::plus<double>());


    auto type = I.getType();

    if (type->isVoidTy()) {
      vec = getValue("voidTy");
    } else if (type->isFloatingPointTy()) {
      vec = getValue("floatTy");
    } else if (type->isIntegerTy()) {
      vec = getValue("integerTy");
    } else if (type->isFunctionTy()) {
      vec = getValue("functionTy");
    } else if (type->isStructTy()) {
      vec = getValue("structTy");
    } else if (type->isArrayTy()) {
      vec = getValue("arrayTy");
    } else if (type->isPointerTy()) {
      vec = getValue("pointerTy");
    } else if (type->isVectorTy()) {
      vec = getValue("vectorTy");
    } else if (type->isEmptyTy()) {
      vec = getValue("emptyTy");
    } else if (type->isLabelTy()) {
      vec = getValue("labelTy");
    } else if (type->isTokenTy()) {
      vec = getValue("tokenTy");
    } else if (type->isMetadataTy()) {
      vec = getValue("metadataTy");
    } else {
      vec = getValue("unknownTy");
    }

    /*switch (I.getType()->getTypeID()) {
    case 0:
      vec = getValue("voidTy");
      break;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      vec = getValue("floatTy");
      break;
    case 11:
      vec = getValue("integerTy");
      break;
    case 12:
      vec = getValue("functionTy");
      break;
    case 13:
      vec = getValue("structTy");
      break;
    case 14:
      vec = getValue("arrayTy");
      break;
    case 15:
      vec = getValue("pointerTy");
      break;
    case 16:
      vec = getValue("vectorTy");
      break;
    default:
      vec = getValue("unknownTy");
    }*/

    scaleVector(vec, WT);

    std::transform(instVector_T.begin(), instVector_T.end(), vec.begin(),
                   instVector_T.begin(), std::plus<double>());
    instVecMap_T[&I] = instVector_T;

    std::transform(instVector.begin(), instVector.end(), vec.begin(),
                   instVector.begin(), std::plus<double>());

    for (unsigned i = 0; i < I.getNumOperands(); i++) {
      Vector vec;
      if (isa<Function>(I.getOperand(i))) {
        vec = getValue("function");
      } else if (isa<PointerType>(I.getOperand(i)->getType())) {
        vec = getValue("pointer");
      } else if (isa<Constant>(I.getOperand(i))) {
        vec = getValue("constant");
      } else {
        vec = getValue("variable");
      }
      scaleVector(vec, WA);

      std::transform(instVector_A.begin(), instVector_A.end(), vec.begin(),
                     instVector_A.begin(), std::plus<double>());
      instVecMap_A[&I] = instVector_A;

      std::transform(instVector.begin(), instVector.end(), vec.begin(),
                     instVector.begin(), std::plus<double>());

      instVecMap[&I] = instVector;
    }

    std::transform(bbVector_O.begin(), bbVector_O.end(), instVector_O.begin(),
                   bbVector_O.begin(), std::plus<double>());
    std::transform(bbVector_T.begin(), bbVector_T.end(), instVector_T.begin(),
                   bbVector_T.begin(), std::plus<double>());
    std::transform(bbVector_A.begin(), bbVector_A.end(), instVector_A.begin(),
                   bbVector_A.begin(), std::plus<double>());
                   
    std::transform(bbVector.begin(), bbVector.end(), instVector.begin(),
                   bbVector.begin(), std::plus<double>());
  }

  return {bbVector_O, bbVector_T, bbVector_A, bbVector};
  // return bbVector;

}

//Function for random walk
std::vector<std::vector<BasicBlock*>> IR2Vec_Symbolic::randomWalk(Function &F, std::vector<BasicBlock*> &block_addresses, int k, int n) {
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
