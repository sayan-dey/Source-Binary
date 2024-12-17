// Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
// Department of Computer Science and Engineering, IIT Hyderabad
//
// This software is available under the BSD 4-Clause License. Please see LICENSE
// file in the top-level directory for more details.
//


//Modified version of Symbolic.cpp to dump opcode histogram as function embedding
//Dumping strings and library embeddings as usual

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


void IR2Vec_Symbolic::getValue_opchist(std::string key, std::map<std::string, int> &opcHist) {
  Vector vec;
  if (opcMap.find(key) == opcMap.end())
    IR2VEC_DEBUG(errs() << "cannot find key in map : " << key << "\n");
  else
    opcHist[key]+=1;
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
      auto tmp = func2Vec_opchist(f, funcStack, opcHist);
      // funcVecMap[&f] = tmp;
      funcVecMap_opchist[&f] = tmp;

      if (level == 'f') {
        res += updatedRes_opchist(tmp, &f, &M);

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

      /*
      // else if (level == 'p') {
      std::transform(pgmVector.begin(), pgmVector.end(), tmp.begin(),
                     pgmVector.begin(), std::plus<double>());
      //adding entire function vector, like earlier

      // }
      */
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
      auto tmp = func2Vec_opchist(f, funcStack, opcHist);
      // funcVecMap[&f] = tmp;
      funcVecMap_opchist[&f] = tmp;
      if (level == 'f') {
        res += updatedRes_opchist(tmp, &f, &M);
        res += "\n";
        noOfFunc++;
      }
    }
  }

  
  if (o)
    *o << res;
}



std::array<Vector, 1> IR2Vec_Symbolic::func2Vec_opchist(Function &F,
                                 SmallVector<Function *, 15> &funcStack, std::map<std::string, int> &opcHist) {

  //std::map<std::string, int> opcHist; 
  //already declared in header...fill with all 0s for seed emb or for keys of std::map<std::string, IR2Vec::Vector> opcMap;

  for(auto it: opcMap)
  {
    string opcode = it.first;
    opcHist[opcode] = 0;
  }

  // auto It = funcVecMap.find(&F);
  // if (It != funcVecMap.end()) {
  //   return It->second;
  // }

  auto It = funcVecMap_opchist.find(&F);
  if (It != funcVecMap_opchist.end()) {
    return It->second;
  }

  funcStack.push_back(&F);

  Vector funcVector_O(DIM, 0);
  // ReversePostOrderTraversal<Function *> RPOT(&F);
  MapVector<const BasicBlock *, double> cumulativeScore;

  // llvm::errs()<<F.getName().str()<<"\n";

  
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
      bb2Vec_opchist(*b, funcStack, opcHist);
    }
    // llvm::errs()<<"\n";
  }

  int ind=0;
  for(auto it:opcHist)
  {
      string opcode = it.first;
      funcVector_O[ind]+=it.second;
      ind++;
  }

  funcStack.pop_back();

  return {funcVector_O};
  // return funcVector;
}

void IR2Vec_Symbolic::bb2Vec_opchist(BasicBlock &B,
                               SmallVector<Function *, 15> &funcStack, std::map<std::string, int> &opcHist) {


  for (auto &I : B) {

    //For separately dumping vectors for opcode only
    getValue_opchist(I.getOpcodeName(), opcHist);
  }

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
