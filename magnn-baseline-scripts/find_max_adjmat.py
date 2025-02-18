# Script for finding the max number of basic blocks of any adj matrix

import os, sys
import angr
import gensim
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import random
import json

from utils import SEED

random.seed(SEED)
np.random.seed(SEED)

# Extract adjacency matrix for the main function
def extract_basic_block_adjacency(binary_path):
    project = angr.Project(binary_path, load_options={"auto_load_libs": False})
    cfg = project.analyses.CFGFast()
    
    main_func = cfg.kb.functions.function(name="main")
    if main_func is None:
        raise ValueError("Main function not found!")
    
    basic_blocks = list(main_func.blocks)
    block_to_id = {block.addr: idx for idx, block in enumerate(basic_blocks)}
    num_blocks = len(basic_blocks)
    adjacency_matrix = np.zeros((num_blocks, num_blocks), dtype=np.int32)
    
    for block in basic_blocks:
        # Skip blocks not in the graph
        if block not in cfg.graph:
            continue
        
        src_id = block_to_id[block.addr]
        for successor in cfg.graph.successors(block):
            # Ensure successor is part of the adjacency matrix
            if successor.addr in block_to_id:
                dest_id = block_to_id[successor.addr]
                adjacency_matrix[src_id, dest_id] = 1

    return adjacency_matrix




binary_paths=[]
    
json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'
# Read the JSON file and load its contents as a dictionary
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)

with open('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_with_8_pass_seqs.json', 'r') as json_file:
    master_dict = json.load(json_file)

directory_path = "/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset"

#Code for populating binary_paths 
for dirpath, _, filenames in os.walk(directory_path):
    if dirpath.endswith('submissions'):
        c_directory = os.path.join(dirpath, 'C')
        if os.path.exists(c_directory) and c_directory in data_dict["classes_train"]:
            class_dict = master_dict[c_directory]

            # if len(binary_paths)>=2:
            #     break

            for _ in range(0,100):
                v2v_config = random.choice(list(class_dict['v2v'].keys()))
                filename = random.choice(class_dict['v2v'][v2v_config])
                filename_without_extension = os.path.splitext(filename)[0]
                binary_filename = filename_without_extension + '.out'
                binary_filepath = os.path.join(c_directory, v2v_config, 'unstripped', binary_filename)
                if os.path.exists(binary_filepath):
                    binary_paths.append(binary_filepath)


max_adjmat_size = 0
bincnt=0
for binary_path in binary_paths:
    bincnt+=1
    adj_matrix = extract_basic_block_adjacency(binary_path)
    max_adjmat_size = max(max_adjmat_size, adj_matrix.shape[0])

print(f"Number of binaries considered: {bincnt}") #9962
print(f"Max adj matrix size: {max_adjmat_size}") #110