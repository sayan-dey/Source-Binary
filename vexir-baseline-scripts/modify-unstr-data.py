# Script for modifying angr-vex addresses of  COFO-Dataset's unstripped data files' 
# main functions to keep them the same as in their stripped versions  
# Only for clang-14

# Sample cmd:
# python genvexir-data.py -vocab /Pramana/VexIR2Vec/checkpoint/ckpt_3M_900E_128D_0.0002LR_adam/seedEmbedding_3M_900E_128D_0.0002LR_adam -angr_path /Pramana/VexIR2Vec/Source_Binary/vexIR/angr-vex -fchunks 5 -num_threads 10

import os
import subprocess
import shutil
import argparse
import json
import pandas as pd

import concurrent.futures

# Root directory path
root = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'
    
# Specify the path to your JSON file containing required files to consider
json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_filedict.json'

# Read the JSON file and load its contents as a dictionary
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)


# For only generating data files req for inferencing...

# Specify the path to your JSON file containing required files to consider 
json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'

# Read the JSON file and load its contents as a dictionary
with open(json_file_path, 'r') as json_file:
    train_test_data_dict = json.load(json_file)



# compilers = ['clang-8', 'clang-10', 'clang-12']
# compilers = ['clang-8']
compilers = ['llvm14']
opts = ['O0', 'O1', 'O2', 'O3', 'Os', 'pass-seq-1', 'pass-seq-2', 
'pass-seq-3', 'pass-seq-4', 'pass-seq-5', 'pass-seq-6', 'pass-seq-7', 'pass-seq-8']
# opts = ['O0', 'O1', 'O2', 'O3', 'Os']
# opts = ['pass-seq-1', 'pass-seq-2', 'pass-seq-3', 'pass-seq-4', 'pass-seq-5', 
# 'pass-seq-6', 'pass-seq-7', 'pass-seq-8']

for compiler in compilers:
    for opt in opts:
        
        print(f"\nRunning for {compiler} and {opt}...\n")
        
        # flag = False
        for dirpath, _, filenames in os.walk(root):
            if dirpath.endswith('submissions'):
                # if dirpath == '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/1130-A/submissions':
                #     flag = True
                # if flag is False:
                #     continue
                c_directory = os.path.join(dirpath, 'C')
                if os.path.exists(c_directory) and c_directory in train_test_data_dict["classes_test"]: #for only generating infer set
                    str_dir = os.path.join(c_directory, f'{compiler}-{opt}','stripped-data-inlined-extcall-rwk50n2-n2-newseedembed')
                    unstr_dir = os.path.join(c_directory, f'{compiler}-{opt}','unstripped-data-inlined-extcall-rw-n0-newseedembed')

                    for file_name in os.listdir(unstr_dir):
                        filename_without_extension = os.path.splitext(file_name)[0]
                        # Create the new file name with the new extension
                        c_filename = filename_without_extension + '.c'
                        if c_filename in data_dict[c_directory]:
                            
                            str_file_path = os.path.join(str_dir, file_name)
                            col_names_str = ['addr', 'key', 'fnName', 'strRefs', 'extlibs', 'embed']
                            str_df = pd.read_csv(str_file_path, sep='\t', names=col_names_str, header=None)
                            str_row_index = str_df[str_df['fnName'] == 'main'].index
                            str_addr = str_df.loc[str_row_index, 'addr']
                            
                            unstr_file_path = os.path.join(unstr_dir, file_name)
                            col_names_unstr = ['addr', 'key', 'fnName', 'embed_unst']
                            unstr_df = pd.read_csv(unstr_file_path, sep='\t', names=col_names_unstr, header=None)
                            
                            unstr_row_index = unstr_df[unstr_df['fnName'] == 'main'].index
                            unstr_df.loc[unstr_row_index, 'addr'] = str(str_addr.values[0])

                            #Modifying the unstr datafile by putting address of main func same as its str version 
                            unstr_df.to_csv(unstr_file_path, sep='\t', header=False, index=False)

                            print(f"Processed {unstr_file_path}")
                            
                            
                        
                    
