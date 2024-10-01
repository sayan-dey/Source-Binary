# For copying unstr data files from x86-clang-8-O0 only for testset of COFO-Dataset to clang-14 dirs

# Sample cmd:
# python copy-unstr-data.py

import os
import subprocess
import shutil
import argparse
import json

import concurrent.futures

# Root directory path
root_dir = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'

# unstripped-data-inlined-extcall-rw-n0-newseedembed

    
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

for compiler in compilers:
    for opt in opts:
        
        print(f"\nRunning for {compiler} and {opt}...\n")
        
        # flag = False
        for dirpath, _, filenames in os.walk(root_dir):
            if dirpath.endswith('submissions'):
                # if dirpath == '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/1130-A/submissions':
                #     flag = True
                # if flag is False:
                #     continue
                c_directory = os.path.join(dirpath, 'C')
                if os.path.exists(c_directory) and c_directory in train_test_data_dict["classes_test"]: #for only generating infer set
                    src_dir = os.path.join(c_directory, 'x86-clang-8-O0')
                    bindirpath = os.path.join(src_dir, 'unstripped')
                    unstr_directory_name = "unstripped-data-inlined-extcall-rw-n0-newseedembed"
                    src_directory_path = os.path.join(src_dir, unstr_directory_name)
                    dest_directory_path = os.path.join(c_directory, f'{compiler}-{opt}', unstr_directory_name)

                    if not os.path.exists(dest_directory_path):
                        os.makedirs(dest_directory_path)
                    
                    # done till here ...

                    for file_name in os.listdir(bindirpath):
                        filename_without_extension = os.path.splitext(file_name)[0]
                        # Create the new file name with the new extension
                        c_filename = filename_without_extension + '.c'
                        if file_name.endswith('.out') and c_filename in data_dict[c_directory]:
                            data_filename = filename_without_extension + '.data'
                            src_unstr_datafile_path = os.path.join(src_directory_path, data_filename)
                            dest_unstr_datafile_path = os.path.join(dest_directory_path, data_filename)
                            # file_paths.append(file_path)
                            cp_cmd = f"cp {src_unstr_datafile_path} {dest_unstr_datafile_path}"
                            subprocess.run(cp_cmd, shell=True, capture_output=True, text=True)
                            print(f"Copied {src_unstr_datafile_path} \n to {dest_unstr_datafile_path}")
                            # exit()
                                                           
