

# Script for parallely generating angr-vex data files (according to new vexir format)
# from .out files in COFO-Dataset 

# Sample cmd:
# python gen_datafiles.py -vocab /Pramana/VexIR2Vec/checkpoint/ckpt_3M_900E_128D_0.0002LR_adam/seedEmbedding_3M_900E_128D_0.0002LR_adam -angr_path /Pramana/VexIR2Vec/vexIR-repo-sayan/angr-vex -fchunks 2 -num_threads 12

import os
import subprocess
import shutil
import argparse
import json

import concurrent.futures

# Root directory path
root_dir = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'

'''
# Function to process a single file
def process_file(file_path, dirpath, angr_vex_cmd):
    subprocess.run(angr_vex_cmd, shell=True, cwd=os.path.abspath(os.path.join(dirpath, '..')), capture_output=True, text=True)
    print(f"Processed: {file_path}")
'''
    
# Function to process a single file with angr-vex.py
def process_file(file_path, angr_vex_cmd):
    # print(angr_vex_cmd)
    # exit()
    subprocess.run(angr_vex_cmd, shell=True, capture_output=True, text=True)
    print(f"Processed: {file_path}")

# Modify execute_angr_command function
def execute_angr_command(root, max_workers):

    # For only generating data files req for inferencing...

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

    
    compilers = ['x86-clang-8', 'x86-clang-10', 'x86-clang-12']
    # compilers = ['clang-8']  
    # compilers = ['llvm14']
    opts = ['O0', 'O1', 'O2', 'O3', 'Os']
    # opts = ['O0', 'O1', 'O2', 'O3', 'Os', 'pass-seq-1', 'pass-seq-2', 'pass-seq-3', 'pass-seq-4', 'pass-seq-5', 'pass-seq-6', 'pass-seq-7', 'pass-seq-8']

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
                        # config_dir = os.path.join(c_directory, f'x86-{compiler}-{opt}')
                        # config_dir = os.path.join(c_directory, f'{compiler}-{opt}')
                        
                        config_dir = os.path.join(c_directory,  f'{compiler}-{opt}')
                        
                        # bindir_list = ['stripped', 'unstripped']
                        bindir_list = ['stripped']
                        
                        # Split the path into its components
                        path_parts = config_dir.split(os.sep)
                        class_name = path_parts[5] #problem number
                        # print(config_dir)
                        # print(class_name)
                        # exit()
                        
                        for bindir in bindir_list:
                            bindirpath = os.path.join(config_dir, bindir)
                            # new_directory_name = f"{bindir}-data-inlined-extcall-rwk50n2-n2-newseedembed-patched-db"
                            new_directory_name = f"{bindir}-k72-n2-patched-cgv-vdb-data"
                            new_directory_path = os.path.join(config_dir, new_directory_name)
                            
                            if not os.path.exists(new_directory_path):
                                os.makedirs(new_directory_path)
                            
                            file_paths = []
                            data_file_paths = []

                            for file_name in os.listdir(bindirpath):
                                filename_without_extension = os.path.splitext(file_name)[0]
                                # Create the new file name with the new extension
                                c_filename = filename_without_extension + '.c'
                                if file_name.endswith('.out') and c_filename in data_dict[c_directory]:
                                    file_path = os.path.join(bindirpath, file_name)
                                    file_paths.append(file_path)
                                    data_file_paths.append(os.path.join(new_directory_path, os.path.splitext(os.path.basename(file_path))[0] + ".data"))
                                    
                            
                            if file_paths:
                                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    for file_path in file_paths:
                                        # Define the corresponding data file path for the file
                                        data_file_path = os.path.join(new_directory_path, f"{os.path.splitext(os.path.basename(file_path))[0]}.data")
                                        
                                        embedding_db_path = os.path.join('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/vexirDB',f'{class_name}.db') 
                                        
                                        # Construct the angr_vex_cmd for the current file
                                        # angr_vex_cmd = f"python {args.angr_path}/angr_db_metadata_dump.py -b {file_path} -adb {new_directory_path} -edb {embedding_db_path} -t 1"
                                        angr_vex_cmd = f"python {args.angr_path}/angr-vex.py -b {file_path} -o {data_file_path} -v {args.vocab} -d -n 2 -edb {embedding_db_path} -fchunks {args.fchunks}"

                                        # Submitting a task for each file to process it with angr-vex.py
                                        executor.submit(process_file, file_path, angr_vex_cmd)
                            
                    

# Rest of the code remains the same...
parser = argparse.ArgumentParser()
parser.add_argument('-vocab', '--vocabulary_path',  dest='vocab', type=str, help='seed embedding vocab path')
parser.add_argument('-fchunks', '--num_func_chunks',  dest='fchunks', type=int, default=32, help='number of func chunks')
parser.add_argument('-angr_path', '--angr_vex_path',  dest='angr_path', type=str, help='angr vex path')
parser.add_argument('-num_threads', '--number_of_threads',  dest='num_threads', type=int, default=1, help='max number of parallel tasks')


args = parser.parse_args()

# For generating data (.data) files for binaries with a maximum of num_threads parallel tasks per directory
execute_angr_command(root_dir, args.num_threads)
