# Script for parallely generating angr-vex data files from .out files in COFO-Dataset (using norm_tokens branch of vexIR)

# Sample cmd:
# python genvexir-data-parallel.py -vocab /Pramana/VexIR2Vec/checkpoint/ckpt_3M_600E_128D_0.002LR_adam_BS256/seedEmbedding_3M_600E_128D_0.002LR_adam_BS256 -angr_path /Pramana/VexIR2Vec/Source_Binary/vexIR/angr-vex -fchunks 5 -num_threads 10

import os
import subprocess
import shutil
import argparse

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
def process_file(file_path, data_file_path, angr_vex_cmd):
    subprocess.run(angr_vex_cmd, shell=True, capture_output=True, text=True)
    print(f"Processed: {file_path}")

# Modify execute_angr_command function
def execute_angr_command(root, max_workers):
    # flag = False
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            # if dirpath == '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/1130-A/submissions':
            #     flag = True
            # if flag is False:
            #     continue
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory):
                config_dir = os.path.join(c_directory, 'x86-clang-12-O0')
                bindir_list = ['stripped', 'unstripped']

                for bindir in bindir_list:
                    bindirpath = os.path.join(config_dir, bindir)
                    new_directory_name = f"{bindir}-data-w7-s1-inlined-extcall-rw-norm2-inst"
                    new_directory_path = os.path.join(config_dir, new_directory_name)
                    
                    if not os.path.exists(new_directory_path):
                        os.makedirs(new_directory_path)
                    
                    file_paths = []
                    data_file_paths = []

                    for file_name in os.listdir(bindirpath):
                        if file_name.endswith('.out'):
                            file_path = os.path.join(bindirpath, file_name)
                            file_paths.append(file_path)
                            data_file_paths.append(os.path.join(new_directory_path, os.path.splitext(os.path.basename(file_path))[0] + ".data"))
                            
                    
                    if file_paths:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            for file_path in file_paths:
                                # Define the corresponding data file path for the file
                                data_file_path = os.path.join(new_directory_path, f"{os.path.splitext(os.path.basename(file_path))[0]}.data")

                                # Construct the angr_vex_cmd for the current file
                                angr_vex_cmd = f"python {args.angr_path}/angr-vex.py -b {file_path} -o {data_file_path} -v {args.vocab} -d -fchunks {args.fchunks} -n 2"

                                # Submitting a task for each file to process it with angr-vex.py
                                executor.submit(process_file, file_path, data_file_path, angr_vex_cmd)
                    
                    '''
                    if file_paths:
                        new_directory_name = f"{bindir}-data-w7-s1-inlined-extcall-rw-norm2-inst"
                        new_directory_path = os.path.join(config_dir, new_directory_name)

                        if not os.path.exists(new_directory_path):
                            os.makedirs(new_directory_path)

                        angr_vex_cmd = f"python {args.angr_path}/angr-vex.py -b {' '.join(file_paths)} -o {' '.join(data_file_paths)} -v {args.vocab} -d -fchunks {args.fchunks} -n 2"

                        # Limiting the number of concurrent tasks
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [executor.submit(process_file, file_path, dirpath, angr_vex_cmd) for file_path, data_file_path in zip(file_paths, data_file_paths)]
                            for future in concurrent.futures.as_completed(futures):
                                future.result()
                    '''
                    

# Rest of the code remains the same...
parser = argparse.ArgumentParser()
parser.add_argument('-vocab', '--vocabulary_path',  dest='vocab', type=str, help='seed embedding vocab path')
parser.add_argument('-fchunks', '--num_func_chunks',  dest='fchunks', type=int, default=32, help='number of func chunks')
parser.add_argument('-angr_path', '--angr_vex_path',  dest='angr_path', type=str, help='angr vex path')
parser.add_argument('-num_threads', '--number_of_threads',  dest='num_threads', type=int, default=1, help='max number of parallel tasks')


args = parser.parse_args()

# For generating data (.data) files for binaries with a maximum of num_threads parallel tasks per directory
execute_angr_command(root_dir, args.num_threads)
