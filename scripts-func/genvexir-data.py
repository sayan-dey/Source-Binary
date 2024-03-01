# Script for generating angr-vex data files from .out files in COFO-Dataset (using norm_tokens branch of vexIR)

# Sample cmd:
# python genvexir-data.py -vocab /Pramana/VexIR2Vec/checkpoint/ckpt_3M_600E_128D_0.002LR_adam_BS256/seedEmbedding_3M_600E_128D_0.002LR_adam_BS256 -angr_path /Pramana/VexIR2Vec/Source_Binary/vexIR/angr-vex -fchunks 10

import os
import subprocess
import shutil
import argparse

# Root directory path
root_dir = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'

# Function to find and execute the command for C files in submissions directories
def execute_angr_command(root):
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory):
                # print(c_directory)
                config_dir = os.path.join(c_directory, 'x86-clang-14-O0')
                bindir_list = ['stripped','unstripped']

                for bindir in bindir_list:

                    bindirpath = os.path.join(config_dir, bindir)
                    # print(bindirpath) #for checking
                    
                    for file_name in os.listdir(bindirpath):
                        if file_name.endswith('.out'):
                            file_path = os.path.join(bindirpath, file_name)

                            # Extract the filename without the extension
                            file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]
                            
                            # Define the new directory name
                            new_directory_name = f"{bindir}-data-w7-s1-inlined-extcall-rw-norm2-inst"
                            
                            # Create the new directory path
                            new_directory_path = os.path.join(config_dir, new_directory_name)
                            
                            # Create the directory if it doesn't exist
                            if not os.path.exists(new_directory_path):
                                os.makedirs(new_directory_path)
                            
                            # Create the .data file path
                            data_file_path = os.path.join(new_directory_path, file_name_without_extension + ".data")
                            # print("data_file_path: ",data_file_path) #working fine

                            # print("file_path: ", file_path)
                            # print("data_file_path: ", data_file_path)

                            # exit()

                            #need to invoke angr from here
                            angr_vex_cmd = f"python {args.angr_path}/angr-vex.py -b {file_path} -o {data_file_path} -v {args.vocab} -d -fchunks {args.fchunks} -n 2"

                            # Executing the command
                            subprocess.run(angr_vex_cmd, shell=True, cwd=os.path.abspath(os.path.join(dirpath, '..')), capture_output=True, text=True)
                            print(f"Processed: {file_path}")




parser = argparse.ArgumentParser()
parser.add_argument('-vocab', '--vocabulary_path',  dest='vocab', type=str, help='seed embedding vocab path')
parser.add_argument('-fchunks', '--num_func_chunks',  dest='fchunks', type=str, help='number of func chunks')
parser.add_argument('-angr_path', '--angr_vex_path',  dest='angr_path', type=str, help='angr vex path')


args = parser.parse_args()

# For generating data (.data) files for binaries
execute_angr_command(root_dir) 


