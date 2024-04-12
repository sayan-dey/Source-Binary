# Script for generating .out files (both stripped and unstripped) from .C files of COFO-Dataset
# Sample cmd: python generate_Binaries.py -num_threads 15

import os
import subprocess
import shutil
import argparse

import concurrent.futures

# Root directory path
root_dir = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'


# Function to process a single C file 
def process_file(file_path, strip_cmd_1, strip_cmd_2, unstrip_cmd):
    
    # Executing the commands
    subprocess.run(strip_cmd_1, shell=True, capture_output=True, text=True)
    subprocess.run(strip_cmd_2, shell=True, capture_output=True, text=True)
    subprocess.run(unstrip_cmd, shell=True, capture_output=True, text=True)
    
    # print(strip_cmd_1)
    # print(strip_cmd_2)
    # print(unstrip_cmd)
    
    print(f"Processed: {file_path}")

# Function to find and execute the command for C files in submissions directories
def execute_clang_command(root, max_workers):
    
    compilers = ['clang-8', 'clang-10', 'clang-12']
    opts = ['O0', 'O1', 'O2', 'O3', 'Os']
    
    for compiler in compilers:
        for opt in opts:
            
            print(f"\nRunning for {compiler} and {opt}...\n")
            
            for dirpath, _, filenames in os.walk(root):
                if dirpath.endswith('submissions'):
                    c_directory = os.path.join(dirpath, 'C')
                    if os.path.exists(c_directory):
                        # print(c_directory)
                        
                        file_paths = []
                        
                        for file_name in os.listdir(c_directory):
                            if file_name.endswith('.c'):
                                file_path = os.path.join(c_directory, file_name)
                                file_paths.append(file_path)
                        
                        
                        if file_paths:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                for file_path in file_paths:
                                    
                                    file_name = os.path.basename(file_path)
                            
                                    parent_directory = os.path.dirname(file_path)
                                    # output_path = file_path[:-2] + '.out'  # Creating output file path

                                    file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]

                                
                                    new_directory_name = f"x86-{compiler}-{opt}"
                                
                                    # print("new_directory_name: ", new_directory_name)
                                    # exit()
                                
                                    new_directory_path = os.path.join(parent_directory, new_directory_name)
                                    if not os.path.exists(new_directory_path):
                                        os.makedirs(new_directory_path)

                                    stripped_directory_path = os.path.join(new_directory_path, "stripped")
                                    unstripped_directory_path = os.path.join(new_directory_path, "unstripped")

                                    if not os.path.exists(stripped_directory_path):
                                        os.makedirs(stripped_directory_path)

                                    if not os.path.exists(unstripped_directory_path):
                                        os.makedirs(unstripped_directory_path)
                                        
                                    stripped_bin_path = os.path.join(stripped_directory_path, file_name_without_extension + ".out")

                                    unstripped_bin_path = os.path.join(unstripped_directory_path, file_name_without_extension + ".out")
                                

                                    # Constructing the commands
                                    strip_cmd_1 = f"{compiler} -o {stripped_bin_path} -g0 -{opt} {file_path}"
                                    strip_cmd_2 = f"strip -s {stripped_bin_path} -o {stripped_bin_path}"
                                    unstrip_cmd = f"{compiler} -o {unstripped_bin_path} -g -{opt} {file_path}"
                                    
                                    # Submitting a task for each file to process it and generate its binary using clang
                                    executor.submit(process_file, file_path, strip_cmd_1, strip_cmd_2, unstrip_cmd)
                



parser = argparse.ArgumentParser()
parser.add_argument('-num_threads', '--number_of_threads',  dest='num_threads', type=int, default=1, help='max number of parallel tasks')
args = parser.parse_args()

# For generating binaries (.out) files for C codes
execute_clang_command(root_dir, args.num_threads) 