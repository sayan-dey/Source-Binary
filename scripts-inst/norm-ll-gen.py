# Sample cmd for generating .ll file from source: ./bin/clang -S -emit-llvm /Pramana/VexIR2Vec/Source_Binary/test/hello.c -o /Pramana/VexIR2Vec/Source_Binary/test/hello.ll

# Sample cmd for generating normalized ll: /Pramana/VexIR2Vec/Source_Binary/IR2Vec/build/bin/ir2vec -collectIR -o test.ll /Pramana/VexIR2Vec/Source_Binary/test/hello.ll

#Script for generating normalized ll files using collectIR of IR2vec
# source code -> ll -> normalized ll (LLVM 14 used all through)

import os
import subprocess
import shutil

# Root directory path
root_dir = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'

# Function to find and execute the command for C files in submissions directories
def execute_clang_command(root):
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory):
                # print(c_directory)
                
                for file_name in os.listdir(c_directory):
                    if file_name.endswith('.c'):
                        file_path = os.path.join(c_directory, file_name)
                        output_path = file_path[:-2] + '.ll'  # Creating output file path

                        # Constructing the command
                        clang_command = f"/home/cs22mtech02005/llvm-project-14/build/bin/clang -S -emit-llvm {file_path} -o {output_path}"
                        # print("file name: ", file_path)
                        # print("output path: ", output_path)

                        # Executing the command
                        subprocess.run(clang_command, shell=True, cwd=os.path.abspath(os.path.join(dirpath, '..')), capture_output=True, text=True)
                        print(f"Processed: {file_path}")


# Function to execute ir2vec command on already generated ll files
def execute_ir2vec_command(root):
    for dirpath, _, filenames in os.walk(root):
        for file_name in filenames:
            if file_name.endswith('.ll'):
                file_path = os.path.join(dirpath, file_name)
                
                # print("File path: ", file_path)
                # exit()

                # Creating output directory 'norm-ll' in the parent directory of ll files
                output_dir = os.path.join(os.path.dirname(file_path), 'norm-ll')
                os.makedirs(output_dir, exist_ok=True)

                # Constructing the ir2vec command
                ir2vec_command = f"/Pramana/VexIR2Vec/Source_Binary/IR2Vec/build/bin/ir2vec -collectIR -o {os.path.join(output_dir, file_name)} {file_path}"

                # Executing the ir2vec command
                subprocess.run(ir2vec_command, shell=True, capture_output=True, text=True)
                print(f"Processed: {file_path}")



# For generating ll files for C codes
# execute_clang_command(root_dir) 

# For generating normalized ll files (as triplets) in norm-ll dir from above ll files
execute_ir2vec_command(root_dir)
