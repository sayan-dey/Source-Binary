#For generating test data csv for v2v binary-binary matching

# script for generating csv file that can be used for inferencing
# only considering llvm14 O3 data for i2v for inferencing

import os
import csv
import re
import pandas as pd
import argparse
from collections import defaultdict
import json
import random
import numpy as np
import sys

sys.path.append("/Pramana/VexIR2Vec/Source_Binary/scripts/scripts-model")

from utils import SEED

random.seed(SEED)
np.random.seed(SEED)
# torch.random.manual_seed(SEED)

#for removing line no from vexir key
def modify_key(x):
    # "('/home/es17btech11025/binsim_dataset/findutils-4.9.0/gl/lib/mbchar.h', 241)mb_width_aux"
    # output = ('/home/es17btech11025/binsim_dataset/findutils-4.9.0/gl/lib/mbchar.h')mb_width_aux
    #print (x)
    x=re.split(', |\)', x)
    modified_str = x[0]+")"+x[2]
    #print(modified_str)
    return modified_str

#for bringing vexir key to llvm ir key format
def format_key(input_string):
    # Define the pattern to match
    pattern = r"\(\'(.*?)\'\)(.*)"
    
    # Use regular expression to match the pattern
    match = re.match(pattern, input_string)
    
    if match:
        # Extract the file path and function name
        file_path = match.group(1)
        function_name = match.group(2).strip()
        
        # Concatenate file path and function name with a colon
        modified_string = f"{file_path}:{function_name}"
        
        return modified_string
    else:
        return input_string  # Return the original string if no match found


#creating the dict from where we will choose randomly
def generate_master_dict(root):
    master_dict = {}
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory) and c_directory in data_dict:
                # master_dict[c_directory] ={}
                class_dict = {}
                class_dict['v2v']={}
                class_dict['i2v']={}
                for v2v_config in v2v_configs:
                    class_dict['v2v'][v2v_config] = data_dict[c_directory]
                for i2v_config in i2v_configs:
                    class_dict['i2v'][i2v_config] = data_dict[c_directory]
                    
                master_dict[c_directory] = class_dict
            
    #----Master dict creation done---------
    
    output_file = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_infer.json'
    with open(output_file, 'w') as file:
        json.dump(master_dict, file, indent=4)

def gen_vexir_df_bin(c_directory, class_dict): #For original binaries of Bin-Src project
    #for v2v file
    v2v_config = random.choice(list(class_dict['v2v'].keys()))   
    # print(v2v_config)
    filename = random.choice(class_dict['v2v'][v2v_config])
    filename_without_extension = os.path.splitext(filename)[0]
    v2v_filename= filename_without_extension + '.data'
    v2v_filepath = os.path.join(c_directory, v2v_config, 'stripped-k72-n2-patched-cgv-vdb-data', v2v_filename)
    col_names_str = ['addr', 'key', 'fnName', 'strRefs', 'extlibs', 'embed_O', 'embed_T', 'embed_A']
    try:
        v2v_st_df = pd.read_csv(v2v_filepath, sep='\t', names=col_names_str, header=None)
    except:
        v2v_st_df = pd.DataFrame(columns=['key_bin', 'strembed_bin','libemb_bin','embed_O_bin', 'embed_T_bin', 'embed_A_bin'])
        return v2v_st_df #returning empty dataframe if file not found
        
    v2v_st_df = v2v_st_df[v2v_st_df['fnName'] == 'main']
    v2v_st_df.reset_index(drop=True, inplace=True) # Reset the index to start from 0
    v2v_st_df.drop(['addr', 'fnName'], axis=1, inplace=True)
    v2v_st_df['key'] = c_directory
    v2v_st_df.rename(columns={'key': 'key_bin', 'strRefs': 'strembed_bin', 'extlibs': 'libemb_bin', 
    'embed_O': 'embed_O_bin', 'embed_T': 'embed_T_bin', 'embed_A': 'embed_A_bin'}, inplace=True)
    
    return v2v_st_df


# def gen_llvm_ir_df(c_directory, class_dict):

def gen_vexir_df_src(c_directory, class_dict): #For original sources of Bin-Src project
    #for i2v file
    i2v_config = random.choice(list(class_dict['i2v'].keys()))   
    # print(i2v_config)
    filename = random.choice(class_dict['i2v'][i2v_config])
    filename_without_extension = os.path.splitext(filename)[0]
    i2v_filename= filename_without_extension + '.data'
    i2v_filepath = os.path.join(c_directory, i2v_config, 'stripped-k72-n2-patched-cgv-vdb-data', i2v_filename)
    col_names_str = ['addr', 'key', 'fnName', 'strRefs', 'extlibs', 'embed_O', 'embed_T', 'embed_A']
    try:
        i2v_st_df = pd.read_csv(i2v_filepath, sep='\t', names=col_names_str, header=None)
    except:
        i2v_st_df = pd.DataFrame(columns=['key_src', 'strembed_src','libemb_src','embed_O_src', 'embed_T_src', 'embed_A_src'])
        return i2v_st_df #returning empty dataframe if file not found
        
    i2v_st_df = i2v_st_df[i2v_st_df['fnName'] == 'main']
    i2v_st_df.reset_index(drop=True, inplace=True) # Reset the index to start from 0
    i2v_st_df.drop(['addr', 'fnName'], axis=1, inplace=True)
    i2v_st_df['key'] = c_directory
    i2v_st_df.rename(columns={'key': 'key_src', 'strRefs': 'strembed_src', 'extlibs': 'libemb_src', 
    'embed_O': 'embed_O_src', 'embed_T': 'embed_T_src', 'embed_A': 'embed_A_src'}, inplace=True)
    
    return i2v_st_df

    
    

def generate_infer_data_csv(root):

    with open('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_infer.json', 'r') as json_file:
        master_dict = json.load(json_file)
    
    # columns = ['key_v2v','embed_v2v','key_i2v','embed_i2v']
    columns = ['key_bin', 'strembed_bin','libemb_bin','embed_O_bin', 'embed_T_bin', 'embed_A_bin',
    'key_src', 'strembed_src','libemb_src','embed_O_src', 'embed_T_src', 'embed_A_src']
    
    # Create an empty DataFrame with only column names
    data_df = pd.DataFrame(columns=columns)
    
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory) and c_directory in data_dict["classes_test"]:
                class_dict = master_dict[c_directory]

                # print(class_dict)

                # ----------creating similar pairs------
                for _ in range(0,100):
                    
                    v2v_st_df = gen_vexir_df_bin(c_directory, class_dict)
                    if v2v_st_df.empty:
                        print("The v2v_st_df DataFrame is empty")
                        continue
                    
                    i2v_df = gen_vexir_df_src(c_directory, class_dict)
                    if i2v_df.empty:
                        print("The i2v_df DataFrame is empty")
                        continue
                    
                    #merging the two dfs
                    df_concat = pd.merge(v2v_st_df, i2v_df, left_index=True, right_index=True) 

                    # print(df_concat)
                    # df_concat.to_csv('/Pramana/VexIR2Vec/Source_Binary/test/df_concat.csv', index=False)
                    # exit()
                    
                    # Concatenate along rows (axis=0, default behavior) to the original df
                    data_df = pd.concat([data_df, df_concat])
                    # print(data_df)
                    # exit()
                    
                
                print(f"Processed: {c_directory}")
    
    
    data_df.to_csv('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/baseline-data/C_data_infer_v2v_baseline.csv', index=False)                
                    
                    

parser = argparse.ArgumentParser()
parser.add_argument('-input', '--input_dir', dest='input_dir',
                    help='input dir path', default=None)    
parser.add_argument('-output', '--output_path', dest='output_path',
                    help='output csv file path', default=None)
args = parser.parse_args()


# Specify the master directory
master_directory = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'
# master_directory = args.input_dir

# if not os.path.exists(args.output_dir):
#     os.makedirs(args.output_dir)
#     print(f"Directory '{args.output_dir}' created successfully")
# else:
#     print(f"Directory '{args.output_dir}' already exists")

# (Vex IR config, LLVM IR config)
# configs = [('x86-clang-12-O0', 'x86-clang-14-O0')]

compilers = ['clang-8', 'clang-10', 'clang-12']
opts = ['O0', 'O1', 'O2', 'O3', 'Os']
v2v_configs = [] 
i2v_configs = []

for compiler in compilers:
    for opt in opts:
        v2v_config = f'x86-{compiler}-{opt}'
        v2v_configs.append(v2v_config)
        

#only considering llvm14 O3 data for i2v for inferencing

i2v_config = 'llvm14-O3'
i2v_configs.append(i2v_config) 
    
    
# Specify the path to your JSON file containing required files to consider
json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'
# Read the JSON file and load its contents as a dictionary
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)

# for creating the dict from where we will choose randomly
# generate_master_dict(master_directory)

# Call the function to generate inference data
generate_infer_data_csv(master_directory)

#------------------------------------------------------------------------------------------------------



