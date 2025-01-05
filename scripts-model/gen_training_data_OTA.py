
#script for creating the dict from where we will choose randomly and generating csv file that can be used for training (and inferencing)

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
    
    output_file = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_with_8_pass_seqs.json'
    with open(output_file, 'w') as file:
        json.dump(master_dict, file, indent=4)
     

def gen_vexir_df(c_directory, class_dict):

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
        v2v_st_df = pd.DataFrame(columns=['key_v2v', 'strembed_v2v','libemb_v2v','embed_O_v2v', 'embed_T_v2v', 'embed_A_v2v'])
        return v2v_st_df #returning empty dataframe if file not found
        
    v2v_st_df = v2v_st_df[v2v_st_df['fnName'] == 'main']
    v2v_st_df.reset_index(drop=True, inplace=True) # Reset the index to start from 0
    v2v_st_df.drop(['addr', 'fnName'], axis=1, inplace=True)
    v2v_st_df['key'] = c_directory
    v2v_st_df.rename(columns={'key': 'key_v2v', 'strRefs': 'strembed_v2v', 'extlibs': 'libemb_v2v', 
    'embed_O': 'embed_O_v2v', 'embed_T': 'embed_T_v2v', 'embed_A': 'embed_A_v2v'}, inplace=True)
    
    return v2v_st_df



def gen_llvm_ir_df(c_directory, class_dict):
    #for i2v file
    i2v_config = random.choice(list(class_dict['i2v'].keys()))   
    # print(i2v_config)
    filename = random.choice(class_dict['i2v'][i2v_config])
    filename_without_extension = os.path.splitext(filename)[0]
    i2v_filename= filename_without_extension + '.txt'
    i2v_filepath = os.path.join(c_directory, i2v_config, 'ir2vec-data-sym-rwk20n2-StrLib-embed-OTA', i2v_filename)
    try:
        i2v_df = pd.read_csv(i2v_filepath, sep='\t', header=None)
    except:
        i2v_df = pd.DataFrame(columns=['embed_O_i2v', 'embed_T_i2v', 'embed_A_i2v', 'strembed_i2v', 'libemb_i2v', 'key_i2v'])
        return i2v_df #returning empty dataframe if file not found
    
    
    
    '''
    # i2v_df = i2v_df.drop(columns=[301]) #this col contains null val
    merged_column = i2v_df.iloc[:, 1:301].values.flatten()
    i2v_embed_O = np.array(merged_column) # Convert the merged column to a NumPy array
    i2v_df = i2v_df.drop(i2v_df.columns[1:301], axis=1) # Drop the original columns
    i2v_df['embed_O'] = str(i2v_embed_O) 

    merged_column = i2v_df.iloc[:, 301:601].values.flatten()
    i2v_embed_T = np.array(merged_column) # Convert the merged column to a NumPy array
    i2v_df = i2v_df.drop(i2v_df.columns[301:601], axis=1) # Drop the original columns
    i2v_df['embed_T'] = str(i2v_embed_T)

    merged_column = i2v_df.iloc[:, 601:901].values.flatten()
    i2v_embed_A = np.array(merged_column) # Convert the merged column to a NumPy array
    i2v_df = i2v_df.drop(i2v_df.columns[601:901], axis=1) # Drop the original columns
    i2v_df['embed_A'] = str(i2v_embed_A)
    '''

    df_combined = pd.DataFrame()

    df_combined['key_i2v'] = i2v_df.iloc[:, 0]
    df_combined['embed_O_i2v'] = i2v_df.iloc[:, 1:301].apply(lambda row: row.values, axis=1)
    df_combined['embed_T_i2v'] =  i2v_df.iloc[:, 301:601].apply(lambda row: row.values, axis=1)
    df_combined['embed_A_i2v'] = i2v_df.iloc[:, 601:901].apply(lambda row: row.values, axis=1)


    df_combined['strembed_i2v'] = i2v_df.iloc[:, 901]
    df_combined['libemb_i2v'] = i2v_df.iloc[:, 902]

    df_combined.drop(['key_i2v'], axis=1, inplace=True)
    df_combined['key_i2v'] = c_directory


    i2v_df = df_combined

    # i2v_df.to_csv('/Pramana/VexIR2Vec/Source_Binary/test/C_train_data_OTA_test.csv', index=False)  
    # exit()

    '''
    i2v_df = i2v_df.rename(columns={i2v_df.columns[0]: 'key'}) # Rename the column at the 0th index
    # i2v_df.drop(['key'], axis=1, inplace=True)
    i2v_df['key'] = c_directory
    # i2v_df.rename(columns={'key': 'key_i2v', 'embed': 'embed_i2v'}, inplace=True)
    i2v_df.rename(columns={'key': 'key_i2v', 901: 'strembed_i2v', 902: 'libemb_i2v', 'embed_O': 'embed_O_i2v', 'embed_T': 'embed_T_i2v', 'embed_A': 'embed_A_i2v'}, inplace=True)
    '''

    return i2v_df

    
def generate_train_data_csv(root):

    with open('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_with_8_pass_seqs.json', 'r') as json_file:
        master_dict = json.load(json_file)
    
    # columns = ['key_v2v','strembed_v2v','libemb_v2v','embed_v2v','key_i2v','strembed_i2v', 'libembed_i2v','embed_i2v', 'label']
    columns = ['key_v2v', 'strembed_v2v','libemb_v2v','embed_O_v2v', 'embed_T_v2v', 'embed_A_v2v',
    'embed_O_i2v', 'embed_T_i2v', 'embed_A_i2v', 'strembed_i2v', 'libemb_i2v', 'key_i2v']

    # Create an empty DataFrame with only column names
    data_df = pd.DataFrame(columns=columns)
    
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory) and c_directory in data_dict["classes_train"]:
                class_dict = master_dict[c_directory]

                # print(class_dict)

                # ----------creating similar pairs------
                for _ in range(0,100):
                    
                    v2v_st_df = gen_vexir_df(c_directory, class_dict)
                    if v2v_st_df.empty:
                        print("The v2v_st_df DataFrame is empty")
                        continue
                    
                    # v2v_st_df.to_csv('/Pramana/VexIR2Vec/Source_Binary/test/C_train_data_OTA_test.csv', index=False)  
                    # exit()
                    
                    i2v_df = gen_llvm_ir_df(c_directory, class_dict)
                    if i2v_df.empty:
                        print("The i2v_df DataFrame is empty")
                        continue

                    
                    #merging the two dfs
                    df_concat = pd.merge(v2v_st_df, i2v_df, left_index=True, right_index=True) 
                    df_concat['label'] = 1
                    # print(df_concat)
                    # df_concat.to_csv('/Pramana/VexIR2Vec/Source_Binary/test/df_concat.csv', index=False)
                    # exit()
                    
                    # Concatenate along rows (axis=0, default behavior) to the original df
                    data_df = pd.concat([data_df, df_concat])
                    # print(data_df)
                    # exit()
                    
                
                # ---------creating dissimilar pairs------
                for _ in range(0,100):
                    
                    v2v_st_df = gen_vexir_df(c_directory, class_dict)
                    if v2v_st_df.empty:
                        print("The v2v_st_df DataFrame is empty")
                        continue
                    
                    # Choose a random class excluding the current v2v class(i.e, c_directory)
                    i2v_class = random.choice([x for x in list(master_dict.keys()) if x != c_directory])
                    i2v_dict = master_dict[i2v_class]
                    
                    i2v_df = gen_llvm_ir_df(i2v_class, i2v_dict)
                    if i2v_df.empty:
                        print("The i2v_df DataFrame is empty")
                        continue
                    
                    #merging the two dfs
                    df_concat = pd.merge(v2v_st_df, i2v_df, left_index=True, right_index=True) 
                    df_concat['label'] = 0
                    # print(df_concat)
                    # df_concat.to_csv('/Pramana/VexIR2Vec/Source_Binary/test/df_concat.csv', index=False)
                    # exit()
                    
                    # Concatenate along rows (axis=0, default behavior) to the original df
                    data_df = pd.concat([data_df, df_concat])
                    # print(data_df)
                    
                    
                print(f"Processed: {c_directory}")
    

    data_df.to_csv('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_train_data_OTA_str_lib_with_8_pass_seqs.csv', index=False)  

                             
                    

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
        
for opt in opts:
    i2v_config = f'llvm14-{opt}'
    i2v_configs.append(i2v_config) 

for ind in range(1,9):
    i2v_config = f'llvm14-pass-seq-{ind}'
    i2v_configs.append(i2v_config) 

    
    
# Specify the path to your JSON file containing required files to consider
# json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_filedict.json'
json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'

# Read the JSON file and load its contents as a dictionary
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)

#for creating the dict from where we will choose randomly
# generate_master_dict(master_directory)

# Call the function to generate training data
generate_train_data_csv(master_directory)
exit()

#----------for testing-----------------

#for i2v file
# i2v_config = random.choice(list(class_dict['i2v'].keys()))   
# print(i2v_config)
# filename = random.choice(class_dict['i2v'][i2v_config])
# filename_without_extension = os.path.splitext(filename)[0]
# i2v_filename= filename_without_extension + '.txt'
# i2v_filepath = os.path.join(c_directory, i2v_config, 'ir2vec-data-sym-rwk20n2-StrLib-embed', i2v_filename)

i2v_filepath = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/1003-A/submissions/C/llvm14-O0/ir2vec-data-sym-rwk20n2-StrLib-embed/39898765.txt'
# i2v_filepath = '/Pramana/VexIR2Vec/Source_Binary/Programs/test3_i2v_strlibembed.txt'
try:
    i2v_df = pd.read_csv(i2v_filepath, sep='\t', header=None)
except:
    i2v_df = pd.DataFrame(columns=['key_i2v', 'strembed_i2v', 'libemb_i2v', 'embed_i2v'])
    # return i2v_df #returning empty dataframe if file not found



# i2v_df = i2v_df.drop(columns=[301]) #this col contains null val
merged_column = i2v_df.iloc[:, 1:301].values.flatten()
i2v_embed = np.array(merged_column) # Convert the merged column to a NumPy array
i2v_df = i2v_df.drop(i2v_df.columns[1:301], axis=1) # Drop the original columns
i2v_df['embed'] = str(i2v_embed) 
i2v_df = i2v_df.rename(columns={i2v_df.columns[0]: 'key'}) # Rename the column at the 0th index
# i2v_df.drop(['key'], axis=1, inplace=True)
i2v_df['key'] = 'c_directory'
# exit()
# i2v_df = i2v_df.rename(columns={i2v_df.columns[301]: 'strembed'})
# i2v_df = i2v_df.rename(columns={i2v_df.columns[302]: 'libembed'})
i2v_df.rename(columns={'key': 'key_i2v', 301: 'strembed_i2v', 302: 'libemb_i2v', 'embed': 'embed_i2v'}, inplace=True)
i2v_df['strembed_i2v'] = i2v_df['strembed_i2v'].apply(lambda x: np.array(x.split(), dtype=float))
print(i2v_df)
print(i2v_df.iloc[0, 1])

cols = len(i2v_df.axes[1])
print(f'\nNumber of columns: {cols}\n')
rows = len(i2v_df.axes[0])
print(f'\nNumber of rows: {rows}\n')


#-------------------------------------------------




