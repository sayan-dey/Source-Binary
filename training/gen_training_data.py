
#script for creating the dict from where we will choose randomly and generating csv file that can be used for training

import os
import csv
import re
import pandas as pd
import argparse
from collections import defaultdict
import json
import random
import numpy as np

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



def gen_vexir_df_old(unst_path, st_path, test_dict=None):
    
    file_dfs = []
    for unst_file, st_file in zip(sorted(os.listdir(unst_path)), sorted(os.listdir(st_path))):
        
        proj = os.path.basename(os.path.dirname(os.path.dirname(unst_path)))
        # if unst_file.split('.')[0] not in test_dict[proj]:
        # if unst_file not in test_dict[proj]:  #considering train set ...will set later
        if True:
            col_names_str = ['addr', 'key', 'fnName', 'strRefs', 'extlibs', 'embed', 'cfg']
            col_names_unstr = ['addr', 'key', 'fnName', 'embed']
            
            unst_df = pd.read_csv(os.path.join(
                unst_path, unst_file), sep='\t', names=col_names_unstr, header=None)
            
            unst_df = unst_df[unst_df["key"].str.contains("NO_FILE_SRC") == False]
            unst_df.reset_index(drop=True, inplace=True)
            unst_df = unst_df[unst_df["fnName"].str.startswith("sub_") == False]
            unst_df.reset_index(drop=True, inplace=True)
            unst_df.addr.astype(str)
            # print('UNST DONE')

            # st_df = pd.DataFrame(st_data, columns=col_names)
            st_df = pd.read_csv(os.path.join(st_path, st_file), sep='\t', names=col_names_str, header=None)
            
            st_df = st_df[st_df['fnName'].str.startswith("sub_") | st_df['fnName'].str.fullmatch("main")]
            st_df.reset_index(drop=True, inplace=True)
            st_df.addr.astype(str)
            
            # print('st_df shape: ', st_df.shape)

            file_df = pd.merge(unst_df, st_df, on='addr', suffixes=('_unst', '_st')) #merging str and unstr df based on fn address
            file_df.key_unst = file_df.key_unst+file_df.fnName_unst
           
                        
            file_df.drop(['fnName_unst', 'embed_unst', 'key_st',
                         'fnName_st'], axis=1, inplace=True)
            
            # column_names = file_df.columns.tolist()
            # print("column names: ", column_names)
            # print(file_df)
            # exit()

            file_dfs.append(file_df)

    return pd.concat(file_dfs) #for converting list to dataframe



def gen_llvm_ir_df_old(llvmir_dir):
    
    columns = ['key', 'embed']

    # Create an empty DataFrame with only column names
    llvmir_df = pd.DataFrame(columns=columns)
    
    for llfile in os.listdir(llvmir_dir):
        llfile_path = os.path.join(llvmir_dir, llfile)
        ll_df = pd.read_csv(llfile_path, sep='\t', header=None, names=[
                                   'key', 'embed'])
        # print(unstripped_df)
        
        # Concatenate along rows (axis=0, default behavior)
        llvmir_df = pd.concat([llvmir_df, ll_df])
    
    
    return llvmir_df
    

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
    
    output_file = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict.json'
    with open(output_file, 'w') as file:
        json.dump(master_dict, file, indent=4)
     

def gen_vexir_df(c_directory, class_dict):
    #for v2v file
    v2v_config = random.choice(list(class_dict['v2v'].keys()))   
    # print(v2v_config)
    filename = random.choice(class_dict['v2v'][v2v_config])
    filename_without_extension = os.path.splitext(filename)[0]
    v2v_filename= filename_without_extension + '.data'
    v2v_filepath = os.path.join(c_directory, v2v_config, 'stripped-data-inlined-extcall-rwk50n2-n2-newseedembed', v2v_filename)
    col_names_str = ['addr', 'key', 'fnName', 'strRefs', 'extlibs', 'embed']
    try:
        v2v_st_df = pd.read_csv(v2v_filepath, sep='\t', names=col_names_str, header=None)
    except:
        v2v_st_df = pd.DataFrame(columns=['key_v2v', 'embed_v2v'])
        return v2v_st_df #returning empty dataframe if file not found
        
    v2v_st_df = v2v_st_df[v2v_st_df['fnName'] == 'main']
    v2v_st_df.reset_index(drop=True, inplace=True) # Reset the index to start from 0
    v2v_st_df.drop(['addr', 'fnName', 'strRefs', 'extlibs'], axis=1, inplace=True)
    v2v_st_df['key'] = c_directory
    v2v_st_df.rename(columns={'key': 'key_v2v', 'embed': 'embed_v2v'}, inplace=True)
    
    return v2v_st_df


def gen_llvm_ir_df(c_directory, class_dict):
    #for i2v file
    i2v_config = random.choice(list(class_dict['i2v'].keys()))   
    # print(i2v_config)
    filename = random.choice(class_dict['i2v'][i2v_config])
    filename_without_extension = os.path.splitext(filename)[0]
    i2v_filename= filename_without_extension + '.txt'
    i2v_filepath = os.path.join(c_directory, i2v_config, 'ir2vec-data-sym-rwk20n2', i2v_filename)
    try:
        i2v_df = pd.read_csv(i2v_filepath, sep='\t', header=None)
    except:
        i2v_df = pd.DataFrame(columns=['key_i2v', 'embed_i2v'])
        return i2v_df #returning empty dataframe if file not found
    
    i2v_df = i2v_df.drop(columns=[301]) #this col contains null val
    merged_column = i2v_df.iloc[:, 1:301].values.flatten()
    i2v_embed = np.array(merged_column) # Convert the merged column to a NumPy array
    i2v_df = i2v_df.drop(i2v_df.columns[1:301], axis=1) # Drop the original columns
    i2v_df['embed'] = str(i2v_embed) 
    i2v_df = i2v_df.rename(columns={i2v_df.columns[0]: 'key'}) # Rename the column at the 0th index
    # i2v_df.drop(['key'], axis=1, inplace=True)
    i2v_df['key'] = c_directory
    i2v_df.rename(columns={'key': 'key_i2v', 'embed': 'embed_i2v'}, inplace=True)
    
    return i2v_df
    

def generate_train_data_csv(root):

    with open('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict.json', 'r') as json_file:
        master_dict = json.load(json_file)
    
    columns = ['key_v2v','embed_v2v','key_i2v','embed_i2v','label']
    
    # Create an empty DataFrame with only column names
    data_df = pd.DataFrame(columns=columns)
    
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory) and c_directory in data_dict:
                class_dict = master_dict[c_directory]

                # print(class_dict)

                # ----------creating similar pairs------
                for _ in range(0,50):
                    
                    v2v_st_df = gen_vexir_df(c_directory, class_dict)
                    if v2v_st_df.empty:
                        print("The v2v_st_df DataFrame is empty")
                        continue
                    
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
                for _ in range(0,50):
                    
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
    
    data_df.to_csv('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_training_data.csv', index=False)                
                    
                    

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
    
    
# Specify the path to your JSON file containing required files to consider
json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_filedict.json'
# Read the JSON file and load its contents as a dictionary
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)

#for creating the dict from where we will choose randomly
# generate_master_dict(master_directory)

# Call the function to generate training data
generate_train_data_csv(master_directory)

#------------------------------------------------------------------------------------------------------



