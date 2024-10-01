# Script for parallely generating angr-vex unstripped data files from stripped data files in COFO-Dataset 
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

unstr_file_path1 = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/1003-A/submissions/C/llvm14-Os/unstripped-data-inlined-extcall-rw-n0-newseedembed/39898765.data'
unstr_file_path2 = '/Pramana/VexIR2Vec/Source_Binary/test/39898765-unstr.data'

col_names_unstr = ['addr', 'key', 'fnName', 'embed_unst']
unstr_df1 = pd.read_csv(unstr_file_path1, sep='\t', names=col_names_unstr, header=None)
unstr_df2 = pd.read_csv(unstr_file_path2, sep='\t', names=col_names_unstr, header=None)

print(unstr_df1)
print(unstr_df2)

exit()


str_file_path ='/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/1003-A/submissions/C/llvm14-Os/stripped-data-inlined-extcall-rwk50n2-n2-newseedembed/39898765.data'
unstr_file_path = '/Pramana/VexIR2Vec/Source_Binary/test/39898765-unstr.data'


col_names_str = ['addr', 'key', 'fnName', 'strRefs', 'extlibs', 'embed']
str_df = pd.read_csv(str_file_path, sep='\t', names=col_names_str, header=None)
str_row_index = str_df[str_df['fnName'] == 'main'].index
str_addr = str_df.loc[str_row_index, 'addr']

# print(type(str(str_addr.values[0])))
# exit()
col_names_unstr = ['addr', 'key', 'fnName', 'embed_unst']
unstr_df = pd.read_csv(unstr_file_path, sep='\t', names=col_names_unstr, header=None)
unstr_row_index = unstr_df[unstr_df['fnName'] == 'main'].index
unstr_df.loc[unstr_row_index, 'addr'] = str(str_addr.values[0])

unstr_df.to_csv(unstr_file_path, sep='\t', header=False, index=False)






# file_paths.append(file_path)
# data_file_paths.append(os.path.join(new_directory_path, os.path.splitext(os.path.basename(file_path))[0] + ".data"))
            
                    
