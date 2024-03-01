import os
import csv
import re
import pandas as pd
import argparse
from collections import defaultdict

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


def gen_vexir_csv(vexir_dir):
    # Extract function mapping from the .data file
    # print("data_file_path: ", data_file_path)
    
    columns = ['key', 'funcname', 'embed', 'bin_name']
    
    # Create an empty DataFrame with only column names
    vexir_df = pd.DataFrame(columns=columns)
    
    for datafile in os.listdir(vexir_dir):
        datafile_path = os.path.join(vexir_dir, datafile)
        unstripped_df = pd.read_csv(datafile_path, sep='\t', header=None, names=[
                                    'addr', 'key', 'funcname', 'embed'])
        unstripped_df['bin_name'] = datafile.split('.data')[0]
        # print(unstripped_df)
        unstripped_df.drop(['addr'], axis=1, inplace=True)
        
        # Concatenate along rows (axis=0, default behavior)
        vexir_df = pd.concat([vexir_df, unstripped_df])
    
    
    vexir_df['key'] = vexir_df['key'] + vexir_df['funcname']
    vexir_df = vexir_df.drop('funcname', axis=1)
    
    vexir_df.key = vexir_df.key.apply(modify_key) #for removing line no.
    vexir_df.key = vexir_df.key.apply(format_key) #for bringing key to llvm ir key format
    
    vexir_df.drop_duplicates(inplace=True)
    
    # print(vexir_df)
    
    #need to modify the key acc to llvm files
    
    return vexir_df
    

    # function_mapping = {}
    # with open(data_file_path, 'r') as data_file:
    #     for line in data_file:
    #         parts = line.strip().split('\t')
    #         if len(parts) == 4 and parts[3] == 'main':
    #             function_mapping[parts[0]] = parts[2]

    # print(function_mapping)
    # return function_mapping
    
def gen_llvmir_csv(llvmir_dir):
    
    columns = ['key', 'embed', 'bin_name']

    # Create an empty DataFrame with only column names
    llvmir_df = pd.DataFrame(columns=columns)
    
    for llfile in os.listdir(llvmir_dir):
        llfile_path = os.path.join(llvmir_dir, llfile)
        ll_df = pd.read_csv(llfile_path, sep='\t', header=None, names=[
                                   'key', 'embed'])
        ll_df['bin_name'] = llfile.split('.ll')[0]
        # print(unstripped_df)
        
        # Concatenate along rows (axis=0, default behavior)
        llvmir_df = pd.concat([llvmir_df, ll_df])
    
    
    
    llvmir_df.drop_duplicates(inplace=True)
    
    # print(vexir_df)
    
    #need to modify the key acc to llvm files
    
    return llvmir_df
    
    return ""


def generate_ground_truth_csv(directory_path):
    
    for config in configs:
        
        # Traverse the directory structure and generate ground truth CSV file
        for root, dirs, files in os.walk(directory_path):
            if 'submissions' in dirs:
                submissions_path = os.path.join(root, 'submissions', 'C')
                if os.path.exists(submissions_path):
                    
                    config_x = config[0]
                    config_y = config[1]
                    
                    vexir_dir = os.path.join(submissions_path, config_x, 'unstripped-data-w7-s1-inlined-extcall-rw-norm2-inst')
                    llvmir_dir = os.path.join(submissions_path, config_y, 'norm-ll-func')
                    
                    print("vexir_dir: ", vexir_dir)
                    print("llvmir_dir: ", llvmir_dir)
                    
                    vexir_csv = gen_vexir_csv(vexir_dir)
                    # print(vexir_csv)
                    print(vexir_csv.key.iloc[0])
                    # exit()
                    
                    llvmir_csv = gen_llvmir_csv(llvmir_dir)
                    
                    suffix_x = '_' + 'vexir'
                    suffix_y = '_' + 'llvmir'
                    merged_csv = pd.merge(vexir_csv, llvmir_csv, on=['key', 'bin_name'], suffixes=(suffix_x, suffix_y))
                    print(merged_csv.shape)
                    
                    
                    testset = merged_csv
                    testset = testset.reset_index()
                    testset = testset.drop(['index'], axis=1)
                    testset = testset.loc[:, ['key', 'bin_name', 'embed' + suffix_x, 'embed' + suffix_y]]
                    test_correct = testset
                    # proj_name = os.path.basename(dir_path_x)
                    # if not os.path.isdir(out_path):
                    #     os.makedirs(out_path,exist_ok=True)
                    test_correct.rename({'key' : 'key'+suffix_x}, axis=1, inplace=True)
                    test_correct.rename({'bin_name' : 'bin_name'+suffix_x}, axis=1, inplace=True)
                    test_correct['key'+ suffix_y] = test_correct['key'+suffix_x]
                    test_correct['bin_name'+ suffix_y] = test_correct['bin_name'+suffix_x]
                    test_correct = test_correct[['key'+suffix_x, 'key'+suffix_y, 'bin_name' + suffix_x, 'bin_name' + suffix_y, 'embed' + suffix_x, 'embed' + suffix_y]]
                    
                    print(test_correct)
                    
                    test_correct = test_correct.drop('embed_vexir', axis=1) #as these are not req
                    test_correct = test_correct.drop('embed_llvmir', axis=1) #as these are not req
                    out_path=args.output_dir
                    proj_name='COFO'
                    test_correct.to_csv(os.path.join(out_path, proj_name + '-full-' + config_x + '-' + config_y + '.csv'), index=False)
                    #fix above path and file name of gt csv
                    
                    '''
                    # Check if x86-clang-12-O0 directory exists
                    x86_path = os.path.join(submissions_path, 'x86-clang-14-O0')
                    if os.path.exists(x86_path):
                        # Check if unstripped-data-w7-s1-inlined-extcall-rw-norm2-inst directory exists
                        data_path = os.path.join(
                            x86_path, 'unstripped-data-w7-s1-inlined-extcall-rw-norm2-inst')
                        print("data_path: ", data_path)
                        # correctly printing path of a unstripped-data-w7-s1-inlined-extcall-rw-norm2-inst
                        # exit()

                        if os.path.exists(data_path):
                            # Extract function mapping from .data file

                            for data_file in os.listdir(data_path):
                                # getting absolute path of data file
                                data_file_path = os.path.join(data_path, data_file)
                                # data_file_path = os.path.join(data_path, 'some_data_file.data')  # Change this to an actual file name
                                function_mapping = extract_function_mapping(
                                    data_file_path)
                                exit()

                            # Process .ll files in norm-ll directory
                            norm_ll_path = os.path.join(
                                submissions_path, 'norm-ll')
                            # correctly printing norm-ll path
                            # only the name of ll file (eg: 1234.ll)
                            print("norm_ll_path: ", norm_ll_path)
                            exit()

                            if os.path.exists(norm_ll_path):
                                ground_truth = defaultdict(list)
                                for ll_file in os.listdir(norm_ll_path):
                                    print("ll file: ", ll_file)
                                    exit()
                                    ll_file_path = os.path.join(
                                        norm_ll_path, ll_file)
                                    function_name = ll_file.split('.')[0]
                                    if function_name in function_mapping.values():
                                        ground_truth['Function'].append(
                                            function_name)
                                        ground_truth['Data_File'].append(
                                            data_file_path)
                                        ground_truth['LL_File'].append(
                                            ll_file_path)

                                # Write ground truth to CSV
                                # csv_file_path = os.path.join(root, 'ground_truth.csv')
                                # with open(csv_file_path, 'w', newline='') as csvfile:
                                #     writer = csv.writer(csvfile)
                                #     writer.writerow(['Function', 'Data_File', 'LL_File'])
                                #     for i in range(len(ground_truth['Function'])):
                                #         writer.writerow([
                                #             ground_truth['Function'][i],
                                #             ground_truth['Data_File'][i],
                                #             ground_truth['LL_File'][i]
                                #         ])
                                print(f"Ground truth CSV created: {csv_file_path}")
                    '''


parser = argparse.ArgumentParser()
parser.add_argument('-input', '--input_dir', dest='input_dir',
                    help='input dir path', default=None)    
parser.add_argument('-output', '--output_dir', dest='output_dir',
                    help='output_dir_path', default=None)
args = parser.parse_args()


# Specify the master directory
# master_directory = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset-test'
master_directory = args.input_dir

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print(f"Directory '{args.output_dir}' created successfully")
else:
    print(f"Directory '{args.output_dir}' already exists")

# (Vex IR config, LLVM IR config)
configs = [('x86-clang-12-O0', 'x86-clang-14-O0')]


# Call the function to generate ground truth CSV for each directory
generate_ground_truth_csv(master_directory)
