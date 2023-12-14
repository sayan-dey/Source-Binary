#File for counting total functions for every project (by counting non-empty lines)
#Eg usage:
#For only non-obfus binaries: python func_counter.py (for total set)
#                             python func_counter.py -test True (for test set)
#For only obfus binaries: python func_counter.py -obfus True -test True

# Modified this script to get instructions of vex ir data in a file 

import os
import re
import argparse

'''
def count_non_empty_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file if line.strip())

def count_non_empty_lines_in_directory(directory_path, proj_dir, test_dict):
    total_non_empty_lines = 0
    
    if args.test: #for test set
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # print(file_path)
                if proj_dir not in test_dict or file_name in test_dict[proj_dir]:
                    total_non_empty_lines += count_non_empty_lines(file_path)
                    
    else:   # for total set  
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # print(file_path)
                # print(file_name)
                # if proj_dir not in test_dict or file_name not in test_dict[proj_dir]:
                total_non_empty_lines += count_non_empty_lines(file_path)
            
    return total_non_empty_lines
'''

def dump_inst(directory_path, proj_dir, test_dict):

    # Define the input and output file paths
    # input_files = ['input_file1.txt', 'input_file2.txt']  # List of input files
    # output_file = 'output_file.txt'  # Output file
    output_file = args.op

    # Initialize an empty list to store the split entries
    split_entries = []

    # Define a regular expression pattern for splitting by <INST> and <END> delimiters
    pattern = r'<INST>|<END>'

    # Compile the regular expression pattern
    split_regex = re.compile(pattern)

                 
    # Loop through each input file
    # for input_file in input_files:
    # for input_file in os.listdir(directory_path):
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            input_file = os.path.join(root, file_name)
            print("processing: ",input_file)
            with open(input_file, 'r') as file:
                for line in file:
                    # Split the line by tabs and get the last entry
                    entries = line.strip().split('\t')
                    last_entry = entries[-1]

                    # Check if the last entry starts with '[' and skip the line if it does
                    if last_entry.startswith('['):
                        continue
                    
                    # Split the last entry using the compiled regex pattern
                    inst_entries = split_regex.split(last_entry)

                    # Replace every | with a space in each split entry
                    inst_entries = [entry.replace('|', ' ') for entry in inst_entries if entry]

                    split_entries.extend(inst_entries)

                
    # Write the split entries to the output file with | replaced by space
    with open(output_file, 'a') as file:
        for entry in split_entries:
            file.write(entry + '\n')


def count_non_empty_lines_in_target_directories(base_directory, prefix, obfus_dirs, proj_dir, test_dict):
    target_directories = []
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)
        if os.path.isdir(item_path):

            target_directory=""
            if not args.obfus: #excluding obfuscated dirs
                if item.startswith(prefix) and item not in obfus_dirs:
                    target_directory = os.path.join(item_path, "stripped-data-w7-s1-inlined-extcall-rw-norm2-inst-delim")
            else: #including only obfuscated dirs
                if item.startswith(prefix) and item in obfus_dirs:
                    target_directory = os.path.join(item_path, "stripped-data-w7-s1-inlined-extcall-rw-norm2-inst-delim")

            if os.path.exists(target_directory):
                target_directories.append(target_directory)


    # total_non_empty_lines = 0
    # for target_directory in target_directories:
    #     total_non_empty_lines += count_non_empty_lines_in_directory(target_directory, proj_dir, test_dict)

    # return total_non_empty_lines

    for target_directory in target_directories:
        dump_inst(target_directory, proj_dir, test_dict)


parser = argparse.ArgumentParser()
parser.add_argument('-obfus', '--use_obfus',  dest='obfus', type=bool, default=False, help='Use obfus bins only if set')
parser.add_argument('-test', '--use_test',  dest='test', type=bool, default=False, help='Use test bins only if set')
parser.add_argument('-op', '--out_file',  dest='op', type=str, help='Output file where instructions should be dumped')


args = parser.parse_args()

parent_dir = "/Pramana/VexIR2Vec/Binaries-notest"
# arch_list = ["x86-data-all", "arm-data-all"]
arch_list = ["x86-data-all"]

# proj_list = ["binutils", "coreutils", "diffutils", "findutils", "openssl"]
# proj_list = ["curl", "lua", "putty", "gzip"]
proj_list = ["findutils"]

test_dict = {}
test_dict['findutils'] = ['xargs.data']

test_dict['diffutils'] = ['cmp.data', 'sdiff.data']

test_dict['binutils'] = ['cxxfilt.data', 'addr2line.data',
                            'nm-new.data', 'gprof.data', 'objcopy.data']

test_dict['coreutils'] = ['who.data', 'stat.data', 'tee.data', 'sha256sum.data', 'sha384sum.data',
                            'sha224sum.data', 'base32.data', 'sha512sum.data', 'unexpand.data', 'expand.data',
                            'base64.data', 'chroot.data', 'env.data', 'sha1sum.data', 'uniq.data',
                            'readlink.data', 'fmt.data', 'stty.data', 'cksum.data', 'head.data', 'realpath.data',
                            'uptime.data', 'wc.data', 'b2sum.data', 'tr.data', 'join.data', 'numfmt.data',
                            'factor.data', 'split.data', 'dd.data', 'rm.data', 'shred.data', 'touch.data']

test_dict['openssl'] = ['crltest.data', 'v3nametest.data', 'cipherlist_test.data', 'tls13ccstest.data', 'ssl_test.data', 
                        'aesgcmtest.data', 'sha_test.data', 'provider_pkey_test.data', 'memleaktest.data', 'cmsapitest.data',
                        'cipherbytes_test.data', 'd2i_test.data', 'pkcs12_format_test.data', 'bntest.data', 
                        'x509_dup_cert_test.data', 'ssl_test_ctx_test.data', 'evp_fetch_prov_test.data', 'dtlstest.data', 
                        'params_api_test.data', 'secmemtest.data']

if args.obfus:
    print("\nFor Obfuscated binaries.....\n\n")
else:
    print("\nFor non-obfuscated binaries....\n\n")

for arch_dir in arch_list:
    for proj_dir in proj_list:
        base_directory = os.path.join(parent_dir, arch_dir, proj_dir)
        # Eg base_directory = "/Pramana/VexIR2Vec/Binaries-notest/x86-data-all/findutils"
        obfus_dirs = ["x86-clang-hybrid-O3", "x86-clang-sub-O3", "x86-clang-bcf-O3", "x86-clang-sub-O3"]
        lis = re.split('-',arch_dir)
        arch = lis[0] #x86 or arch
        
        count_non_empty_lines_in_target_directories(base_directory, lis[0]+"-"+"clang", obfus_dirs, proj_dir, test_dict)
        # count_non_empty_lines_in_target_directories(base_directory, lis[0]+"-"+"gcc", obfus_dirs, proj_dir, test_dict)

        
        print()


