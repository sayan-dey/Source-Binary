#Extracting lines_to_extract number of instructions from norm-ll (dir containing normalized ll files)
# dirs of COFO dataset

import os

# Root directory path
root_dir = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset'
# lines_to_extract = 50000
lines_to_extract = float('inf')  # Extract all lines

# Function to extract a total of lines_to_extract non-empty lines from all ll files in norm-ll directories
def extract_lines_collectively(root):
    total_lines_extracted = 0
    collected_lines = []

    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith('norm-ll'):
            for file_name in filenames:
                if file_name.endswith('.ll'):
                    file_path = os.path.join(dirpath, file_name)
                    print(f"File path: {file_path}")

                    # Reading and collecting non-empty lines from ll files
                    with open(file_path, 'r') as file:
                        for line in file:
                            if line.strip() != '' and total_lines_extracted < lines_to_extract:
                                collected_lines.append(line.strip())
                                total_lines_extracted += 1

                    if total_lines_extracted >= lines_to_extract:
                        break

            if total_lines_extracted >= lines_to_extract:
                break

    # Dumping the collected lines into a new file
    output_file = '/Pramana/VexIR2Vec/Source_Binary/Dataset/COFO_norm_ll_insts_full.txt'
    with open(output_file, 'w') as output:
        for line in collected_lines:
            output.write(line + '\n')

    print(f"Total non-empty lines extracted: {total_lines_extracted}")
    print(f"Collected lines dumped to: {output_file}")

# Call the function with the root directory
extract_lines_collectively(root_dir)
