
#number of each of llvm ir and vexir insts to be dumped
lines_to_extract = 5000


# Function to extract first lines_to_extract lines from each of vexir_file and llvmir file and dump into a new file
def extract_lines_from_files():
    vexir_file = '/Pramana/VexIR2Vec/Source_Binary/Dataset/vexir-findutils-x86-clang-insts.txt'
    llvmir_file = '/Pramana/VexIR2Vec/Source_Binary/Dataset/COFO_norm_C_ll_insts_full.txt'
    files = [vexir_file, llvmir_file]
    # files = [vexir_file]
    # files = [llvmir_file]

    extracted_lines = []

    for file_name in files:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            #for taking first lines_to_extract insts (used for training data gen)
            # extracted_lines.extend(lines[:lines_to_extract])
            #for taking last lines_to_extract insts (used for validation (evaluation) data gen)
            extracted_lines.extend(lines[-lines_to_extract:])


    # Dumping the collected lines into a new file
    output_file = '/Pramana/VexIR2Vec/Source_Binary/Dataset/vex-ll-ir-last5000-eval.txt'
    with open(output_file, 'w') as output:
        for line in extracted_lines:
            output.write(line)

    print(f"Extracted insts from file(s) dumped to: {output_file}")

# Call the function
extract_lines_from_files()
