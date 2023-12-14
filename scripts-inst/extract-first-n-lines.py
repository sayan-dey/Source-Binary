def copy_lines(input_file, output_file, num_lines=50000):
    with open(input_file, 'r', encoding='utf-8') as in_file:
        lines = in_file.readlines()[:num_lines]

        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.writelines(lines)

# Replace 'input_file.txt' with the path to your original file
input_file_path = '/Pramana/VexIR2Vec/Source_Binary/Dataset/concat-vex-ll-ir-50000.txt'

# Replace 'output_file.txt' with the desired output file name
output_file_path = '/Pramana/VexIR2Vec/Source_Binary/Dataset/vex-ir-50000.txt'

# Copy the first 50,000 lines to the output file
copy_lines(input_file_path, output_file_path)