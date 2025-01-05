#For plotting cumulative count of funcs vs number of basic blocks

import matplotlib.pyplot as plt
import json
import numpy as np

# File paths for the JSON files
json_file_path1 = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_bb_counts_ir2vec_llvm-14_O0.json'
json_file_path2 = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_bb_counts_vexir2vec_clang-12_O0.json'

# Read the nested dictionaries from the JSON files
with open(json_file_path1, 'r') as json_file1:
    data1 = json.load(json_file1)

with open(json_file_path2, 'r') as json_file2:
    data2 = json.load(json_file2)

# Convert JSON keys and values to sorted lists
keys1 = sorted(map(int, data1.keys()))  # Sorted basic block counts
values1 = [data1[str(k)] for k in keys1]

keys2 = sorted(map(int, data2.keys()))  # Sorted basic block counts
values2 = [data2[str(k)] for k in keys2]

# Calculate cumulative sum of values
cumulative_counts1 = np.cumsum(values1)
cumulative_counts2 = np.cumsum(values2)

# Determine the maximum x-axis value
max_key = max(max(keys1), max(keys2))

# Generate x-ticks dynamically
below_50_ticks = list(range(0, 51, 10))
above_50_ticks = list(range(50, max_key + 1, 20))
x_ticks = below_50_ticks + above_50_ticks

# Plot the cumulative counts
plt.figure(figsize=(12, 8))

# Plot for the first dataset
plt.plot(keys1, cumulative_counts1, marker='p', linestyle='-', color='blue', label='For LLVM IR')

# Plot for the second dataset
plt.plot(keys2, cumulative_counts2, marker='h', linestyle='-', color='green', label='For VexIR')

# Highlight the range < 50 for both datasets
below_50_keys1 = [k for k in keys1 if k < 50]
below_50_counts1 = [cumulative_counts1[keys1.index(k)] for k in below_50_keys1]
plt.fill_between(below_50_keys1, below_50_counts1, color='yellow', alpha=0.3)

below_50_keys2 = [k for k in keys2 if k < 50]
below_50_counts2 = [cumulative_counts2[keys2.index(k)] for k in below_50_keys2]
plt.fill_between(below_50_keys2, below_50_counts2, color='yellow', alpha=0.3)

# Set x-ticks and customize the plot
plt.xticks(x_ticks)
plt.xlabel('Basic Blocks', fontsize=14)
plt.ylabel('Cumulative Count of Functions', fontsize=14)
plt.title('Comparison of Basic Blocks vs Cumulative Count of Functions', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc="lower right")

# Save the plot for slides
plt.savefig('/Pramana/VexIR2Vec/Source_Binary/basic_blocks_vs_functions_v2v_i2v_fixed.png', dpi=300)

# Show the plot
plt.show()


exit()







import matplotlib.pyplot as plt
import json
import numpy as np

# json_file_path ='/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_bb_counts_ir2vec_llvm-14_O0.json'
json_file_path ='/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_bb_counts_vexir2vec_clang-12_O0.json'

# Read the nested dictionary from the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)


# Convert JSON keys and values to sorted lists
keys = sorted(map(int, data.keys()))  # Sorted basic block counts
values = [data[str(k)] for k in keys]

# Calculate cumulative sum of values
cumulative_counts = np.cumsum(values)

# Plot the cumulative count
plt.figure(figsize=(12, 8))
# plt.plot(keys, cumulative_counts, marker='p', linestyle='-', color='blue', label='Cumulative Count')
plt.plot(keys, cumulative_counts, marker='h', linestyle='-', color='green', label='Cumulative Count')


# Highlight the range < 50
below_50_keys = [k for k in keys if k < 50]
below_50_counts = [cumulative_counts[keys.index(k)] for k in below_50_keys]
plt.fill_between(below_50_keys, below_50_counts, color='yellow', alpha=0.3)

# Custom x-ticks for < 50
plt.xticks(list(range(0, 51, 10)) + [k for k in keys if k >= 50][::20])



# Customize the plot
plt.xlabel('Basic Blocks', fontsize=14)
plt.ylabel('Cumulative Count of Functions', fontsize=14)
plt.title('For Vex IR', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc="lower right")

# Save the plot for slides
plt.savefig('/Pramana/VexIR2Vec/Source_Binary/basic_blocks_vs_functions_cum_v2v.png', dpi=300)

# Show the plot
plt.show()


'''

# Convert JSON keys and values to lists
keys = list(map(int, data.keys()))
values = list(data.values())

# Plot the data
plt.figure(figsize=(12, 8))
# plt.plot(keys, values, marker='p', linestyle='-', color='blue', label='Functions')
plt.plot(keys, values, marker='h', linestyle='-', color='green', label='Functions')


# Setting log scale if needed
plt.xscale('log')  # Log scale for x-axis (optional, remove if not needed)
plt.yscale('log')  # Log scale for y-axis (optional, remove if not needed)

# Add labels and title
plt.xlabel('Basic Blocks', fontsize=14)
plt.ylabel('Function Count', fontsize=14)
plt.title('For Vex IR', fontsize=16)

# Adding a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add a legend
plt.legend(fontsize=12)

# Save the plot for slides
plt.savefig('/Pramana/VexIR2Vec/Source_Binary/basic_blocks_vs_functions_v2v.png', dpi=300)

# Show the plot
plt.show()
'''