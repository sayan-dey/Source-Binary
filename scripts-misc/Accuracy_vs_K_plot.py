import matplotlib.pyplot as plt

# Data
Num_Neighbors = [10, 25, 50, 100]

# Accuracy1 = [47.09, 70.89, 86.62, 93.02] #with OTA
# Accuracy1 = [44.36, 66.50, 82.19, 88.15] #with OTA

# Accuracy1 = [41.33, 61.32, 79.63, 90.26] #without OTA (with Strings and Libs)
Accuracy1 = [45.03, 63.50, 79.74, 92.75] #without OTA (with Strings and Libs)

# Accuracy2 = [33.16, 55.88, 74.83, 86.95] # without Strings and Libs
Accuracy2 = [31.84, 58.80, 71.92, 80.39] # without Strings and Libs

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(Num_Neighbors, Accuracy1, marker='s', linestyle='-', color='blue', label='Top-K Accuracy % (with Strings and Libraries)')
ax.plot(Num_Neighbors, Accuracy2, marker='o', linestyle='-', color='green', label='Top-K Accuracy % (without Strings and Libraries)')

# Add titles and labels
ax.set_title('Source to Binary Matching', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Neighbors (K)', fontsize=14)
ax.set_ylabel('Top-K Accuracy', fontsize=14)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend
ax.legend(fontsize=12)

# Customize the plot style
plt.style.use('seaborn-darkgrid')

# Save the plot to a file
figfile = '/Pramana/VexIR2Vec/Source_Binary/Accuracy_vs_K_plot_Src_Bin_match_w_wo_StrLib.png'
plt.savefig(figfile)

# Show the plot
plt.show()
