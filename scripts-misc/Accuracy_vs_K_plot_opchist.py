import matplotlib.pyplot as plt

# Data
Num_Neighbors = [10, 25, 50, 100]

Accuracy1 = [47.09, 70.89, 86.62, 93.02] #with OTA
# Accuracy1 = [44.36, 66.50, 82.19, 88.15] #with OTA

Accuracy2 = [14.89, 29.66, 36.78, 47.46] #for opc-hist
# Accuracy2 = [30.96, 52.77, 70.37, 84.24] #for opc-hist

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(Num_Neighbors, Accuracy1, marker='s', linestyle='-', color='blue', label='Top-K Accuracy % (Our approach)')
ax.plot(Num_Neighbors, Accuracy2, marker='^', linestyle='-', color='coral', label='Top-K Accuracy % (Opcode Histogram)')

# Add titles and labels
ax.set_title('Matching Binary-to-Source: Our approach vs Opcode Histogram', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Neighbors (K)', fontsize=14)
ax.set_ylabel('Top-K Accuracy', fontsize=14)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend
ax.legend(fontsize=14, loc="lower right")

# Customize the plot style
plt.style.use('seaborn-darkgrid')

# Save the plot to a file
figfile = '/Pramana/VexIR2Vec/Source_Binary/Accuracy_vs_K_plot_Bin_Src_OTA_opchist_match.png'
plt.savefig(figfile)

# Show the plot
plt.show()
