import matplotlib.pyplot as plt

# Data
Num_Neighbors = [10, 25, 50, 100]
# Accuracy1 = [41.33, 61.32, 79.63, 90.26]
Accuracy1 = [45.03, 63.50, 79.74, 92.75]

# Accuracy2 = [36.01, 57.90, 77.42, 89.67]
Accuracy2 = [35.35, 59.62, 76.02, 90.44]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(Num_Neighbors, Accuracy1, marker='s', linestyle='-', color='blue', label='Top-K Accuracy % (Source to Binary)')
ax.plot(Num_Neighbors, Accuracy2, marker='d', linestyle='-', color='red', label='Top-K Accuracy % (VexIR2Vec)')

# Add titles and labels
ax.set_title('Matching Source-to-Binary vs Binary-to-Binary (VexIR2Vec)', fontsize=16, fontweight='bold')
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
figfile = '/Pramana/VexIR2Vec/Source_Binary/Accuracy_vs_K_plot_Src_Bin_V2V_match.png'
plt.savefig(figfile)

# Show the plot
plt.show()
