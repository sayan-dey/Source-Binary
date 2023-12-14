#For ploting epoch vs loss plot while finetuning codebert

import matplotlib.pyplot as plt
import os

# Read the contents of the file
logfile = '/Pramana/VexIR2Vec/Source_Binary/trained-mlm-insts50000-e100-bs64-cpu/log.txt'
with open(logfile, 'r') as file:
    lines = file.readlines()

epochs = []
losses = []

# Extract epoch and loss values from each line
for line in lines:

    if line.strip() == '':  # Check for an empty line
        break

    # Remove unwanted characters and extract epoch and loss values
    clean_line = line.strip('{}\n').replace("'", "").replace(",", "").replace("{","").replace("}","")
    values = clean_line.split()
    print("values: ",values,"\n")
    epoch = float(values[5])
    loss = float(values[1])

    # Appending to respective lists
    epochs.append(epoch)
    losses.append(loss)

print("epochs: ",epochs)
print("loss: ",losses)

# Plotting the epoch vs loss
plt.figure(figsize=(14, 10))
plt.plot(epochs, losses, marker='o', linestyle='-')
plt.xlabel('Epoch', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.title('Epoch vs Loss', fontsize=28)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Save the figure as a PDF file
parent = os.path.dirname(logfile)
figfile = os.path.join(parent,'epoch_vs_loss_plot.pdf')
plt.savefig(figfile)

plt.show()
