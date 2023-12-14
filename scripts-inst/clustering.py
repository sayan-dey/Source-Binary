import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Path to the file containing token embeddings
embedding_file_path = "/Pramana/VexIR2Vec/Source_Binary/Dataset/tokens/concat-vex-ll-ir-50000_token_emb.txt"

# Load token embeddings from the file
tokens = []
embeddings = []

with open(embedding_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        data = line.strip().split(' ')
        token = data[0]
        # print("token: ", token)
        token_embedding = list(map(float, data[1:]))
        # print("token embedding: ", token_embedding)
        # exit()
        
        tokens.append(token)
        embeddings.append(token_embedding)



# Convert embeddings to numpy array
embeddings_array = np.array(embeddings)
# print(len(averaged_embeddings.values()))

# Apply t-SNE to reduce the dimensionality of embeddings to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Plot the t-SNE embeddings
plt.figure(figsize=(10, 10))
for i in range(len(tokens)):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], tokens[i], fontsize=8)
plt.title('t-SNE Plot of Token Embeddings', fontsize=18)
plt.xlabel('t-SNE Dimension 1', fontsize=16)
plt.ylabel('t-SNE Dimension 2', fontsize=16)

# Save the plot to a file (change 'plot_filename.png' to your desired filename and format)
plot_filename = "/Pramana/VexIR2Vec/Source_Binary/Dataset/tokens/concat-vex-ll-ir-50000_token_emb-tsne.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()


