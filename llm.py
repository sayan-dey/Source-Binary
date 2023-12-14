# Using CodeBERT as of now

# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("feature-extraction", model="microsoft/codebert-base")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Specify the directory where you want to save the model and tokenizer
model_directory = "/Pramana/VexIR2Vec/Source_Binary/codebert-model"
# tokenizer_directory = "/Pramana/VexIR2Vec/Source_Binary/codebert-tokenizer"

# Save the tokenizer
tokenizer.save_pretrained(model_directory)

# Save the model
model.save_pretrained(model_directory)