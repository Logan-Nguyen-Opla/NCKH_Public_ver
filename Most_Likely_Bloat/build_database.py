import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- NEW STATE-OF-THE-ART EMBEDDING MODEL ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# --- DO NOT EDIT BELOW ---

# 1. Load your data
print("Loading data...")
df = pd.read_csv('public_train.csv')

# 2. Create the knowledge base from real news (label == 0)
knowledge_base_df = df[df['label'] == 0].dropna(subset=['post_message'])
documents = knowledge_base_df['post_message'].tolist()
print(f"Created a knowledge base with {len(documents)} articles.")

# 3. Load the new, powerful embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (this may be a large download)...")
# Important: bge-m3 requires enabling trust_remote_code
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

# 4. Create the embeddings
print("Creating embeddings for the knowledge base (this will be slower with the new model)...")
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True, show_progress_bar=True)

# 5. Build the FAISS index
print("Building FAISS index...")
doc_embeddings_np = doc_embeddings.cpu().numpy().astype('float32')
d = doc_embeddings_np.shape[1] 
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings_np) # type: ignore
print("FAISS index built successfully!")

# --- Save the index and documents ---
faiss.write_index(index, "knowledge_base.index")
with open("documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

print("Knowledge base saved successfully.")