import faiss

# Initialize FAISS index
index = faiss.IndexFlatL2(384)  # Dimensionality must match embedding model

def retrieve_documents(query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return indices  # Return document indices