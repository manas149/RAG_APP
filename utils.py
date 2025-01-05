import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Download NLTK resources
# nltk.download("punkt")
    
# Chunking the document
def chunk_text(text, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += token_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def create_faiss_index(chunks, embedding_model_name="all-MiniLM-L6-v2"):
    """Create and return a FAISS index with embeddings."""
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    absolute_path = os.path.abspath("RAG_PROJECT")
    os.path.join(absolute_path, "faiss_index")
    faiss.write_index(index, "faiss_index")
    return index, model
