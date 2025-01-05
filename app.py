import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np
from pypdf import PdfReader
from utils import chunk_text, create_faiss_index
from sentence_transformers import SentenceTransformer

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Load documentation and preprocess
document = ""

# Reading the document
reader = PdfReader('data/bhagavad-gita.pdf')
for page in reader.pages:
    document += page.extract_text()

chunks = chunk_text(document)

# Create FAISS index and embedding model
try:
    index = faiss.read_index("faiss_index")
    embedding_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
except:
    index, embedding_model = create_faiss_index(chunks, "distilbert-base-nli-stsb-mean-tokens")

# Load T5 model and tokenizer
# model_name = "t5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Streamlit UI
st.title("Bhagavad Gita Q&A Assistant")
query = st.text_input("Ask a question about Bhagavad Gita:")

if query:
    # Retrieve relevant chunks
    query_embedding = embedding_model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), k=2)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Generate context and response
    context = f"Query: {query}\nRelevant Information: {' '.join(retrieved_chunks)}"
    input_text = f"answer the question based on the information: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=200, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # outputs = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display results
    st.subheader("Answer:")
    st.write(response)

    st.subheader("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        st.write(f"- {chunk}")
