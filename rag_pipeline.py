import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

MODEL_NAME = "llama3"
TOP_K = 3


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def build_vector_db(chunks):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return embed_model, index

def retrieve_chunks(question, chunks, embed_model, index, top_k=TOP_K):
    # Embed the question
    q_vector = embed_model.encode([question])

    # FAISS search
    D, I = index.search(q_vector, top_k)

    # Safely retrieve chunks
    retrieved = []
    if len(I) > 0 and len(chunks) > 0:
        for i in I[0]:
            if i < len(chunks) and i != -1:  
                retrieved.append(chunks[i])

    return retrieved


def answer_ollama(context_chunks, question):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the context below.

Context:
{context}

Question: {question}
Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']