import streamlit as st
from utils import preprocess_image, extract_text_easyocr
from rag_pipeline import build_vector_db, retrieve_chunks, answer_ollama, chunk_text

st.set_page_config(page_title="Document based Q/A using RAG")
st.title("Document based Q/A using RAG")

# Multiple file uploader
uploaded_files = st.file_uploader(
    "Upload one or more text images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

question = st.text_input("Ask a question from the document text")

if uploaded_files:
    all_docs_chunks = []  # List of chunk lists per document
    doc_names = []        # Document/file names

    # Process each uploaded image
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        img, processed = preprocess_image(uploaded_file.read())
        text = extract_text_easyocr(img)

        st.subheader(f"Extracted Text from {uploaded_file.name}")
        st.text_area("", text, height=200)

        # Chunk text (skip empty text)
        if text.strip():
            chunks = chunk_text(text)
            all_docs_chunks.append(chunks)
            doc_names.append(uploaded_file.name)
        else:
            st.warning(f"No text found in {uploaded_file.name}")

    # Build FAISS index for all chunks combined
    all_chunks_flat = [chunk for doc_chunks in all_docs_chunks for chunk in doc_chunks]

    if all_chunks_flat:
        embed_model, index = build_vector_db(all_chunks_flat)

        retrieved_chunks = []

        # Retrieve top chunks for each document
        for chunks, doc_name in zip(all_docs_chunks, doc_names):
            top_chunks = retrieve_chunks(question, chunks, embed_model, index)
            # Only include valid chunks and label with document name
            for chunk in top_chunks:
                retrieved_chunks.append((doc_name, chunk))

        # Display retrieved chunks
        st.subheader("Retrieved Context from Document(s)")
        if retrieved_chunks:
            for doc_name, chunk in retrieved_chunks:
                st.info(f"{doc_name}: {chunk}")
        else:
            st.warning("No relevant context found for your question.")

        # Generate AI answer using retrieved chunks
        if retrieved_chunks:
            answer = answer_ollama([chunk for _, chunk in retrieved_chunks], question)
            st.subheader("AI's Answer")
            st.success(answer)
        else:
            st.warning("Cannot generate answer because no context was retrieved.")
    else:
        st.error("No text chunks available to build vector database.")