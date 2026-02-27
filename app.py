import streamlit as st
from utils import preprocess_image, extract_text_easyocr
from rag_pipeline import build_vector_db, retrieve_chunks,answer_ollama, chunk_text

st.set_page_config(page_title="Document based Q/A using RAG")
st.title(" Document based Q/A using RAG ")

uploaded_file = st.file_uploader("Upload a text image", type=["png", "jpg", "jpeg"])
question = st.text_input("Ask a question from the image text")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img, processed = preprocess_image(uploaded_file.read())
    text = extract_text_easyocr(img)

    st.subheader("Extracted Text from Document")
    st.text_area("", text, height=200)

    if text and question:
        chunks = chunk_text(text)
        embed_model, index = build_vector_db(chunks)
        retrieved = retrieve_chunks(question, chunks, embed_model, index)
        answer = answer_ollama(retrieved, question)

    st.subheader("Retrieved Context from Document")
    for ch in retrieved:
        st.info(ch)

    st.subheader("AI's Answer")
    st.success(answer)