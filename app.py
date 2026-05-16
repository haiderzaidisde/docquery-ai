import streamlit as st
import os
from rag_engine import load_and_split_pdf, create_vector_store, get_answer

# Page title
st.title("📄 AI Document Q&A Chatbot")
st.write("Upload a PDF and ask me anything about it!")

# Sidebar - PDF upload
with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file to uploads folder
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process PDF
        with st.spinner("Processing PDF..."):
            chunks = load_and_split_pdf(save_path)
            st.session_state.vector_store = create_vector_store(chunks)

        st.success("✅ PDF processed! Ask me anything.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("Ask a question about your PDF..."):
    if "vector_store" not in st.session_state:
        st.warning("⚠️ Please upload a PDF first!")
    else:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    st.session_state.vector_store,
                    question,
                    st.session_state.chat_history
                )
                st.write(answer)

        # Save to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )