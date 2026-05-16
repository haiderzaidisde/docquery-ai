import os
import PyPDF2
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

# Load .env file
load_dotenv()

def load_and_split_pdf(pdf_path):
    """Read PDF and split into chunks"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    if not text.strip():
        raise ValueError("Could not extract text from this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("PDF text was too short to process.")

    return chunks

def create_vector_store(chunks):
    """Create FAISS vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_answer(vector_store, question, chat_history):
    """Get answer from Groq based on PDF context"""
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant. Answer based on the context below.

Context:
{context}

Chat History:
{history_text}

Question: {question}

Answer:"""

    llm = ChatGroq(
       model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    response = llm.invoke(prompt)
    return response.content