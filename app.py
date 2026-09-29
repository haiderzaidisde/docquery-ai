from langchain_core.documents import Document
import streamlit as st
import time
import PyPDF2
import re
import pytesseract
import pypdfium2 as pdfium

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS


# Tesseract OCR engine location on Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def _secret(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LumeDocs",
    page_icon="📄",
    layout="wide"
)

google_key = _secret("GOOGLE_API_KEY")
groq_key = _secret("GROQ_API_KEY")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Workspace")
    st.markdown("---")

    st.subheader("🔐 System Status")
    if google_key and groq_key:
        st.success("LumeDocs is ready to process your documents.")
    else:
        st.error(
            "Missing API keys. Add GOOGLE_API_KEY and GROQ_API_KEY to "
            "`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`)."
        )

    st.markdown("---")

    if google_key and groq_key:
        st.subheader("📄 Add a Document")
        uploaded_file = st.file_uploader("Upload your document", type="pdf")

        if uploaded_file is not None:
            # FIX #03 — File size check
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"📦 File size: {file_size_mb:.2f} MB")

            if file_size_mb > 10:
                st.error(f"⚠️ File too large ({file_size_mb:.1f} MB). Please upload a PDF under 10MB.")
            else:
                if st.button("🚀 Process PDF"):
                    progress = st.progress(0)
                    status = st.empty()

                    # FIX #05 — Step 1: Extract text page by page
                    status.info("📖 Step 1/4 — Extracting text from PDF...")
                    progress.progress(10)

                    pages = []

                    try:
                        reader = PyPDF2.PdfReader(uploaded_file)
                        total_pages = len(reader.pages)

                        # Open the PDF for rendering pages into images when OCR is needed.
                        pdf_document = pdfium.PdfDocument(uploaded_file.getvalue())

                        # OCR threshold: pages with fewer than 60 letters or digits
                        # are treated as possible scanned pages.
                        OCR_TEXT_THRESHOLD = 60

                        for page_number, page in enumerate(reader.pages, start=1):
                            try:
                                # First, try extracting normal PDF text.
                                extracted = page.extract_text() or ""
                            except Exception:
                                extracted = ""

                            extracted = extracted.strip()

                            # Count meaningful letters and digits in the extracted text.
                            meaningful_chars = len(re.findall(r"[A-Za-z0-9]", extracted))

                            method = "text"

                            # If there is very little text, try OCR.
                            if meaningful_chars < OCR_TEXT_THRESHOLD:
                                try:
                                    status.info(
                                        f"🔎 OCR: Checking page {page_number} of {total_pages}..."
                                    )

                                    # Render this PDF page as an image at approximately 200 DPI.
                                    pdf_page = pdf_document[page_number - 1]
                                    image = pdf_page.render(scale=200 / 72).to_pil()

                                    # Extract text from the image using Tesseract.
                                    ocr_text = pytesseract.image_to_string(
                                        image,
                                        lang="eng"
                                    ).strip()

                                    # Use OCR text if it contains meaningful content.
                                    if len(re.findall(r"[A-Za-z0-9]", ocr_text)) > meaningful_chars:
                                        extracted = ocr_text
                                        method = "ocr"

                                except Exception as ocr_error:
                                    st.warning(
                                        f"OCR could not process page {page_number}: {ocr_error}"
                                    )

                            pages.append({
                                "page_number": page_number,
                                "text": extracted,
                                "method": method,
                            })

                            # Update progress as each page is processed.
                            progress.progress(
                                10 + int(15 * page_number / total_pages)
                            )

                    except Exception as e:
                        progress.empty()
                        status.empty()
                        st.error(
                            f"Could not read this PDF. The file may be corrupt, "
                            f"encrypted, or unreadable. Details: {e}"
                        )

                    else:
                        text = "\n\n".join(page["text"] for page in pages)
                        progress.progress(25)

                        # FIX #05 — Step 2: Chunking
                        status.info("✂️ Step 2/4 — Splitting into chunks...")
                        progress.progress(40)

                        splitter = RecursiveCharacterTextSplitter(
                             chunk_size=1000,
                             chunk_overlap=200
                        )

                        # Create a document for each PDF page
                        page_documents = [
                            Document(
                               page_content=page["text"],
                               metadata={"page_number": page["page_number"]}
                            )
                            for page in pages
                            if page["text"].strip()
                        ]

                        # Split pages into chunks while preserving page numbers
                        chunk_documents = splitter.split_documents(page_documents)

                        chunks = [doc.page_content for doc in chunk_documents]
                        progress.progress(55)

                        if not chunks:
                           progress.empty()
                           status.empty()
                           st.error(
                               "No extractable text was found in this PDF. "
                                "The PDF may be blank, unreadable, or contain images "
                                "that OCR could not recognize."
                            )
                        else:
                                                        # Step 3: Generate embeddings
                            status.info("🧠 Step 3/4 — Generating embeddings...")
                            progress.progress(60)

                            try:
                                embeddings = GoogleGenerativeAIEmbeddings(
                                    model="models/gemini-embedding-001",
                                    google_api_key=google_key
                                )

                                progress.progress(80)

                                # Step 4: Build the FAISS index
                                status.info("🔍 Step 4/4 — Building search index...")

                                vector_store = FAISS.from_documents(
                                    chunk_documents,
                                    embedding=embeddings
                                )

                                # Save everything after successful processing
                                st.session_state.vector_store = vector_store
                                st.session_state.google_key = google_key
                                st.session_state.groq_key = groq_key
                                st.session_state.document_info = {
                                    "name": uploaded_file.name,
                                    "pages": total_pages,
                                    "size": file_size_mb,
                                    "status": "Ready"
                                }

                                progress.progress(100)
                                status.success("✅ Processing complete!")

                                st.success(
                                    f"✅ Done! Processed {total_pages} pages, "
                                    f"created {len(chunks)} chunks."
                                )

                                with st.expander("📄 Page Extraction Report"):
                                    for page in pages:
                                        method = (
                                            "OCR"
                                            if page["method"] == "ocr"
                                            else "Normal text extraction"
                                        )
                                        st.write(
                                            f"Page {page['page_number']}: {method}"
                                        )

                            except Exception as e:
                                progress.empty()
                                status.empty()

                                st.error(
                                    "❌ Failed to generate embeddings or build the search index. "
                                    "Please check your Google API key, internet connection, "
                                    "and API usage limits."
                                )
                                st.exception(e)

if "document_info" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Document Information")

    st.sidebar.caption(st.session_state.document_info["name"])
    st.sidebar.write(f"**Pages:** {st.session_state.document_info['pages']}")
    st.sidebar.write(f"**Size:** {st.session_state.document_info['size']:.2f} MB")
    st.sidebar.success(f"✅ {st.session_state.document_info['status']}")

    st.sidebar.markdown("---")

st.sidebar.caption("Built by Haidar Zaidi")

# ── INITIALIZE CHAT ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="color: #1F3A5F; margin-bottom: 0;">
        📄 LumeDocs
    </h1>
    <p style="color: #64748B; font-size: 17px; margin-top: 5px;">
        Your Intelligent Document Assistant
    </p>
    """,
    unsafe_allow_html=True
)
if "vector_store" not in st.session_state:
    st.subheader("Welcome to LumeDocs!")
    st.write(
        "Upload a PDF using the sidebar, then ask questions about its contents."
    )
st.markdown("---")

# Clear Chat button
if st.button(" Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Show previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # FIX #07 — Show sources
        if message["role"] == "assistant" and "sources" in message:
            for i, src in enumerate(message["sources"], 1):
                page_label = src.splitlines()[0] if src.startswith("Page ") else "Page unknown"

                with st.expander(f"📚 Source {i} — {page_label}"):
                    st.markdown(src)

# Chat input
if question := st.chat_input("Ask LumeDocs anything about your PDF..."):
    if "vector_store" not in st.session_state:
        st.warning("⚠️ Please upload a PDF first!")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve top 3 chunks
                    docs = st.session_state.vector_store.similarity_search(question, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    sources = [
                        f"Page {doc.metadata.get('page_number', 'Unknown')}\n\n{doc.page_content}"
                        for doc in docs
                    ]

                    # Build history
                    history_text = ""
                    for msg in st.session_state.chat_history:
                        history_text += f"{msg['role']}: {msg['content']}\n"

                    # Build prompt
                    prompt = f"""You are a careful document question-answering assistant.

Your task is to answer the user's question using ONLY the information
provided in the PDF context below.

Follow these rules strictly:
1. Use only facts explicitly supported by the PDF context.
2. Do not add information from your own knowledge or make assumptions.
3. If the context does not contain the answer, say:
   "I could not find this information in the provided document."
4. Pay close attention to line numbers, section names, codes, dates,
   and amounts. Do not mix up their meanings.
5. When explaining a specific line or code, use the exact meaning
   given in the context.
6. If the question asks for a summary, include only information
   supported by the context.
7. If the context is incomplete or unclear, explain what is missing
   instead of guessing.
8. Chat history is provided for conversation continuity only.
   Do not treat it as evidence or use it to add facts.
9. Keep the answer clear, accurate, and easy to understand.

PDF Context:
{context}

Chat History:
{history_text}

User Question:
{question}

Answer:"""

                    # FIX #09 — Rate limit retry logic
                    max_retries = 3
                    retry_delays = [5, 15, 30]
                    answer = None

                    for attempt in range(max_retries):
                        try:
                            llm = ChatGroq(
                                model="openai/gpt-oss-20b",
                                api_key=st.session_state.groq_key
                            )
                            response = llm.invoke(prompt)
                            answer = response.content
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < max_retries - 1:
                                wait = retry_delays[attempt]
                                st.warning(f"⏳ Rate limit reached. Retrying in {wait} seconds...")
                                time.sleep(wait)
                            else:
                                raise e

                    st.write(answer)

                    # FIX #07 — Show sources
                    for i, src in enumerate(sources, 1):
                        page_label = src.splitlines()[0] if src.startswith("Page ") else "Page unknown"

                        with st.expander(f"📚 Source {i} — {page_label}"):
                             st.markdown(src)

                except Exception as e:
                    answer = f"Error: {str(e)}"
                    sources = []
                    st.error(answer)

        # Save to history
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
