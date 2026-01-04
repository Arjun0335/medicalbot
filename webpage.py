import streamlit as st
import os
import uuid
from datetime import datetime

import pypdf
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# =====================================================
# READ API KEYS
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API Keys not found. Add them in Streamlit Secrets or environment variables.")
    st.stop()

# =====================================================
# CLIENT INITIALIZATION (SAFE)
# =====================================================
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# =====================================================
# PINECONE INDEX
# =====================================================
INDEX_NAME = "medii"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical PDF Chatbot")

# =====================================================
# SIMPLE TEXT SPLITTER (NO LANGCHAIN)
# =====================================================
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# =====================================================
# PDF TEXT EXTRACTION (pypdf)
# =====================================================
def extract_text_from_pdf(file):
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# =====================================================
# STORE DOCUMENT IN PINECONE
# =====================================================
def store_document_in_pinecone(chunks):
    vectors = []

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        vectors.append((
            str(uuid.uuid4()),
            embedding,
            {"text": chunk}
        ))

    index.upsert(vectors)

# =====================================================
# RETRIEVE CONTEXT
# =====================================================
def retrieve_from_pinecone(query):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    return "\n".join(
        match["metadata"]["text"] for match in results["matches"]
    )

# =====================================================
# CHATBOT RESPONSE
# =====================================================
def get_chatbot_response(query):
    context = retrieve_from_pinecone(query)

    prompt = f"""
You are a medical assistant.
Answer ONLY using the context below.
If the answer is not available, say:
"Information not available in the uploaded document."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("📄 Upload a Medical PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        store_document_in_pinecone(chunks)

    st.success("✅ Document uploaded and indexed successfully")

# =====================================================
# CHAT HISTORY
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# USER INPUT
# =====================================================
user_input = st.chat_input("Ask a question from the PDF...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_chatbot_response(user_input)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
