import streamlit as st
import os
from datetime import datetime
import uuid

import pypdf
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# =====================================================
# API KEYS
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("API keys not found. Please set them in environment variables or Streamlit Secrets.")
    st.stop()

# =====================================================
# CLIENTS
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
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical Chatbot")

# =====================================================
# TEXT SPLITTER (NO LANGCHAIN)
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
def store_document_in_pinecone(text_chunks):
    vectors = []

    for chunk in text_chunks:
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
# RETRIEVE FROM PINECONE
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
    relevant_text = retrieve_from_pinecone(query)

    prompt = f"""
You are a medical assistant.
Answer ONLY using the context below.
If not found, say "Information not available in the document."

Context:
{relevant_text}

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
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        text_chunks = split_text(extracted_text)
        store_document_in_pinecone(text_chunks)

    st.success("✅ Document uploaded and processed successfully!")

# =====================================================
# CHAT HISTORY
# =====================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# USER INPUT
# =====================================================
user_input = st.chat_input("Type your message...")

if user_input:
    timestamp = datetime.now().strftime("%H:%M")

    st.session_state["messages"].append(
        {"role": "user", "content": user_input, "time": timestamp}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chatbot_response = get_chatbot_response(user_input)
            st.markdown(chatbot_response)

    st.session_state["messages"].append(
        {"role": "assistant", "content": chatbot_response, "time": timestamp}
    )
