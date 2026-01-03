import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
import os
import pypdf
from datetime import datetime

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize Pinecone instance
pc = os.getenv("PINECONE_API_KEY")
# Check if the index exists, create if not
if "medii" not in pc.list_indexes().names():
    pc.create_index(
        name="medii",
        dimension=1536,  # Ensure this matches the embedding model
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Use correct region
    )

# Connect to the index
index = pc.Index("medii")

openai.api_key = OPENAI_API_KEY

# Page configuration
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# Function to extract text from PDF using pypdf
def extract_text_from_pdf(file):
    reader = pypdf.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    chunks = split_text(text)
    return chunks

# Function to store document text in Pinecone
def store_document_in_pinecone(text_chunks):
    for i, chunk in enumerate(text_chunks):
        vector = openai.Embedding.create(input=[chunk], model="text-embedding-ada-002")["data"][0]["embedding"]
        index.upsert([(f"doc_chunk_{i}", vector, {"text": chunk})])

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    text_chunks = split_text_into_chunks(extracted_text)
    store_document_in_pinecone(text_chunks)
    st.success("Document uploaded and processed successfully!")

# Function to retrieve relevant text from Pinecone
def retrieve_from_pinecone(query):
    query_vector = openai.Embedding.create(input=[query], model="text-embedding-ada-002")["data"][0]["embedding"]
    results = index.query(query_vector, top_k=3, include_metadata=True)
    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]
    return "\n".join(retrieved_texts)

# Function to get chatbot response
def get_chatbot_response(query):
    relevant_text = retrieve_from_pinecone(query)
    prompt = f"Relevant context: {relevant_text}\nUser question: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Display chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

chat_container = st.container()
with chat_container:
    for msg in st.session_state["messages"]:
        st.markdown(f'<div class="chat-container"><div class="chat-message">{msg["text"]}</div><small>{msg["time"]}</small></div>', unsafe_allow_html=True)

# User input
user_input = st.text_input("Type your message...", key="user_input")
if user_input:
    timestamp = datetime.now().strftime("%H:%M")
    chatbot_response = get_chatbot_response(user_input)
    st.session_state["messages"].append({"text": user_input, "time": timestamp})
    st.session_state["messages"].append({"text": chatbot_response, "time": timestamp})

