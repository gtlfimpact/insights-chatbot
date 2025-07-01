import streamlit as st
import os
import faiss, pickle, numpy as np
from openai import OpenAI

# ─── Configure OpenAI client ─────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# ─── Load pre-built index + data ────────────────────────────────────
index = faiss.read_index("faiss_index.bin")
meta   = pickle.load(open("meta.pkl", "rb"))
docs   = pickle.load(open("docs.pkl", "rb"))

# ─── Streamlit UI ────────────────────────────────────────────────────
st.title("Grantee Report Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the grantee impact reports:")

if query:
    # 1) Embed the prompt
    emb_resp = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    vector = emb_resp.data[0].embedding
    xq     = np.array(vector, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(xq)

    # 2) Retrieve top 30 chunks and deduplicate
    D, I = index.search(xq, 30)
    chunks = [docs[i] for i in I[0]]
    unique_chunks = list(set(chunks))
    combined = "\n\n".join(unique_chunks)

    # 3) Ask GPT-4 with broader context
    prompt = (
        "Use the excerpts below to answer the question. You may see mentions of different grantee organizations or themes.\n\n"
        f"{combined}\n\n"
        f"Question: {query}\nAnswer:"
    )
    chat_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = chat_resp.choices[0].message.content

    # 4) Store and show
    st.session_state.chat_history.append({"question": query, "answer": answer})
    query = ""  # clear input

# Display all previous Q&A
for i, turn in enumerate(st.session_state.chat_history):
    st.markdown(f"**Q{i+1}:** {turn['question']}")
    st.markdown(f"**A{i+1}:** {turn['answer']}")
    st.markdown("---")
