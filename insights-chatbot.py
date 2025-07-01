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

    # 2) Retrieve top 5 chunks
    D, I = index.search(xq, 5)
    context = "\n\n".join(docs[i] for i in I[0])

    # 3) Ask GPT-4 with those contexts
    prompt = (
        "Use the excerpts below to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    chat_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    # 4) Show the answer
    st.write(chat_resp.choices[0].message.content)
