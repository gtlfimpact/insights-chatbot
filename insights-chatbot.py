import streamlit as st
import openai, faiss, pickle, numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load pre-generated assets
index = faiss.read_index("faiss_index.bin")
meta = pickle.load(open("meta.pkl", "rb"))
docs = pickle.load(open("docs.pkl", "rb"))

st.title("Grantee Report Chatbot")
query = st.text_input("Ask a question about the grantee impact reports:")

if query:
    emb = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
    xq = np.array(emb['data'][0]['embedding']).astype('float32').reshape(1, -1)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, 5)

    context = "\n\n".join([docs[i] for i in I[0]])
    prompt = f"Use the excerpts below to answer the question.\n\n{context}\n\nQuestion: {query}\nAnswer:"

    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(resp['choices'][0]['message']['content'])
