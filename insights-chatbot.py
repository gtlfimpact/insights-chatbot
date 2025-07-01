import streamlit as st
import os
import faiss, pickle, numpy as np
from openai import OpenAI

# ─── Configure OpenAI client ─────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# ─── Load pre-built index + data ────────────────────────────────────
index = faiss.read_index("faiss_index.bin")
meta  = pickle.load(open("meta.pkl", "rb"))
docs  = pickle.load(open("docs.pkl", "rb"))

# ─── Initialize session state ───────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Streamlit UI Layout ────────────────────────────────────────────
st.title("Grantee Report Chatbot")

with st.sidebar:
    st.markdown("### Settings")
    show_sources = st.checkbox("Show retrieved excerpts", value=False)
    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []

# ─── Chat Input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the grantee impact reports:"):
    # 1. Embed query
    emb_resp = client.embeddings.create(
        input=[prompt],
        model="text-embedding-ada-002"
    )
    xq = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(xq)

    # 2. Retrieve top 50 chunks and deduplicate
    D, I = index.search(xq, 50)
    retrieved = [(docs[i], meta[i]["file"]) for i in I[0]]
    seen = set()
    unique_chunks = []
    for text, source in retrieved:
        if text not in seen:
            unique_chunks.append((text, source))
            seen.add(text)

    context_text = "\n\n".join(chunk for chunk, _ in unique_chunks)

    # 3. Build prompt
    qa_prompt = (
        "You are a helpful assistant reading excerpts from grantee impact reports. "
        "Use the excerpts below to answer the user question in a clear, accurate, and complete way.\n\n"
        f"{context_text}\n\n"
        f"Question: {prompt}\nAnswer:"
    )

    # 4. Generate answer
    chat_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": qa_prompt}]
    )
    answer = chat_resp.choices[0].message.content

    # 5. Save interaction
    st.session_state.chat_history.append({
        "question": prompt,
        "answer": answer,
        "sources": unique_chunks if show_sources else None
    })

# ─── Display History (latest at top) ────────────────────────────────
for i, turn in enumerate(reversed(st.session_state.chat_history)):
    with st.chat_message("user"):
        st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {turn['question']}")
    with st.chat_message("assistant"):
        st.markdown(f"**A{len(st.session_state.chat_history) - i}:** {turn['answer']}")
        if turn.get("sources"):
            with st.expander("🔍 View retrieved excerpts"):
                for j, (chunk, source) in enumerate(turn["sources"][:5]):
                    st.markdown(f"**{j+1}. Source: _{source}_**")
                    st.markdown(chunk)
                    st.markdown("---")
