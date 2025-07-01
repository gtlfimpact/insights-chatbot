import streamlit as st
import os
import faiss, pickle, numpy as np
import time
from openai import OpenAI, RateLimitError

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

# ─── Safe Chat Completion Function with Retry ───────────────────────
def safe_chat_completion(prompt, retries=3, delay=3):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
        except RateLimitError:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return None

# ─── Chat Input + Process ───────────────────────────────────────────
with st.container():
    prompt = st.chat_input("Ask a question about the grantee impact reports:")

    if prompt:
        with st.spinner("Thinking..."):
            emb_resp = client.embeddings.create(
                input=[prompt],
                model="text-embedding-ada-002"
            )
            xq = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(xq)

            D, I = index.search(xq, 50)
            retrieved = [(docs[i], meta[i]["file"]) for i in I[0]]
            seen = set()
            unique_chunks = []
            for text, source in retrieved:
                if text not in seen:
                    unique_chunks.append((text, source))
                    seen.add(text)

            context_text = "\n\n".join(chunk for chunk, _ in unique_chunks)

            qa_prompt = (
                "You are a helpful assistant reading excerpts from grantee impact reports. "
                "Use the excerpts below to answer the user question in a clear, accurate, and complete way.\n\n"
                f"{context_text}\n\n"
                f"Question: {prompt}\nAnswer:"
            )

            chat_resp = safe_chat_completion(qa_prompt)
            if chat_resp is None:
                answer = "⚠️ We're hitting request limits to the model. Please try again in a few seconds."
            else:
                answer = chat_resp.choices[0].message.content

            st.session_state.chat_history.append({
                "question": prompt,
                "answer": answer,
                "sources": unique_chunks if show_sources else None
            })

# ─── Display History (latest at bottom, above input) ───────────────
for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(f"**Q{i+1}:** {turn['question']}")
    with st.chat_message("assistant"):
        st.markdown(f"**A{i+1}:** {turn['answer']}")
        if turn.get("sources"):
            with st.expander("🔍 View retrieved excerpts"):
                for j, (chunk, source) in enumerate(turn["sources"][:5]):
                    st.markdown(f"**{j+1}. Source: _{source}_**")
                    st.markdown(chunk)
                    st.markdown("---")
