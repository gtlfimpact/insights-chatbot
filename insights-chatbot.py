# Deployment-Ready Streamlit App with Auto-Updating from Google Drive

# 1. Install dependencies
#!pip install --quiet openai faiss-cpu streamlit PyPDF2 python-docx google-api-python-client google-auth-httplib2 google-auth-oauthlib

# 2. Authenticate Google Drive (for scheduled refresh, use service account in production)
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io, os, openai, faiss, pickle, time
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
import numpy as np

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

# Connect to google drive 
from google.colab import drive
import os

if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')

# 3. Google Drive folder access
FOLDER_IDS = {
    'reports': '1KtimzlEqcEllzHALi7d8lIqktL1ny-AD',
    'transcripts': '1MAxd8tJleY2BErIzoiHgUxtnQVbHnP8S'
}

# Replace with your service account file
SERVICE_ACCOUNT_FILE = '/content/drive/MyDrive/Colab Notebooks/insights-chatbot-b0f1e34c297d.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=creds)

# 4. Helper: download and read files
def list_files(folder_id):
    all_files = []
    page_token = None
    while True:
        resp = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
            pageToken=page_token
        ).execute()
        batch = resp.get('files', [])
        print(f"Found {len(batch)} in {folder_id}: {[f['name'] for f in batch]}")
        all_files.extend(batch)
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    print(f"Total for {folder_id}: {len(all_files)}")
    return all_files

def download_file(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def extract_text(file_name, fh, mime):
    ext = Path(file_name).suffix.lower()
    if mime == 'application/pdf' or ext == '.pdf':
        return ''.join(p.extract_text() or '' for p in PdfReader(fh).pages)
    if mime.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml.document') or ext == '.docx':
        return '\n'.join(p.text for p in Document(fh).paragraphs)
    if mime.startswith('text/'):
        return fh.read().decode('utf-8')
    return ''

# 5. Load and chunk all documents
docs, meta = [], []
for tag, folder_id in FOLDER_IDS.items():
    files = list_files(folder_id)
    if not files:
        raise RuntimeError(f"No files found in shared Drive folder {folder_id}")
    for f in files:
        fh = download_file(f['id'])
        text = extract_text(f['name'], fh, f['mimeType'])
        if not text.strip():
            print(f"Skipped empty: {f['name']}")
            continue
        for chunk in text.split("\n\n"):
            c = chunk.strip()
            if len(c) > 200:
                docs.append(c)
                meta.append({'file': f['name'], 'folder': tag})

print("Total text chunks:", len(docs))
if not docs:
    raise ValueError("No text chunks extracted—verify extraction logic.")

# 6. Embed and store FAISS index
import os
import pickle
import faiss
import numpy as np
from openai import OpenAI

def chunk_text(text, chunk_size=1000, overlap=200):
    tokens = text.split()
    return [
        " ".join(tokens[i : i+chunk_size])
        for i in range(0, len(tokens), chunk_size - overlap)
    ]

# re-chunk
re_docs, re_meta = [], []
for doc, info in zip(docs, meta):
    for sub in chunk_text(doc):
        re_docs.append(sub)
        re_meta.append(info)
docs, meta = re_docs, re_meta

from openai import OpenAI
client = OpenAI()

def get_embeddings(texts, model="text-embedding-ada-002", batch_size=200):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend(d.embedding for d in resp.data)
    return embs

emb_list = get_embeddings(docs)

import numpy as np, faiss
X = np.array(emb_list, dtype="float32")
if X.size == 0:
    raise ValueError("No embeddings to index – check emb_list")
if X.ndim == 1:
    X = X.reshape(1, -1)

faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# ─── then your persistence code follows ───
faiss.write_index(index, "faiss_index.bin")
with open("meta.pkl","wb") as f: pickle.dump(meta, f)
with open("docs.pkl","wb") as f: pickle.dump(docs, f)

# 7. Streamlit App (save as app.py)
%%writefile insights-chatbot.py
import streamlit as st
import openai, faiss, pickle, numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]

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

# 8. requirements.txt
%%writefile requirements.txt
streamlit
openai
faiss-cpu
PyPDF2
python-docx
google-api-python-client
google-auth
google-auth-httplib2
google-auth-oauthlib

# Now upload to GitHub and deploy at streamlit.io/cloud
# Set OPENAI_API_KEY in Streamlit Cloud secrets manager
# Schedule regular rebuilds via GitHub Actions for fresh Drive sync
