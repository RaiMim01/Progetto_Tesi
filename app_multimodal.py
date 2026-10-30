# ==========================================
# PATCH PER PYTHON 3.8 SULLA JETSON NANO
# Deve stare in cima allo script!
# ==========================================
import typing
if not hasattr(typing.ForwardRef, '__annotations__'):
    typing.ForwardRef.__annotations__ = property(lambda self: {})

from transformers.image_processing_utils import BaseImageProcessor
_orig_image_call = BaseImageProcessor.__call__
def _patched_image_call(self, images=None, **kwargs):
    for text_param in ["padding", "max_length", "stride", "truncation", "pad_to_multiple_of"]:
        kwargs.pop(text_param, None)
    return _orig_image_call(self, images=images, **kwargs)
BaseImageProcessor.__call__ = _patched_image_call

# ==========================================
# IMPORT LIBRERIE
# ==========================================
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import json
import time
import base64
import urllib.request
import urllib.error
import fitz  # PyMuPDF
import torch
import chromadb
from chromadb.config import Settings
import streamlit as st
from transformers import AutoModel
from dotenv import load_dotenv

# Configurazione percorsi
DATA_PATH = "documents"
CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"

# ==========================================
# 1. CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(
    page_title="Il mio Assistente Universitario",
    page_icon="👩‍🎓",
    layout="wide"
)

# ==========================================
# 2. INIZIALIZZAZIONE RISORSE (CACHING)
# ==========================================
@st.cache_resource
def load_multimodal_resources():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    
    if not gemini_api_key:
        st.error("🔑 Ops! Manca la chiave API di Gemini nel file .env!")
        st.stop()
        
    # Connessione al database vettoriale locale
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME) 
    except Exception:
        st.error(f"⚠️ Non trovo la collezione '{COLLECTION_NAME}' in '{CHROMA_PATH}'. Hai eseguito '3_create_database_multimodal.py'?")
        st.stop()
    
    # Caricamento modello di embedding Alibaba AutoModel
    model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embedding_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    return gemini_api_key, collection, embedding_model

gemini_key, db_collection, embedding_model = load_multimodal_resources()

# ==========================================
# 3. FUNZIONI DI SUPPORTO
# ==========================================
def get_alibaba_text_embedding(text):
    """Trasforma la domanda dell'utente in un vettore con Alibaba AutoModel."""
    with torch.no_grad():
        emb = embedding_model.get_text_embeddings(texts=[text])
        if hasattr(emb, "cpu"):
            emb = emb.cpu().tolist()
        elif hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb[0]

def extract_image_from_pdf(filename, page_num):
    """Estrae la pagina del PDF dal disco e la converte in Base64 per Gemini."""
    filepath = os.path.join(DATA_PATH, filename)
    doc = fitz.open(filepath)
    page = doc.load_page(page_num - 1)  # PyMuPDF usa indici base 0
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    image_bytes = pix.tobytes("jpeg")
    return base64.b64encode(image_bytes).decode("utf-8")

# ==========================================
# 4. PIPELINE MULTIMODALE
# ==========================================
def run_multimodal_pipeline(user_query: str):
    # 1. Vettorizzazione domanda e ricerca top-3 su ChromaDB
    query_vector = get_alibaba_text_embedding(user_query)
    results = db_collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    metadatas = results['metadatas'][0]
    
    # 2. Estrazione dinamica delle immagini dai PDF trovati
    gemini_parts = []
    top_sources = []
    
    for idx, meta in enumerate(metadatas, 1):
        filename = meta.get("source")
        page_num = meta.get("page")
        
        try:
            b64_img = extract_image_from_pdf(filename, page_num)
            top_sources.append({"file": filename, "page": page_num, "b64": b64_img})
            
            # Etichetta e dati immagine per Gemini API
            gemini_parts.append({"text": f"--- IMMAGINE {idx} (Fonte: {filename}, Pagina {page_num}) ---"})
            gemini_parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": b64_img
                }
            })
        except Exception as e:
            print(f"Errore caricamento immagine {filename} pag {page_num}: {e}")

    # 3. Costruzione Prompt per Gemini
    prompt_text = f"""Ciao! Sei un assistente universitario esperto e preciso.
Guarda attentamente le immagini fornite e rispondi alla domanda dello studente basandoti ESCLUSIVAMENTE sulle informazioni, tabelle o schemi presenti in esse.

REGOLE:
1. Cita sempre la fonte specifica per ogni affermazione [Fonte: NOME_FILE, Pagina X].
2. Se l'informazione non è visibile nelle immagini, dichiara che i documenti forniti non contengono la risposta.

DOMANDA DELLO STUDENTE:
{user_query}

RISPOSTA:"""

    gemini_parts.insert(0, {"text": prompt_text})

    # 4. Invocazione API REST Gemini 3.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": gemini_parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    answer_text = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                answer_text = res_data['candidates'][0]['content']['parts'][0]['text']
                break
        except urllib.error.HTTPError as err:
            if err.code == 503 and attempt < 3:
                time.sleep(2 * attempt)
            else:
                return f"⚠️ Errore HTTP {err.code}: {err.reason}", top_sources
        except Exception as err:
            return f"⚠️ Errore durante la chiamata a Gemini: {err}", top_sources

    if not answer_text:
        answer_text = "⚠️ I server di Google sono momentaneamente occupati. Riprova tra qualche secondo!"

    return answer_text, top_sources

# ==========================================
# 5. FRONTEND STREAMLIT
# ==========================================
with st.sidebar:
    st.title("📚 Info Progetto")
    st.markdown("""
    Ciao! 👋 Questo è il mio progetto di tesi. 
    Ho creato un assistente che **"guarda" le pagine PDF** (grafici, tabelle e piani di studio) per fornirti risposte contestualizzate!
    """)
    st.divider()
    st.subheader("Architettura 🛠️")
    st.markdown("""
    * **Embedding Visivo/Testo:** Alibaba Qwen2-VL
    * **Vector DB:** ChromaDB (`multimodal_kb`)
    * **LLM Multimodale:** Gemini 3.5 Flash
    """)
    st.divider()
    if st.button("🧹 Pulisci la chat"):
        st.session_state.messages = []
        st.rerun()

st.title("👩‍🎓 Il tuo tutor universitario AI (Multimodale)")
st.caption("Chiedimi pure informazioni su corsi, regolamenti o sbocchi lavorativi!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🖼️ Visualizza le pagine consultate"):
                cols = st.columns(len(message["sources"]))
                for idx, src in enumerate(message["sources"]):
                    with cols[idx]:
                        st.image(base64.b64decode(src['b64']), caption=f"{src['file']} (Pag. {src['page']})", use_column_width=True)

if user_input := st.chat_input("Es: Quali sono gli sbocchi occupazionali previsti?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Sto consultando le pagine dei documenti visivi... 🔍"):
            answer, sources = run_multimodal_pipeline(user_input)
            
            st.markdown(answer)
            
            if sources:
                with st.expander("🖼️ Visualizza le pagine consultate"):
                    cols = st.columns(len(sources))
                    for idx, src in enumerate(sources):
                        with cols[idx]:
                            st.image(base64.b64decode(src['b64']), caption=f"{src['file']} (Pag. {src['page']})", use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })