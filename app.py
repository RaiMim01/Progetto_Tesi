try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import warnings
warnings.filterwarnings("ignore")

import json
import time
import urllib.request
import urllib.error
import torch
import torch.nn as nn
import chromadb
from chromadb.config import Settings
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURAZIONE PAGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="DocFlow AI - Assistente Universitario",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# 2. INIZIALIZZAZIONE RISORSE CON CACHING
# ==========================================
# @st.cache_resource garantisce che i modelli PyTorch e il DB vengano caricati
# in RAM UNA SOLA VOLTA all'avvio, evitando di rallentare la CPU i3 a ogni click.
@st.cache_resource
def load_rag_resources():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    
    if not gemini_api_key:
        st.error("🔑 Errore: Chiamata API fallita. Manca GEMINI_API_KEY nel file .env!")
        st.stop()
        
    # Connessione ChromaDB (Sintassi nativa per ChromaDB 0.3.29 con DuckDB)
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="vector_db",
        anonymized_telemetry=False
    ))
    
    try:
        collection = chroma_client.get_collection(name="knowledge_base")
    except Exception:
        st.error("ERRORE: Impossibile trovare la collezione 'knowledge_base'. Esegui prima '1_create_database.py'.")
        st.stop()
    
    # Modello Bi-Encoder (Fase 1)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Modello Cross-Encoder (Fase 2)
    reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    return gemini_api_key, collection, embedding_model, reranker

# Caricamento risorse in memoria
gemini_api_key, collection, embedding_model, reranker = load_rag_resources()

# ==========================================
# 3. PIPELINE DI RETRIEVAL E SINTESI
# ==========================================
def run_rag_pipeline(user_query: str):
    # FASE 1: Retrieval Bi-Encoder (k=10)
    query_tensor = embedding_model.encode(user_query, convert_to_tensor=True)
    results = collection.query(
        query_embeddings=[query_tensor.cpu().tolist()],
        n_results=10
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    chunk_ids = results['ids'][0]

    # FASE 2: Re-ranking Cross-Encoder
    query_doc_pairs = [[user_query, doc] for doc in documents]
    scores_tensor = reranker.predict(
        query_doc_pairs,
        activation_fct=nn.Sigmoid(),
        convert_to_tensor=True,
        batch_size=8
    )

    ranked_results = []
    for orig_idx, (doc, meta, dist, score, chunk_id) in enumerate(zip(documents, metadatas, distances, scores_tensor, chunk_ids)):
        ranked_results.append({
            'original_rank': orig_idx + 1,
            'chunk_id': chunk_id,
            'doc': doc,
            'metadata': meta,
            'chroma_dist': dist,
            'rerank_score': score.item()
        })

    ranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_6 = ranked_results[:6]

    # FASE 3: Prompt Engineering e LLM (Chiamata REST a Gemini)
    formatted_context = ""
    for idx, item in enumerate(top_6, start=1):
        source = item['metadata'].get('source', 'Sconosciuta')
        page = item['metadata'].get('page', 'Sconosciuta')
        chunk_id = item['chunk_id']
        text = item['doc']
        formatted_context += f"--- DOCUMENTO {idx} (ID Chunk: {chunk_id} | Fonte: {source} | Pag: {page}) ---\n{text}\n\n"

    prompt = f"""Sei un orientatore universitario rigoroso, preciso ed esaustivo.
Il tuo compito è rispondere alla domanda dell'utente basandoti ESCLUSIVAMENTE sui documenti forniti nel CONTESTO sottostante.

REGOLE TASSATIVE DI RISPOSTA:
1. FEDELTÀ TESTUALE ASSOLUTA: Utilizza ESATTAMENTE i nomi delle figure professionali e dei settori presenti nel testo. NON inventare, sintetizzare con sinonimi o parafrasare i titoli.
2. NESSUNA ALLUCINAZIONE: Se un'informazione non è esplicitamente citata nei documenti, NON includerla per alcun motivo.
3. CITAZIONE FONTI: Alla fine di OGNI concetto o ruolo riportato, inserisci la citazione esatta usando il file e la pagina indicati. Formato obbligatorio: [Fonte: NOME_FILE, Pag. X]
4. STRUTTURA: Presenta i ruoli e gli sbocchi in un elenco puntato chiaro e descrittivo.

CONTESTO DEI DOCUMENTI ESTRATTI:
{formatted_context}

DOMANDA DELL'UTENTE:
{user_query}

RISPOSTA:"""

    # Chiamata API REST con retry per sovraccarichi temporanei
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={gemini_api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
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
                return f"⚠️ Errore HTTP {err.code}: {err.reason}", top_6
        except Exception as err:
            return f"⚠️ Errore durante la chiamata a Gemini: {err}", top_6

    if not answer_text:
        answer_text = "⚠️ Server Google temporaneamente occupati. Riprova tra qualche secondo."

    return answer_text, top_6

# ==========================================
# 4. INTERFACCIA UTENTE (STREAMLIT FRONTEND)
# ==========================================

# Sidebar Informativa
with st.sidebar:
    st.title("🎓 DocFlow AI")
    st.subheader("Architettura RAG")
    st.markdown("""
    **Specifiche di Sistema:**
    * **Embedding:** `paraphrase-multilingual-MiniLM-L12-v2`
    * **Vector DB:** ChromaDB 0.3.29 (DuckDB + Parquet)
    * **Re-ranker:** Cross-Encoder PyTorch (`mMiniLMv2-L12`)
    * **Sintesi:** Gemini Flash REST API (`temp=0.1`)
    """)
    st.divider()
    if st.button("🗑️ Cancella Cronologia Chat"):
        st.session_state.messages = []
        st.rerun()

# Titolo Principale
st.title("🏛️ Assistente Virtuale Documenti Universitari")
st.caption("Sistema locale RAG con pipeline a due stadi (Bi-Encoder + Cross-Encoder PyTorch)")

# Inizializzazione della cronologia messaggi in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi della cronologia
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Visualizza Fonti ed Evidenze Re-ranking"):
                for idx, src in enumerate(message["sources"], start=1):
                    st.write(f"**[{idx}] {src['metadata'].get('source', 'N/D')}** — Pagina {src['metadata'].get('page', 'N/D')}")
                    st.caption(f"Score Cross-Encoder: `{src['rerank_score']:.4f}` | Posizione Originale DB: `#{src['original_rank']}`")
                    st.text(src['doc'][:250] + "...")
                    st.divider()

# Input dell'utente
if user_input := st.chat_input("Fai una domanda sui regolamenti o sui corsi di studio..."):
    # Mostra la domanda dell'utente
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Elaborazione della risposta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Ricerca semantica su ChromaDB e Re-ranking PyTorch in corso..."):
            answer, top_sources = run_rag_pipeline(user_input)
            
            # Mostra testo risposta
            st.markdown(answer)
            
            # Mostra le fonti dentro una sezione espandibile
            with st.expander("📚 Visualizza Fonti ed Evidenze Re-ranking"):
                for idx, src in enumerate(top_sources, start=1):
                    st.write(f"**[{idx}] {src['metadata'].get('source', 'N/D')}** — Pagina {src['metadata'].get('page', 'N/D')}")
                    st.caption(f"Score Cross-Encoder: `{src['rerank_score']:.4f}` | Posizione Originale DB: `#{src['original_rank']}`")
                    st.text(src['doc'][:250] + "...")
                    st.divider()

    # Salva la risposta e le fonti in session_state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": top_sources
    })