import os
import warnings
warnings.filterwarnings("ignore")

# Disabilita l'invio anonimo di dati di utilizzo da parte di ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"

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

# 1. CONFIGURAZIONE PAGINA STREAMLIT

st.set_page_config(
    page_title="Assistente Universitario",
    page_icon="🎓",
    layout="wide"
)

# 2. INIZIALIZZAZIONE RISORSE CON CACHING

# @st.cache_resource garantisce che i modelli PyTorch e il DB vengano caricati
# in RAM UNA SOLA VOLTA all'avvio
@st.cache_resource
def load_rag_resources():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    
    if not gemini_api_key:
        st.error("Errore: Chiamata API fallita. Manca GEMINI_API_KEY nel file .env!")
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


# 3. PIPELINE DI RETRIEVAL E SINTESI

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

    # Gemini 3.5 Flash
    # Definizione dell'url per Gemini a cui inviare la richiesta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={gemini_api_key}"
    
    # Definizione del payload JSON (dizionario) richiesto per la chiamata all'API di Gemini
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }

    # Preparazione della richiesta HTTP con url, payload e headers (che specifica il tipo di chiamata json)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # Invio della richiesta a Gemini e gestione della risposta
    answer_text = None
    for attempt in range(1, 4):
        try:
            # urlopen() invia la richiesta HTTP POST
            # read().decode() legge la risposta e la coverte da byte a stringa formato JSON
            # json.loads() converte la stringa JSON in un dizionario Python
            # La risposta contiene la risposta generata da Gemini, che viene estratta dal dizionario e stampata 
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                answer_text = res_data['candidates'][0]['content']['parts'][0]['text']
                break

        # Il primo livello gestisce gli errori delle API HTTP      
        except urllib.error.HTTPError as err:
            if err.code == 503 and attempt < 3:
                time.sleep(2 * attempt)
            else:
                return f"Errore HTTP {err.code}: {err.reason}", top_6
        # Il secondo livello gestisce gli altri errori che non vengono catturati nel primo livello
        except Exception as err:
            return f"Errore durante la chiamata a Gemini: {err}", top_6

    if not answer_text:
        answer_text = "Server Google temporaneamente occupati. Riprova tra qualche secondo."

    return answer_text, top_6


# 4. INTERFACCIA UTENTE (STREAMLIT FRONTEND)

# Sidebar Informativa
with st.sidebar:
    st.title("Info Progetto")
    st.markdown("""
    Applicazione basata su tecnice di rappresentazione vettoriale (embedding) e misure di similarità.\n
    Integra un sistema **RAG (Retrivial-Augmented Generation)** per rispondere a domande su regolamenti universitari, corsi e sbocchi occupazionali 
    e un **Recommender System Testuale** per suggerire frammenti di testo correlati!
    """)
    st.divider()
    
    st.subheader("Architettura")
    st.markdown("""
    * **Modello di Embedding Testuale (Bi-Encoder):** `paraphrase-multilingual-MiniLM-L12-v2`
    * **Vector DB:** ChromaDB (Collection:`knowledge_base`)
    * **Re-ranking con Cross-Encoder:** `mMiniLMv2-L12`
    * **Sintesi LLM:** Gemini 3.5 Flash
    * **Recommender:** Cosine Similarity
    """)
    st.divider()
    
    # Pulsante per resettare la cronologia della chat
    if st.button("🗑️ Cancella Cronologia Chat"):
        st.session_state.messages = []
        st.rerun()

# Titolo Principale
st.title("Assistente Universitario 🎓")

# Creazione delle Schede (Tabs) dell'Interfaccia
tab_chat, tab_rec = st.tabs(["Assistente RAG", "Recommender System Testuale"])


# TAB 1: SISTEMA RAG CON RE-RANKING

with tab_chat:
    st.caption("Sistema locale RAG con modelli di Embedding Testuali Bi-Encoder e Cross-Encoder")

    # Inizializzazione della cronologia messaggi in session_state se non presente
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 1. Creazione del contenitore per i messaggi SOPRA la barra di input
    chat_container = st.container()

    # 2. Barra di input posizionata sempre in fondo al Tab
    user_input = st.chat_input("Fai una domanda sui regolamenti o sui corsi di studio...")

    # 3. Rendering della cronologia e dei nuovi messaggi dentro il contenitore superiore
    with chat_container:
        # Rendering della cronologia dei messaggi precedentemente inviati
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("Visualizza Fonti e Punteggi Similarità"):
                        for idx, src in enumerate(message["sources"], start=1):
                            st.write(f"**[{idx}] {src['metadata'].get('source', 'N/D')}** — Pagina {src['metadata'].get('page', 'N/D')}")
                            st.caption(f"Score Cross-Encoder: `{src['rerank_score']:.4f}` | Posizione Originale DB: `#{src['original_rank']}`")
                            st.text(src['doc'][:250] + "...")
                            st.divider()

        # Input utente e gestione del flusso di richiesta/risposta
        if user_input:
            # 1. Registrazione e visualizzazione immediata della domanda dell'utente
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 2. Elaborazione ed esecuzione della pipeline RAG a due stadi
            with st.chat_message("assistant"):
                with st.spinner("Sto consultando i documenti testuali... 🔍"):
                    answer, top_sources = run_rag_pipeline(user_input)
                    
                    # Visualizzazione della risposta generata dall'LLM
                    st.markdown(answer)
                    
                    # Visualizzazione espandibile con l'evidenza delle fonti e del re-ranking
                    with st.expander("Visualizza Fonti e Punteggi Similarità"):
                        for idx, src in enumerate(top_sources, start=1):
                            st.write(f"**[{idx}] {src['metadata'].get('source', 'N/D')}** — Pagina {src['metadata'].get('page', 'N/D')}")
                            st.caption(f"Score Cross-Encoder: `{src['rerank_score']:.4f}` | Posizione Originale DB: `#{src['original_rank']}`")
                            st.text(src['doc'][:250] + "...")
                            st.divider()

            # 3. Salvataggio della risposta in session_state e rerun dell'interfaccia per allinearla
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": top_sources
            })
            st.rerun()


# TAB 2: RECOMMENDER SYSTEM TESTUALE

with tab_rec:
    st.subheader("Raccomandazione Documenti Testuali")
    st.caption("Seleziona un frammento di testo per trovare i 3 contenuti più simili presenti nel database vettoriale.")

    # 1. Recupero di tutti i chunk di testo censiti nel DB vettoriale
    all_docs = collection.get(include=["metadatas", "documents"])
    
    if not all_docs["ids"]:
        st.warning("Il database vettoriale è vuoto! Esegui prima lo script '1_create_database.py'.")
    else:
        # Creazione di un dizionario per popolare il menu a tendina (Etichetta leggibile -> ID record DB)
        options_map = {}
        for doc_id, meta, text in zip(all_docs["ids"], all_docs["metadatas"], all_docs["documents"]):
            label = f"{meta.get('source', 'N/D')} (Pag. {meta.get('page', 'N/D')}) - {text[:60]}..."
            options_map[label] = doc_id

        # Menu a tendina per selezionare il chunk target di partenza
        selected_label = st.selectbox("Seleziona un documento/chunk di partenza:", list(options_map.keys()))
        selected_id = options_map[selected_label]

        # Avvio del calcolo delle raccomandazioni al click sul pulsante
        if st.button("🔍 Cerca Documenti Correlati", type="primary"):
            # 2. Recupero dell'embedding vettoriale già calcolato e salvato nel DB
            target_record = collection.get(ids=[selected_id], include=["embeddings", "documents", "metadatas"])
            target_vector = target_record["embeddings"][0]
            
            # Anteprima visiva del documento target selezionato dall'utente
            st.info(f"**Documento Selezionato:** {target_record['metadatas'][0].get('source')} (Pagina {target_record['metadatas'][0].get('page')})\n\n_{target_record['documents'][0]}_")
            st.divider()

            # 3. Query vettoriale per similarità coseno su ChromaDB (n_results=4 per poter escludere il documento stesso)
            results = collection.query(
                query_embeddings=[target_vector],
                n_results=4,
                include=["documents", "metadatas", "distances"]
            )

            # 4. Rendering delle 3 schede raccomandate distribuite su 3 colonne affiancate
            st.markdown("### Documenti Raccomandati")
            cols = st.columns(3)
            rec_idx = 0
            
            for doc, meta, dist, rec_id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
                # Esclusione del documento target dai risultati della raccomandazione
                if rec_id == selected_id:
                    continue 
                
                with cols[rec_idx]:
                    st.markdown(f"##### Raccomandazione #{rec_idx + 1}")
                    st.write(f"**Fonte:** {meta.get('source', 'N/D')}")
                    st.write(f"**Pagina:** {meta.get('page', 'N/D')}")
                    st.metric(label="Distanza Coseno", value=f"{dist:.4f}")
                    st.caption(f"**ID Chunk:** `{rec_id}`")
                    st.text_area("Estratto testo:", value=doc, height=180, disabled=True, key=f"rec_txt_{rec_id}")
                rec_idx += 1