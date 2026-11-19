import os
import io
import gc
import jetson_patches

# Disabilita l'invio anonimo di dati di utilizzo da parte di ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Disabilita gli avvisi del tokenizer 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import base64
import urllib.request
import urllib.error
import fitz  # PyMuPDF
from PIL import Image
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

# Parametri Edge VLM locale (llama.cpp server su porta 18080)
TOP_K_IMAGES = 1
LLAMA_SERVER_URL = "http://127.0.0.1:18080/v1/chat/completions"

# 1. CONFIGURAZIONE PAGINA

st.set_page_config(
    page_title="Assistente Universitario",
    page_icon="🎓",
    layout="wide"
)

# 2. INIZIALIZZAZIONE RISORSE (CACHING)

@st.cache_resource
def load_multimodal_resources():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    
    if not gemini_api_key:
        st.error("Errore: Chiamata API fallita. Manca GEMINI_API_KEY nel file .env!")
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
        st.error(f"Errore: Collezione '{COLLECTION_NAME}' in '{CHROMA_PATH}' non trovata! Esegui prima: '3_create_database_multimodal.py'")
        st.stop()
    
    # Caricamento modello di embedding Alibaba AutoModel
    model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embedding_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    return gemini_api_key, collection, embedding_model

# Caricamento risorse in memoria
gemini_key, db_collection, embedding_model = load_multimodal_resources()


# 3. FUNZIONI DI SUPPORTO

def get_alibaba_text_embedding(text):
    """Calcola l'embedding di una stringa di testo usando il modello multimodale Alibaba Qwen2-VL."""
    # Disabilitazione del calcolo dei gradienti per risparmiare memoria
    with torch.no_grad():
        #Calcolo dell'embedding della query testuale passata
        emb = embedding_model.get_text_embeddings(texts=[text])
        # Conversione del tensore PyTorch in lista Python per la serializzazione
        if hasattr(emb, "cpu"):
            emb = emb.cpu().tolist()
        elif hasattr(emb, "tolist"):
            emb = emb.tolist()
        # Restituzione dell'embedding
        return emb[0]

def extract_image_from_pdf(filename, page_num):
    """Estrae un'immagine da un PDF specifico e la converte in Base64 per Gemini."""
    # Costruzione del percorso e apertura del file pdf con PyMuPDF
    filepath = os.path.join(DATA_PATH, filename)
    doc = fitz.open(filepath)
    # Recupero della pagina con conversione dell'indice partendo da 0 (PyMuPDF) rispetto a 1 (come salvato nel database)
    page = doc.load_page(page_num - 1)
    # Renderizzazione della pagina in un'immagine (pixmap) con una scala di 1.5x (zoom) per una migliore qualità
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    # Conversione dei pixel in bytes formato JPEG 
    image_bytes = pix.tobytes("jpeg")
    # Doppia conversione: prima i bytes jpeg vengono codificati in Base64, poi convertiti in stringa UTF-8
    return base64.b64encode(image_bytes).decode("utf-8")


# 3B. FUNZIONI DI SUPPORTO PER VLM EDGE (RIDIMENSIONAMENTO PER BUDGET TOKEN VISIVI)

def extract_pil_image_from_pdf(filename, page_num):
    """Renderizza una pagina PDF a 1.5x come oggetto PIL Image per successivo downscaling controllato."""
    filepath = os.path.join(DATA_PATH, filename)
    doc = fitz.open(filepath)
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def pil_image_to_base64(img, max_side=768):
    """Applica downscaling LANCZOS (max 768px) e compressione JPEG 85% per contenere i patch token visivi entro -c 2048."""
    img_copy = img.copy()
    img_copy.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    img_copy.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# 4. PIPELINE MULTIMODALE

def run_multimodal_pipeline(user_query: str):
    start_total = time.time()

    # 1. Vettorizzazione domanda e ricerca top-3 su ChromaDB
    query_vector = get_alibaba_text_embedding(user_query)
    results = db_collection.query(
        query_embeddings=[query_vector],
        n_results=4
    )

    metadatas = results['metadatas'][0]
    
    # 2. Estrazione dinamica delle immagini dai PDF trovati

    # Lista che conterrà un dizionario per ogni immagine con un testo descrittivo e l'immagine in Base64
    gemini_parts = []
    top_sources = []
    
    for idx, meta in enumerate(metadatas, 1):
        filename = meta.get("source")
        page_num = meta.get("page")
        
        try:
            # Estrazione dell'immagine dal PDF e conversione in Base64 grazie alla funzione extract_image_from_pdf
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
    prompt_text = f"""Sei un assistente universitario esperto. Ti ho fornito le immagini delle pagine dei documenti ufficiali più pertinenti alla richiesta dell'utente.
    
    COMPITI:
    1. Guarda attentamente le immagini fornite.
    2. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sulle tabelle, schemi o testi visibili in queste immagini.
    3. Cita sempre la fonte specificando il nome del file e la pagina (es. [Fonte: Regolamento.pdf, Pagina 3]).
    4. Se l'informazione non è presente in nessuna delle immagini, di' chiaramente che i documenti non contengono la risposta.
        
    DOMANDA UTENTE: {user_query}
    RISPOSTA:"""

    # Aggiungo il prompt testuale all'inizio della lista per Gemini
    gemini_parts.insert(0, {"text": prompt_text})

    # 4. Invocazione API REST Gemini 3.5 Flash
    # Definizione dell'url per Gemini a cui inviare la richiesta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={gemini_key}"
    
    # Definizione del payload JSON (dizionario) richiesto per la chiamata all'API di Gemini
    payload = {
        "contents": [{"parts": gemini_parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192, "thinkingConfig": {"thinkingLevel": "LOW"}}
    }

    # Preparazione della richiesta HTTP con url, payload e headers (che specifica il tipo di chiamata json)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # Invio della richiesta a Gemini e gestione della risposta
    answer_text = None
    metrics_info = ""
    for attempt in range(1, 4):
        try:
            start_gen = time.time()
            # urlopen() invia la richiesta HTTP POST
            # read().decode() legge la risposta e la coverte da byte a stringa formato JSON
            # json.loads() converte la stringa JSON in un dizionario Python
            # La risposta contiene la risposta generata da Gemini, che viene estratta dal dizionario e stampata 
            with urllib.request.urlopen(req, timeout=120) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                answer_text = res_data['candidates'][0]['content']['parts'][0]['text']
                finish_reason = res_data['candidates'][0].get('finishReason', 'N/D')
                usage = res_data.get('usageMetadata', {})
                prompt_tokens = usage.get('promptTokenCount', 0)
                completion_tokens = usage.get('candidatesTokenCount', 0)

            gen_time = time.time() - start_gen
            total_time = time.time() - start_total
            tokens_sec = completion_tokens / gen_time if gen_time > 0 else 0.0

            metrics_info = (
                f"Generazione: {gen_time:.2f} s | Totale: {total_time:.2f} s | "
                f"Token: {prompt_tokens} in (visivi+testo) / {completion_tokens} out ({tokens_sec:.2f} tok/s) | Stop: {finish_reason}"
            )
            break

        # Il primo livello gestisce gli errori delle API HTTP
        except urllib.error.HTTPError as err:
            if err.code == 503 and attempt < 3:
                time.sleep(2 * attempt)
            else:
                return f"Errore HTTP {err.code}: {err.reason}", top_sources, metrics_info
        # Il secondo livello gestisce gli altri errori che non vengono catturati nel primo livello
        except Exception as err:
            return f"Errore durante la chiamata a Gemini: {err}", top_sources, metrics_info

    if not answer_text:
        answer_text = "I server di Google sono momentaneamente occupati. Riprova tra qualche secondo!"

    return answer_text, top_sources, metrics_info


# 4B. PIPELINE MULTIMODALE EDGE LOCALE (QWEN2-VL-2B VIA LLAMA.CPP)

def run_edge_multimodal_pipeline(user_query: str):
    start_total = time.time()

    # 1. Vettorizzazione query e retrieval denso Top-K immagini su ChromaDB
    query_vector = get_alibaba_text_embedding(user_query)
    results = db_collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_IMAGES
    )

    # Pulizia buffer temporanei in RAM prima dell'inferenza VLM locale su Jetson
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadatas = results["metadatas"][0]

    # 2. Estrazione pagina PDF, downscaling a 768px e codifica Base64 formato OpenAI Vision
    message_content = []
    top_sources = []

    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        try:
            img = extract_pil_image_from_pdf(source, page)
            img_base64 = pil_image_to_base64(img, max_side=768)
            top_sources.append({"file": source, "page": page, "b64": img_base64})

            message_content.append({
                "type": "text",
                "text": f"--- IMMAGINE {idx} (Fonte reale: {source}, Pagina {page}) ---"
            })
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        except Exception as e:
            print(f"Errore caricamento immagine Edge {source} pag {page}: {e}")

    # 3. Costruzione payload multimodale e chiamata HTTP sincrona a llama-server locale
    fonti_reali = ", ".join([f"{m.get('source')} - Pagina {m.get('page')}" for m in metadatas])

    prompt_text = (
        f"Leggi attentamente il testo e le tabelle nell'immagine o nelle immagini sopra.\n"
        f"Rispondi in italiano in modo diretto, preciso e ben strutturato alla seguente domanda, "
        f"estraendo tutte le informazioni pertinenti (materie, CFU, semestri, requisiti o regole) "
        f"presenti nella pagina o nelle pagine:\n\n"
        f"{user_query}"
    )
    message_content.append({"type": "text", "text": prompt_text})

    # Parametri di decoding calibrati per Qwen2-VL-2B Q4_K_M (penalità ripetizione su elenchi tabellari)
    payload = {
        "model": "qwen2-vl-2b",
        "messages": [{"role": "user", "content": message_content}],
        "max_tokens": 300,
        "temperature": 0.1,
        "repeat_penalty": 1.2,
        "repeat_last_n": 256,
        "presence_penalty": 0.2
    }

    req = urllib.request.Request(
        LLAMA_SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        start_gen = time.time()
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
            answer_text = result["choices"][0]["message"]["content"].strip()
            finish_reason = result["choices"][0].get("finish_reason", "N/D")
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Append deterministico della fonte recuperata dal database vettoriale
            answer_text = f"{answer_text}\n\n[Fonte: {fonti_reali}]"

        gen_time = time.time() - start_gen
        total_time = time.time() - start_total
        tokens_sec = completion_tokens / gen_time if gen_time > 0 else 0.0

        metrics_info = (
            f"Generazione: {gen_time:.2f} s | Totale: {total_time:.2f} s | "
            f"Token: {prompt_tokens} in (visivi+testo) / {completion_tokens} out ({tokens_sec:.2f} tok/s) | Stop: {finish_reason}"
        )
        return answer_text, top_sources, metrics_info

    except urllib.error.HTTPError as err:
        return f"Errore HTTP {err.code} da llama-server: {err.reason}", top_sources, ""
    except urllib.error.URLError as err:
        return (
            f"Impossibile raggiungere `llama-server` su `{LLAMA_SERVER_URL}`. "
            f"Verifica che il server VLM locale (Qwen2-VL-2B) sia avviato sulla porta 18080.\n\nDettaglio: {err.reason}",
            top_sources,
            ""
        )
    except Exception as err:
        return f"Errore durante la chiamata a Qwen2-VL-2B locale: {err}", top_sources, ""


# 5. FRONTEND STREAMLIT (INTERFACCIA MULTIMODALE)

# Sidebar Informativa
with st.sidebar:
    st.title("Info Progetto")
    st.markdown("""
    Applicazione basata su tecnice di rappresentazione vettoriale (embedding) e misure di similarità.\n
    Integra un sistema **RAG (Retrivial-Augmented Generation)** Multimodale Cloud e Edge Locale per rispondere a domande su regolamenti universitari, corsi e sbocchi occupazionali 
    e un **Recommender System Multimodale** per suggerire pagine con strutture o contenuti correlati!
    """)
    st.divider()
    
    st.subheader("Architettura")
    st.markdown("""
- **Modello di Embedding Multimodale:** `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`
- **Vector DB:** ChromaDB (Collection:`multimodal_kb`)
- **Sintesi VLM Cloud:** Gemini 3.5 Flash 
- **Sintesi VLM Edge Locale:** Qwen2-VL-2B GGUF eseguito con llama.cpp
- **Recommender:** Cosine Similarity
    """)
    st.divider()

    # Pulsante per resettare la cronologia della chat
    if st.button("🗑️ Cancella Cronologia Chat"):
        st.session_state.messages = []
        st.session_state.edge_messages = []
        st.rerun()

# Titolo Principale
st.title("Assistente Universitario 🎓")

# Creazione delle Schede (Tabs) dell'Interfaccia (con VLM Edge al centro)
tab_chat, tab_edge, tab_rec = st.tabs([
    "Assistente RAG Gemini Cloud",
    "Assistente RAG Locale Qwen2-VL-2B",
    "Recommender System Multimodale"
])


# TAB 1: SISTEMA RAG CON MODELLO MULTIMODALE (CLOUD)

with tab_chat:
    st.caption("Sistema RAG con Modello di Embedding Multimodale e sintesi Cloud Gemini 3.5 Flash")

    # Inizializzazione della cronologia messaggi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 1. Creazione del contenitore per i messaggi SOPRA la barra di input
    chat_container = st.container()

    # 2. Barra di input posizionata in fondo al Tab (key univoca tra tab)
    user_input = st.chat_input("Fai una domanda sui regolamenti o sui corsi di studio...", key="chat_input_mm_cloud")

    # 3. Rendering della cronologia e dei nuovi messaggi dentro il contenitore superiore
    with chat_container:
        # Mostra la cronologia precedente
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("metrics"):
                    st.caption(f"⏱️ {message['metrics']}")
                if "sources" in message and message["sources"]:
                    with st.expander("Visualizza le pagine consultate"):
                        cols = st.columns(len(message["sources"]))
                        for idx, src in enumerate(message["sources"]):
                            with cols[idx]:
                                st.image(
                                    base64.b64decode(src['b64']), 
                                    caption=f"{src['file']} (Pag. {src['page']})", 
                                    use_container_width=True
                                )

        # Gestione del nuovo messaggio inviato dall'utente
        if user_input:
            # Mostra immediatamente la domanda dell'utente sopra la barra
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Genera e mostra la risposta dell'assistente sopra la barra
            with st.chat_message("assistant"):
                with st.spinner("Sto consultando le pagine dei documenti..."):
                    answer, sources, metrics_info = run_multimodal_pipeline(user_input)
                    st.markdown(answer)
                    if metrics_info:
                        st.caption(f"⏱️ {metrics_info}")
                    
                    if sources:
                        with st.expander("Visualizza le pagine consultate"):
                            cols = st.columns(len(sources))
                            for idx, src in enumerate(sources):
                                with cols[idx]:
                                    st.image(
                                        base64.b64decode(src['b64']), 
                                        caption=f"{src['file']} (Pag. {src['page']})", 
                                        use_container_width=True
                                    )

            # Salva la risposta nella sessione e ricarica l'interfaccia per allinearla
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metrics": metrics_info,
                "sources": sources
            })
            st.rerun()


# TAB 2: SISTEMA RAG MULTIMODALE EDGE LOCALE (QWEN2-VL-2B)

with tab_edge:
    st.caption("Sistema RAG con Modello di Embedding Multimodale e sintesi Locale con Qwen2-VL-2B (GGUF + mmproj via llama.cpp)")

    # Cronologia isolata per la sessione VLM locale
    if "edge_messages" not in st.session_state:
        st.session_state.edge_messages = []

    edge_chat_container = st.container()
    edge_user_input = st.chat_input("Fai una domanda al VLM locale Qwen2-VL-2B...", key="chat_input_mm_edge")

    with edge_chat_container:
        # Rendering storico messaggi locali e telemetria di inferenza
        for message in st.session_state.edge_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("metrics"):
                    st.caption(f"⏱️ {message['metrics']}")
                if "sources" in message and message["sources"]:
                    with st.expander("Visualizza la pagina consultata"):
                        cols = st.columns(len(message["sources"]))
                        for idx, src in enumerate(message["sources"]):
                            with cols[idx]:
                                st.image(
                                    base64.b64decode(src['b64']),
                                    caption=f"{src['file']} (Pag. {src['page']})",
                                    use_container_width=True
                                )

        if edge_user_input:
            st.session_state.edge_messages.append({"role": "user", "content": edge_user_input})
            with st.chat_message("user"):
                st.markdown(edge_user_input)

            # Esecuzione pipeline VLM locale con visione diretta della pagina PDF
            with st.chat_message("assistant"):
                with st.spinner("Elaborazione visiva locale con Qwen2-VL-2B..."):
                    answer, sources, metrics_info = run_edge_multimodal_pipeline(edge_user_input)
                    st.markdown(answer)
                    if metrics_info:
                        st.caption(f"⏱️ {metrics_info}")

                    if sources:
                        with st.expander("Visualizza la pagina consultata"):
                            cols = st.columns(len(sources))
                            for idx, src in enumerate(sources):
                                with cols[idx]:
                                    st.image(
                                        base64.b64decode(src['b64']),
                                        caption=f"{src['file']} (Pag. {src['page']})",
                                        use_container_width=True
                                    )

            st.session_state.edge_messages.append({
                "role": "assistant",
                "content": answer,
                "metrics": metrics_info,
                "sources": sources
            })
            st.rerun()


# TAB 3: RECOMMENDER SYSTEM MULTIMODALE

with tab_rec:
    st.subheader("Raccomandazione delle Pagine PDF")
    st.caption("Seleziona una pagina PDF dal database per visualizzare le 3 pagine visivamente e concettualmente più simili.")

    all_pages = db_collection.get(include=["metadatas"])
    
    if not all_pages["ids"]:
        st.warning("Il database multimodale è vuoto! Esegui prima '3_create_database_multimodal.py'.")
    else:
        options_map = {}
        for page_id, meta in zip(all_pages["ids"], all_pages["metadatas"]):
            label = f"{meta.get('source', 'N/D')} — Pagina {meta.get('page', 'N/D')}"
            options_map[label] = (page_id, meta.get('source'), meta.get('page'))

        selected_label = st.selectbox("Seleziona una pagina PDF di partenza:", list(options_map.keys()))
        selected_id, selected_file, selected_page_num = options_map[selected_label]

        if st.button("🔎 Cerca Pagine Correlate", type="primary"):
            try:
                target_b64 = extract_image_from_pdf(selected_file, selected_page_num)
                
                st.markdown("#### Pagina Selezionata")
                st.image(
                    base64.b64decode(target_b64), 
                    caption=f"TARGET: {selected_file} (Pag. {selected_page_num})", 
                    width=300
                )
                st.divider()
            except Exception as e:
                st.error(f"Impossibile caricare l'immagine target: {e}")

            target_record = db_collection.get(ids=[selected_id], include=["embeddings"])
            target_vector = target_record["embeddings"][0]

            results = db_collection.query(
                query_embeddings=[target_vector],
                n_results=4,
                include=["metadatas", "distances"]
            )

            st.markdown("### Pagine PDF Raccomandate")
            cols = st.columns(3)
            rec_idx = 0

            for meta, dist, rec_id in zip(results['metadatas'][0], results['distances'][0], results['ids'][0]):
                if rec_id == selected_id:
                    continue

                rec_file = meta.get('source')
                rec_page = meta.get('page')

                with cols[rec_idx]:
                    st.markdown(f"##### Raccomandazione #{rec_idx + 1}")
                    st.metric(label="Distanza Coseno", value=f"{dist:.4f}")
                    st.caption(f"**File:** `{rec_file}` (Pag. {rec_page})")
                    
                    try:
                        rec_b64 = extract_image_from_pdf(rec_file, rec_page)
                        st.image(
                            base64.b64decode(rec_b64), 
                            caption=f"{rec_file} - Pag. {rec_page}", 
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Errore caricamento immagine: {e}")
                        
                rec_idx += 1