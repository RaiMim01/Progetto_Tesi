try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import time
import warnings
import base64
warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import typing

# 1. Patch ForwardRef per compatibilità Python 3.8
if not hasattr(typing.ForwardRef, '__annotations__'):
    typing.ForwardRef.__annotations__ = property(lambda self: {})

from transformers.image_processing_utils import BaseImageProcessor

# 2. Patch su __call__ di BaseImageProcessor
_orig_image_call = BaseImageProcessor.__call__

def _patched_image_call(self, images=None, **kwargs):
    for text_param in ["padding", "max_length", "stride", "truncation", "pad_to_multiple_of"]:
        kwargs.pop(text_param, None)
    return _orig_image_call(self, images=images, **kwargs)

BaseImageProcessor.__call__ = _patched_image_call

print("Avvio dello script RAG Multimodale...", flush=True)

import chromadb
from chromadb.config import Settings
import pymupdf as fitz
import torch
from transformers import AutoModel
from dotenv import load_dotenv
import urllib.request
import json

# ==========================================
# 1. CONFIGURAZIONE API GEMINI E DATABASE
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
if not GEMINI_API_KEY:
    raise ValueError("Nessuna chiave trovata. Controlla il file .env!")

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"
DATA_PATH = "documents"


def get_alibaba_text_embedding(model, text):
    """Calcola l'embedding di una stringa di testo usando Qwen2-VL (Alibaba)."""
    with torch.no_grad():
        emb = model.get_text_embeddings(texts=[text])
        if hasattr(emb, "cpu"):
            emb = emb.cpu().tolist()
        elif hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb[0]


def extract_image_from_pdf(filename, page_num):
    """Estrae un'immagine da un PDF specifico e la converte in Base64 per Gemini."""
    filepath = os.path.join(DATA_PATH, filename)
    doc = fitz.open(filepath)
    # PyMuPDF usa indici base 0, ma noi abbiamo salvato page_id partendo da 1
    page = doc.load_page(page_num - 1) 
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    image_bytes = pix.tobytes("jpeg")
    return base64.b64encode(image_bytes).decode("utf-8")


def main():
    # ==========================================
    # FASE 1: RETRIEVAL (Ricerca per similarità)
    # ==========================================
    print(f"1. Connessione al database vettoriale in '{CHROMA_PATH}'...", flush=True)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    print("2. Caricamento modello di embedding multimodale Alibaba...", flush=True)
    model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embedding_model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )

    user_query = "Mostrami lo schema del piano di studi e le materie del primo anno."
    print(f"\nDomanda dell'utente: '{user_query}'", flush=True)

    print("3. Vettorizzazione della domanda...", flush=True)
    query_vector = get_alibaba_text_embedding(embedding_model, user_query)

    # Chiediamo i 3 risultati più simili (Top-3 è ideale per le immagini con Gemini)
    print("4. Ricerca delle 3 pagine più pertinenti...", flush=True)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3 
    )

    metadatas = results['metadatas'][0]
    
    # ==========================================
    # FASE 2: PREPARAZIONE IMMAGINI PER GEMINI
    # ==========================================
    print("5. Estrazione dinamica delle immagini dai PDF...", flush=True)
    
    gemini_image_parts = []
    
    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        print(f"   -> Recupero immagine {idx}: {source} (Pagina {page})")
        
        base64_img = extract_image_from_pdf(source, page)
        
        # Aggiungiamo un'etichetta testuale per Gemini per fargli capire quale immagine sta guardando
        gemini_image_parts.append({"text": f"--- IMMAGINE {idx} (Fonte: {source}, Pagina {page}) ---"})
        # Aggiungiamo i dati raw dell'immagine
        gemini_image_parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64_img
            }
        })

    # ==========================================
    # FASE 3: GENERAZIONE CON LLM (GEMINI)
    # ==========================================
    print("\n6. Invio richiesta a Gemini 3.5 Flash (Analisi Multimodale in corso...)", flush=True)

    prompt_text = f"""Sei un assistente universitario esperto. Ti ho fornito le immagini delle pagine dei documenti ufficiali più pertinenti alla richiesta dell'utente.
    
    COMPITI:
    1. Guarda attentamente le immagini fornite.
    2. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sulle tabelle, schemi o testi visibili in queste immagini.
    3. Cita sempre la fonte specificando il nome del file e la pagina (es. [Fonte: Regolamento.pdf, Pagina 3]).
    4. Se l'informazione non è presente in nessuna delle immagini, di' chiaramente che i documenti non contengono la risposta.
    
    DOMANDA UTENTE: {user_query}
    RISPOSTA:"""

    # Costruiamo il payload per Gemini: Prompt testuale + Immagini in Base64
    parts = [{"text": prompt_text}] + gemini_image_parts

    # NOTA: Gemini 3.5 Flash è la versione corretta (nella tua API precedente era scritto 3.5 che non esiste)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2}
    }

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            answer_text = res_data['candidates'][0]['content']['parts'][0]['text']
            
            print("\n=========================================================")
            print("🤖 RISPOSTA GENERATA DALL'IA:")
            print("=========================================================")
            print(answer_text)
            print("=========================================================\n")
            
    except urllib.error.HTTPError as err:
        print(f"\nERRORE HTTP {err.code}: {err.read().decode('utf-8')}")
    except Exception as err:
        print(f"\nERRORE: {err}")

if __name__ == "__main__":
    main()