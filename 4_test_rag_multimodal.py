import os
import time
import base64
import jetson_patches

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Avvio dello script in corso...", flush=True)

import chromadb
from chromadb.config import Settings
import pymupdf as fitz
import torch
from transformers import AutoModel
from dotenv import load_dotenv
import urllib.request
import json

# CONFIGURAZIONE GEMINI API KEY E DATABASE

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
if not GEMINI_API_KEY:
    raise ValueError("Nessuna chiave trovata. Controlla il file .env!")

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"
DATA_PATH = "documents"

def get_alibaba_text_embedding(model, text):
    """Calcola l'embedding di una stringa di testo usando il modello multimodale Alibaba Qwen2-VL."""
    # Disabilitazione del calcolo dei gradienti per risparmiare memoria
    with torch.no_grad():
        # Calcolo dell'embedding della query testuale passata
        emb = model.get_text_embeddings(texts=[text])
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


def main():
    # FASE 1: RETRIEVAL (Ricerca per similarità)
    
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
    
    # FASE 2: PREPARAZIONE IMMAGINI PER GEMINI
   
    print("5. Estrazione dinamica delle immagini dai PDF...", flush=True)
    
    # Lista che conterrà un dizionario per ogni immagine con un testo descrittivo e l'immagine in Base64
    gemini_image_parts = []
    
    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        print(f"   -> Recupero immagine {idx}: {source} (Pagina {page})")
        
        # Estrazione dell'immagine dal PDF e conversione in Base64 grazie alla funzione extract_image_from_pdf
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

    # FASE 3: GENERAZIONE RISPOSTA CON LLM (GEMINI)
    
    print("\n6. Invio richiesta a Gemini 3.5 Flash...", flush=True)

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

    # Gemini 3.5 Flash
    # Definizione dell'url per Gemini a cui inviare la richiesta 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Definizione del payload JSON (dizionario) richiesto per la chiamata all'API di Gemini
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2}
    }

    # Preparazione della richiesta HTTP con url, payload e headers (che specifica il tipo di chiamata json)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )

    # Invio della richiesta a Gemini e gestione della risposta
    try:
        # urlopen() invia la richiesta HTTP POST
        # read().decode() legge la risposta e la coverte da byte a stringa formato JSON
        # json.loads() converte la stringa JSON in un dizionario Python
        # La risposta contiene la risposta generata da Gemini, che viene estratta dal dizionario e stampata 
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            answer_text = res_data['candidates'][0]['content']['parts'][0]['text']
            
            print("RISPOSTA GENERATA DA GEMINI:")
            print(answer_text)

    # Il primo livello gestisce gli errori delle API HTTP       
    except urllib.error.HTTPError as err:
        print(f"\nERRORE HTTP {err.code}: {err.read().decode('utf-8')}")
    # Il secondo livello gestisce gli altri errori che non vengono catturati nel primo livello
    except Exception as err:
        print(f"\nERRORE: {err}")

if __name__ == "__main__":
    main()